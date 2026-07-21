#!/usr/bin/env bash
# collect-gpu-diag.sh — one-shot diagnostic bundle for a GPU that fails to plot.
#
# Runs a fixed sequence of probes and two instrumented plot attempts, and writes
# everything to a single file. Nothing is uploaded; nothing is deleted. Intended
# to be run once and the resulting file sent to whoever is debugging.
#
#   ./scripts/collect-gpu-diag.sh [--devices N] [--k 28]
#
# Takes roughly 5-10 minutes. Safe to Ctrl-C; partial output is still useful.
#
# The k=28 attempt is run inside a memory-limited scope where systemd allows it,
# so that a host that runs out of RAM kills the plotter instead of the desktop
# session. That containment is also the experiment: if the plot dies at the cap
# and completes without it, the problem is host memory, not the GPU.

set -u

DEVICES=""
K=28
while [ $# -gt 0 ]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --k)       K="$2";       shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

OUT="xchplot2-diag-$(date +%Y%m%d-%H%M%S).txt"
# Repo root relative to this script, so the local build is found no matter
# which directory the script is invoked from.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BIN="$REPO/target/release/xchplot2"
BIN="$(command -v xchplot2 || echo "$LOCAL_BIN")"
DEVARG=""
[ -n "$DEVICES" ] && DEVARG="--devices $DEVICES"

exec > >(tee "$OUT") 2>&1

sec() { printf '\n\n========== %s ==========\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
# Never let one probe's failure end the bundle.
try() { "$@" 2>&1 || echo "  (command failed: $*)"; }

echo "xchplot2 diagnostic bundle"
echo "date: $(date -Is)"
echo "host: $(uname -a)"

sec "1. BINARY PROVENANCE  (is the build actually current?)"
cat <<'EOM'
The most common cause of "I rebuilt and nothing changed":

    cargo install --path .   ->  ~/.cargo/bin/xchplot2
    cargo build --release    ->  ./target/release/xchplot2

Those are DIFFERENT FILES. A rebuild does not update an installed copy, and a
bare `xchplot2` runs whichever one is on PATH. To update an installed copy:

    cargo install --path . --force

EOM

check_bin() {
    b="$1"; tag="$2"
    if [ -z "$b" ] || [ ! -x "$b" ]; then
        printf '  %-13s %s\n' "$tag" "(absent)"
        return
    fi
    printf '  %-13s %s\n' "$tag" "$b"
    printf '  %-13s mtime %s   size %s\n' "" \
        "$(date -r "$b" '+%Y-%m-%d %H:%M' 2>/dev/null)" \
        "$(stat -c %s "$b" 2>/dev/null)"
    for s in "XCHPLOT2_HOST_RESERVE_MB:host-RAM gate" \
             "pinned host allocation of:pinned-alloc chokepoint" \
             "asynchronous backend error:device-fault attribution" \
             "plot compression failed after:writer attribution"; do
        pat="${s%%:*}"; label="${s#*:}"
        if strings "$b" 2>/dev/null | grep -qF "$pat"; then
            printf '  %-13s   PRESENT  %s\n' "" "$label"
        else
            printf '  %-13s   MISSING  %s\n' "" "$label"
        fi
    done
}
PATH_BIN="$(command -v xchplot2 2>/dev/null || true)"
check_bin "$PATH_BIN"   "on PATH:"
check_bin "$LOCAL_BIN"  "local build:"
echo
if [ -n "$PATH_BIN" ] && [ -x "$LOCAL_BIN" ] \
   && ! cmp -s "$PATH_BIN" "$LOCAL_BIN"; then
    echo "  >>> THE TWO BINARIES DIFFER. Everything below used: $BIN"
    echo "  >>> If that is the stale one, this bundle describes the OLD code."
fi
echo
if [ -d .git ]; then
    echo "repo HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
    echo "repo dirty: $(git status --porcelain 2>/dev/null | wc -l) file(s)"
fi

sec "2. HOST MEMORY"
try free -h
echo; try swapon --show
echo; grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|Committed_AS|Dirty' /proc/meminfo
echo; echo "top 8 RSS consumers:"
try ps -eo rss,comm --sort=-rss --no-headers
echo

sec "3. GPU / DRIVER ENVIRONMENT"
for c in /sys/class/drm/card*; do
    [ -e "$c/device/vendor" ] || continue
    drv=$(basename "$(readlink -f "$c/device/driver" 2>/dev/null)" 2>/dev/null)
    printf '%s: vendor=%s device=%s driver=%s\n' \
        "$(basename "$c")" \
        "$(cat "$c/device/vendor" 2>/dev/null)" \
        "$(cat "$c/device/device" 2>/dev/null)" \
        "${drv:-unknown}"
done
echo
echo "-- GPU engine timeouts (a kernel outrunning these is reset by the driver) --"
found_to=0
for f in /sys/class/drm/card*/device/tile*/gt*/engines/*/job_timeout_ms \
         /sys/class/drm/card*/device/tile*/gt*/engines/*/preempt_timeout_us \
         /sys/class/drm/card*/device/preempt_timeout_ms \
         /sys/class/drm/card*/device/enable_hangcheck; do
    [ -r "$f" ] || continue
    found_to=1
    printf '  %s = %s\n' "$f" "$(cat "$f" 2>/dev/null)"
done
[ "$found_to" = 0 ] && echo "  (no engine timeout knobs readable)"
echo
have acpp-info && { echo "-- acpp-info --"; try acpp-info; } || echo "(acpp-info not on PATH)"
echo
have clinfo && try clinfo -l || true
for v in ZES_ENABLE_SYSMAN NEOReadDebugKeys ACPP_TARGETS ACPP_VISIBILITY_MASK \
         XCHPLOT2_HOST_RESERVE_MB; do
    echo "env $v=${!v-<unset>}"
done

sec "4. DMESG BASELINE (before any plotting)"
try sh -c 'dmesg -T 2>/dev/null | tail -40 || sudo -n dmesg -T 2>/dev/null | tail -40'
DMESG_MARK=$(date +%s)

# --- instrumented run helper -------------------------------------------------
# Samples vmstat and peak RSS around a plot attempt. $1=label, rest=extra args.
run_probe() {
    label="$1"; shift
    sec "$label"
    echo "command: $BIN bench -k $* $DEVARG -n 1"
    vmlog=$(mktemp); rsslog=$(mktemp)
    have vmstat && (vmstat 1 > "$vmlog" 2>&1 &  echo $! > "$vmlog.pid")

    start=$(date +%s.%N)
    # shellcheck disable=SC2086
    $BIN bench -k $* $DEVARG -n 1 &
    pid=$!
    hwm=0
    while kill -0 "$pid" 2>/dev/null; do
        v=$(awk '/^VmHWM:/{print $2}' "/proc/$pid/status" 2>/dev/null)
        [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
        sleep 0.2
    done
    wait "$pid"; rc=$?
    end=$(date +%s.%N)

    [ -f "$vmlog.pid" ] && kill "$(cat "$vmlog.pid")" 2>/dev/null
    echo
    echo "exit code: $rc"
    awk -v s="$start" -v e="$end" 'BEGIN{ printf "wall: %.2f s\n", e-s }'
    awk -v h="$hwm" 'BEGIN{ printf "peak RSS: %.2f GiB\n", h/1048576 }'
    if [ -s "$vmlog" ]; then
        echo "-- vmstat: si/so nonzero = swapping; high sy = kernel-bound --"
        head -3 "$vmlog"
        awk 'NR>2{print}' "$vmlog" | awk '{print}' | tail -25
    fi
    rm -f "$vmlog" "$vmlog.pid" "$rsslog"
}

sec "5. CONTROL — small k (short kernels, small footprint)"
echo "If this SUCCEEDS and k=$K fails, the failure is scale-dependent."
run_probe "5a. k=22 run" 22

sec "6. TARGET — k=$K, memory-capped"
echo "Capped so an out-of-RAM host kills the plotter, not your session."
if have systemd-run && systemd-run --user --scope true >/dev/null 2>&1; then
    cap_gib=$(awk '/MemTotal/{printf "%d", ($2/1048576)*0.75}' /proc/meminfo)
    echo "running under systemd scope, MemoryMax=${cap_gib}G, swap disabled"
    echo "  -> killed at the cap = host memory is the constraint"
    echo "  -> same failure well under the cap = not memory"
    # shellcheck disable=SC2086
    try systemd-run --user --scope -q \
        -p MemoryMax=${cap_gib}G -p MemorySwapMax=0 \
        $BIN bench -k $K $DEVARG -n 1
else
    echo "(systemd-run --user unavailable; running uncapped)"
fi

run_probe "6b. k=$K run, uncapped + instrumented" "$K"

sec "7. DMESG (GPU resets, hangcheck, OOM kills all land here)"
dmesg_out=$(dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null \
            || dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null || true)
if [ -z "$dmesg_out" ]; then
    cat <<'EOM'
  UNREADABLE — kernel.dmesg_restrict is set and passwordless sudo is not
  available. This section is important: a GPU engine reset or an OOM kill
  appears ONLY here. Please re-run just this part and include the output:

      sudo dmesg -T | tail -100
      sudo dmesg -T | grep -iE 'xe |i915|drm|reset|hang|oom|killed process'
EOM
else
    echo "-- last 60 lines --"
    printf '%s\n' "$dmesg_out" | tail -60
    echo
    echo "-- lines matching reset/hang/oom/gpu --"
    printf '%s\n' "$dmesg_out" \
        | grep -iE 'xe |i915|drm|reset|hang|oom|killed process|GPU' | tail -40 \
        || echo "  (none — no GPU reset and no OOM kill was logged)"
fi
: "$DMESG_MARK"

sec "8. PARITY TESTS (settles codegen: these run tiny, no memory pressure)"
ran_any=0
for t in sycl_g_x_parity sycl_sort_parity sycl_bucket_offsets_parity sycl_t1_parity; do
    for p in "./build/$t" "./$t"; do
        if [ -x "$p" ]; then
            ran_any=1
            printf '%-30s ' "$t"
            if timeout 300 "$p" >/dev/null 2>&1; then echo PASS; else echo "FAIL/ERROR"; fi
            break
        fi
    done
done
[ "$ran_any" = 0 ] && cat <<'EOM'
(not built. To build them:
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j --target sycl_g_x_parity sycl_sort_parity \
         sycl_bucket_offsets_parity sycl_t1_parity
 If these PASS while k=28 fails, the kernels are correct and the fault is
 environmental — driver, timeout, or memory.)
EOM

sec "DONE"
echo "Bundle written to: $OUT"
