#!/usr/bin/env bash
# collect-gpu-diag.sh — one-shot diagnostic bundle for a GPU that fails to plot.
#
# Runs a fixed sequence of probes and writes everything to a single file.
# Nothing is uploaded; nothing is deleted. Run it once, send the file.
#
#   ./scripts/collect-gpu-diag.sh [--devices N] [--k 28] [--no-build]
#
# With --no-build, minutes. A full run is dominated by compiling parity tests.
# Every step is bounded (see PROBE_LIMIT / PARITY_LIMIT below), so it cannot
# hang indefinitely on a wedged GPU. Safe to Ctrl-C;
# partial output is still useful.
#
# Home directory paths are redacted from the output, because these bundles get
# pasted into issue trackers.
#
# STRUCTURE — the small-k probe BRANCHES the run:
#
#   small k passes, target k fails  -> scale-dependent. Chase memory, VRAM
#                                      tiers, allocation size, timeouts.
#   small k also fails              -> not scale. Everything about capacity is
#                                      a dead end; the kernels are wrong or the
#                                      toolchain is. Go straight to parity.
#
# The first version of this script assumed the first branch and hardcoded k=22
# as a "control" that passes. On the report it was written for, k=22 failed
# too, and the bundle spent its time measuring memory on a machine whose GPU
# could not run a 34 MB workload. Do not assume the branch. Test it.

set -u

DEVICES=""
K=28
SMALL_K=22
DO_BUILD=1
# Per-step ceilings. A diagnostic that outlives the attention of the person
# running it collects nothing. These are deliberately tight: on a healthy host
# every one of these steps finishes in seconds.
PROBE_LIMIT=${PROBE_LIMIT:-420}      # one plot attempt
PARITY_LIMIT=${PARITY_LIMIT:-240}    # one parity binary
while [ $# -gt 0 ]; do
    case "$1" in
        --devices)  DEVICES="$2"; shift 2 ;;
        --k)        K="$2";       shift 2 ;;
        --small-k)  SMALL_K="$2"; shift 2 ;;
        --no-build) DO_BUILD=0;   shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

OUT="xchplot2-diag-$(date +%Y%m%d-%H%M%S).txt"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BIN="$REPO/target/release/xchplot2"
BIN="$(command -v xchplot2 || echo "$LOCAL_BIN")"
DEVARG=""
[ -n "$DEVICES" ] && DEVARG="--devices $DEVICES"
BUILD_DIR="$REPO/build-diag"

# Redact identifying paths on the way to the file. These bundles are meant to
# be shared, and a home directory carries a real name more often than not.
ME="$(id -un 2>/dev/null || echo __nouser__)"
exec > >(sed -E "s#${HOME}#~#g; s#/home/${ME}#~#g; s#\\b${ME}\\b#<user>#g" \
         | tee "$OUT") 2>&1

sec()  { printf '\n\n========== %s ==========\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
try()  { "$@" 2>&1 || echo "  (command failed: $*)"; }

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
if [ -n "$PATH_BIN" ] && [ -x "$LOCAL_BIN" ] && ! cmp -s "$PATH_BIN" "$LOCAL_BIN"; then
    echo "  >>> THE TWO BINARIES DIFFER. Everything below used: $BIN"
    echo "  >>> If that is the stale one, this bundle describes the OLD code."
fi
echo
if [ -d "$REPO/.git" ]; then
    echo "repo HEAD:  $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
    echo "repo dirty: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) file(s)"

    # A bundle from a tree that predates the fix it is meant to test reads like
    # a verdict and is not one. Name the missing commits here, at the top, where
    # it cannot be missed. Bounded and non-fatal -- never block on the network.
    upstream=$(git -C "$REPO" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)
    if [ -n "$upstream" ]; then
        timeout 30 git -C "$REPO" fetch --quiet 2>/dev/null \
            || echo "            (remote unreachable; comparing against the last fetch)"
        behind=$(git -C "$REPO" rev-list --count "HEAD..$upstream" 2>/dev/null || echo 0)
        if [ "${behind:-0}" -gt 0 ]; then
            echo
            echo "  >>> THIS TREE IS $behind COMMIT(S) BEHIND $upstream:"
            git -C "$REPO" log --oneline "HEAD..$upstream" 2>/dev/null | sed 's/^/  >>>     /'
            echo "  >>> If you are testing a fix, it is NOT in this build. Update both"
            echo "  >>> copies -- an installed binary and a build dir are separate files:"
            echo "  >>>     git pull && cargo install --path . --force"
            echo "  >>>     rm -rf $BUILD_DIR   # force the parity build to reconfigure"
        fi
    fi
fi

sec "2. HOST MEMORY"
try free -h
echo; try swapon --show
echo; grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|Committed_AS|Dirty' /proc/meminfo

sec "3. GPU / DRIVER ENVIRONMENT"
for c in /sys/class/drm/card*; do
    [ -e "$c/device/vendor" ] || continue
    case "$(basename "$c")" in *-*) continue ;; esac
    drv=$(basename "$(readlink -f "$c/device/driver" 2>/dev/null)" 2>/dev/null)
    printf '%s: vendor=%s device=%s driver=%s\n' \
        "$(basename "$c")" \
        "$(cat "$c/device/vendor" 2>/dev/null)" \
        "$(cat "$c/device/device" 2>/dev/null)" \
        "${drv:-unknown}"
done
echo
echo "-- GPU engine timeouts --"
found_to=0
for f in /sys/class/drm/card*/device/tile*/gt*/engines/*/job_timeout_ms \
         /sys/class/drm/card*/device/tile*/gt*/engines/*/preempt_timeout_us \
         /sys/class/drm/card*/device/preempt_timeout_ms; do
    [ -r "$f" ] || continue
    found_to=1
    printf '  %s = %s\n' "$f" "$(cat "$f" 2>/dev/null)"
done
[ "$found_to" = 0 ] && echo "  (none readable)"
echo
# acpp-info is usually NOT on PATH -- AdaptiveCpp installs to a prefix that
# only the build system knows about. Look where it actually lands.
echo "-- acpp-info (device + backend enumeration) --"
ACPP_INFO=""
for p in "$(command -v acpp-info 2>/dev/null || true)" \
         /opt/adaptivecpp/bin/acpp-info /usr/local/bin/acpp-info \
         /usr/lib/AdaptiveCpp/bin/acpp-info "$HOME/.local/bin/acpp-info"; do
    if [ -n "$p" ] && [ -x "$p" ]; then ACPP_INFO="$p"; break; fi
done
if [ -n "$ACPP_INFO" ]; then
    echo "($ACPP_INFO)"
    try timeout 120 "$ACPP_INFO"
else
    echo "  NOT FOUND. This matters: it is the only thing that says how many"
    echo "  devices AdaptiveCpp sees and which backends are loaded. If you"
    echo "  know your AdaptiveCpp prefix, run <prefix>/bin/acpp-info by hand."
fi
echo
# Which backends are even installed. These names are also the tokens
# ACPP_VISIBILITY_MASK accepts (semicolon-separated), so this doubles as the
# reference for masking a backend out.
echo "-- installed AdaptiveCpp backends --"
found_be=0
for d in /opt/adaptivecpp/lib/hipSYCL /usr/lib/hipSYCL /usr/local/lib/hipSYCL \
         /usr/lib64/hipSYCL; do
    for so in "$d"/librt-backend-*.so; do
        [ -e "$so" ] || continue
        found_be=1
        printf '  %s\n' "$(basename "$so" | sed 's/librt-backend-//; s/\.so$//')"
    done
done
[ "$found_be" = 0 ] && echo "  (none found at the usual prefixes)"
echo
# THE device list, as xchplot2 itself sees it. Without this the bundle cannot
# say which device a run actually used -- and --devices takes an INDEX, not a
# count, so `--devices 1` means "the device at index 1", which is not
# necessarily the GPU under investigation. AdaptiveCpp's HIP backend has been
# observed enumerating a ROCm CPU agent as a GPU on hosts whose real AMD GPU
# is unsupported, which shifts every later index by one.
echo "-- xchplot2 devices (indices here are what --devices takes) --"
try timeout 120 "$BIN" devices
echo
for v in ZES_ENABLE_SYSMAN NEOReadDebugKeys ACPP_TARGETS ACPP_VISIBILITY_MASK \
         ACPP_ADAPTIVITY_LEVEL ACPP_DEBUG_LEVEL XCHPLOT2_HOST_RESERVE_MB; do
    echo "env $v=${!v-<unset>}"
done

# Kernel log. dmesg is root-only under kernel.dmesg_restrict on most distros
# now, but journalctl -k is frequently readable by a normal user, and a GPU
# page fault or engine reset appears ONLY here.
klog() {
    dmesg -T 2>/dev/null \
        || journalctl -k --no-pager -q 2>/dev/null \
        || sudo -n dmesg -T 2>/dev/null \
        || true
}

sec "4. KERNEL LOG BASELINE (before any plotting)"
kb="$(klog)"
if [ -z "$kb" ]; then
    echo "  UNREADABLE via both dmesg and journalctl -k."
else
    printf '%s\n' "$kb" | tail -25
fi

# --- plot probe ---------------------------------------------------------
# Returns 0 on success. Samples peak RSS so a memory story can be confirmed
# or (more usefully) ruled out.
PROBE_RC=0
# A hang and a crash both exit non-zero, and "it hung" narrows nothing. These
# two need opposite fixes and only a live sample can tell them apart:
#
#   burning CPU  -> host-side spin, or a JIT that never finishes. No GPU work
#                   is in flight, which is why the kernel log stays silent.
#   blocked      -> a submission or event that never completes. The device is
#                   idle and nobody is going to wake us.
#
# Sample BEFORE killing: once the process is gone this is unrecoverable, and
# a rerun of a nondeterministic hang may not reproduce it.
probe_hung_process() {
    hp="$1"
    hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
    c1=$(awk '{print $14+$15}' "/proc/$hp/stat" 2>/dev/null || echo "")
    sleep 3
    c2=$(awk '{print $14+$15}' "/proc/$hp/stat" 2>/dev/null || echo "")
    echo "     --- what the hung process was doing ---"
    if [ -n "$c1" ] && [ -n "$c2" ]; then
        d=$((c2 - c1))
        echo "     CPU: $d ticks in 3s (one fully busy core = $((hz * 3)))"
        if   [ "$d" -gt $((hz * 2)) ]; then
            echo "     => BURNING CPU. Host-side spin or JIT; no kernel is stuck."
        elif [ "$d" -lt $((hz / 5)) ]; then
            echo "     => BLOCKED, device idle. Waiting on something that never lands."
        else
            echo "     => partially busy."
        fi
    fi
    echo "     state:   $(awk '/^State:/{$1=""; print}' "/proc/$hp/status" 2>/dev/null)"
    echo "     threads: $(awk '/^Threads:/{print $2}' "/proc/$hp/status" 2>/dev/null)"
    echo "     wchan:   $(tr -d '\0' < "/proc/$hp/wchan" 2>/dev/null)"
    for tool in eu-stack pstack; do
        command -v "$tool" >/dev/null 2>&1 || continue
        echo "     --- $tool (install elfutils/gdb if absent) ---"
        timeout 30 "$tool" -p "$hp" 2>/dev/null | head -40 | sed 's/^/     /'
        break
    done
}

run_probe() {
    label="$1"; kk="$2"
    sec "$label"
    echo "command: $BIN bench -k $kk $DEVARG -n 1 --warmup 0"
    plog=$(mktemp)
    start=$(date +%s.%N)
    # Redirect to a file rather than piping: $! after a pipeline is the LAST
    # element's pid, so both the RSS sampling below and the exit status would
    # describe `tail` instead of the plotter -- and a probe that always
    # reports success would silently disable the branch in section 5.
    # shellcheck disable=SC2086
    $BIN bench -k "$kk" $DEVARG -n 1 --warmup 0 > "$plog" 2>&1 &
    pid=$!
    hwm=0
    # Watchdog. The plot cannot simply be wrapped in `timeout`, because then
    # $! is timeout's pid and the RSS sampling below would measure the wrong
    # process -- so the deadline is enforced here instead. It has to exist: a
    # GPU that has been reset out from under its context can leave a submission
    # waiting forever, and an unbounded probe turns a 10-minute diagnostic into
    # an overnight one that collects nothing.
    ticks=0
    probe_killed=0
    while kill -0 "$pid" 2>/dev/null; do
        v=$(awk '/^VmHWM:/{print $2}' "/proc/$pid/status" 2>/dev/null)
        [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
        sleep 0.2
        ticks=$((ticks + 1))
        if [ "$ticks" -gt $((PROBE_LIMIT * 5)) ]; then
            echo "  !! no exit after ${PROBE_LIMIT}s -- killing it."
            echo "     A hang here is itself a finding: report it."
            probe_hung_process "$pid"
            kill -9 "$pid" 2>/dev/null
            probe_killed=1
            break
        fi
    done
    wait "$pid" 2>/dev/null; PROBE_RC=$?
    [ "$probe_killed" = 1 ] && PROBE_RC=124
    end=$(date +%s.%N)
    # A dying Level Zero context emits the same submit failure hundreds of
    # times. Summarise by shape first, then show the head verbatim -- the
    # first few lines are where the actual first failure is.
    echo "-- output by message shape (digits collapsed) --"
    sed -E 's/[0-9]+/#/g; s/0x[0-9a-fA-F]+/0xH/g' "$plog" \
        | sort | uniq -c | sort -rn | head -12 | sed 's/^/  /'
    echo "-- first 30 lines verbatim --"
    head -30 "$plog" | sed 's/^/  /'
    rm -f "$plog"
    echo
    echo "exit code: $PROBE_RC"
    awk -v s="$start" -v e="$end" 'BEGIN{ printf "wall: %.2f s\n", e-s }'
    awk -v h="$hwm" 'BEGIN{ printf "peak RSS: %.2f GiB\n", h/1048576 }'
    return $PROBE_RC
}

sec "5. SMALL-k PROBE — k=$SMALL_K  (this decides what the rest means)"
run_probe "5a. k=$SMALL_K" "$SMALL_K"
SMALL_OK=$PROBE_RC
echo
if [ "$SMALL_OK" = 0 ]; then
    echo ">>> SMALL k PASSED. The failure is scale-dependent; k=$K is worth"
    echo ">>> running, and memory/VRAM/tier explanations are live."
else
    echo ">>> SMALL k FAILED, at a few tens of MB on any modern card."
    echo ">>> The failure is NOT about capacity. Host RAM, VRAM tiers, buffer"
    echo ">>> sizes and the k=28 working set are all ruled out. Skipping the"
    echo ">>> k=$K run -- it cannot add information -- and going to parity."
fi

sec "6. PARITY TESTS — do the kernels compute the right answers?"
cat <<'EOM'
Read the labels. The suite is NOT uniformly probative:

  [ORACLE]  compares GPU output against an INDEPENDENT CPU implementation.
            A failure here names a broken kernel. This is real evidence.

  [SELF]    compares one GPU path against ANOTHER GPU path (sharded vs
            single-device, scatter vs gather). If a kernel is systematically
            wrong, both sides are wrong in the same way and the test PASSES.
            A green [SELF] test proves almost nothing about correctness.

So: a fully green run does NOT exonerate the match kernels. T2 match, T3 match
and the proof-fragment values have CPU ground truth only in the .cu tests,
which do not build without CUDA -- i.e. not on the AMD or Intel hosts where
you would need them. That is a real coverage hole, not an oversight of this
script.
EOM
echo
if [ "$DO_BUILD" = 1 ]; then
    # A configure that FAILED still writes a partial CMakeCache.txt, and every
    # cached value in it is then sticky: CMakeLists guards its autodetects with
    # `if(NOT DEFINED ...)` so a stale entry silently wins on the next run. A
    # bad first configure therefore poisons every retry, including one made
    # after the bug that caused it was fixed. Start clean if the directory was
    # never brought to a working state.
    if [ -d "$BUILD_DIR" ] && [ ! -f "$BUILD_DIR/Makefile" ] \
       && [ ! -f "$BUILD_DIR/build.ninja" ]; then
        echo "  ($BUILD_DIR has no generated build system -- a previous"
        echo "   configure failed and left a stale cache. Removing it.)"
        rm -rf "$BUILD_DIR"
    fi
    # Both flags are passed explicitly rather than left to CMakeLists'
    # autodetect, because -D beats the cache and the autodetect does not.
    #
    # ACPP_TARGETS=generic: the autodetect picks hip:gfx<first agent> on a host
    # with any AMD GPU -- including an integrated one, or a ROCm CPU agent --
    # which AOT-pins these binaries to amdgcn. On a machine being diagnosed for
    # an Intel or NVIDIA GPU that means the parity tests cannot dispatch to the
    # card under investigation at all.
    #
    # XCHPLOT2_BUILD_CUDA: off unless this host actually has nvcc. Otherwise
    # enable_language(CUDA) kills the configure over a toolkit that has nothing
    # to do with the GPU being diagnosed.
    if command -v nvcc >/dev/null 2>&1 || [ -x /opt/cuda/bin/nvcc ] \
       || [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_FLAG="-DXCHPLOT2_BUILD_CUDA=ON"
    else
        CUDA_FLAG="-DXCHPLOT2_BUILD_CUDA=OFF"
    fi
    echo "-- configuring $BUILD_DIR ($CUDA_FLAG) --"
    try cmake -B "$BUILD_DIR" -S "$REPO" -DCMAKE_BUILD_TYPE=Release \
              -DACPP_TARGETS=generic "$CUDA_FLAG" 2>&1 | tail -15
    echo
    echo "-- building (slow; AdaptiveCpp TUs) --"
    try cmake --build "$BUILD_DIR" -j"$(nproc)" --target \
        hellosycl sycl_g_x_parity sycl_bucket_offsets_parity \
        sycl_sort_parity sycl_sort_u32_u64_parity \
        sycl_streaming_partition_parity sycl_t1_parity 2>&1 | tail -25
else
    echo "(--no-build: using whatever is already in $BUILD_DIR)"
fi
echo
run_parity() {
    name="$1"; kind="$2"; what="$3"
    p=""
    for cand in "$BUILD_DIR/tools/parity/$name" "$BUILD_DIR/$name" \
                "$REPO/build/tools/parity/$name"; do
        [ -x "$cand" ] && { p="$cand"; break; }
    done
    if [ -z "$p" ]; then
        printf '  %-8s %-34s NOT BUILT   %s\n' "$kind" "$name" "$what"
        return
    fi
    plog=$(mktemp)
    phang=$(mktemp)
    : > "$phang"
    # stdbuf: these tests print a line per case, and WHERE THEY STOP is the
    # finding. Block-buffered into a file, SIGKILL discards the buffer, so a
    # hang reports nothing at all -- precisely when the output matters most.
    # That cost a round trip: a bundle came back with three timed-out tests and
    # not one case line between them.
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "$p" > "$plog" 2>&1 &
    else
        "$p" > "$plog" 2>&1 &
    fi
    pp=$!
    waited=0
    while kill -0 "$pp" 2>/dev/null && [ "$waited" -lt "$PARITY_LIMIT" ]; do
        sleep 2
        waited=$((waited + 2))
    done
    if kill -0 "$pp" 2>/dev/null; then
        probe_hung_process "$pp" | sed 's/^     /        /' > "$phang"
        kill -9 "$pp" 2>/dev/null
        wait "$pp" 2>/dev/null || true
        rc=124
    else
        wait "$pp"
        rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        printf '  %-8s %-34s PASS        %s\n' "$kind" "$name" "$what"
    else
        printf '  %-8s %-34s *** FAIL ***  %s\n' "$kind" "$name" "$what"
        # WHICH cases failed is the whole diagnostic value. These tests sweep
        # sizes and seeds and print a line each; "it failed" narrows nothing,
        # while "passes at count=16, fails from 16384 up" points straight at
        # the multi-tile path. Discarding this output cost a full round trip.
        printf '        rc=%s\n' "$rc"
        # The head carries the "device: <name>" line these tests print at
        # startup. Without it a failure cannot be attributed to a device --
        # and on a host where GPU 0 is a ROCm CPU agent, "the sort is broken"
        # and "the sort is broken ON THE CARD YOU CARE ABOUT" are different
        # findings. Tail alone cost a round trip.
        echo "        --- first 4 lines (device attribution) ---"
        head -4 "$plog" | sed 's/^/          /'
        echo "        --- failing cases ---"
        grep -iE 'fail|mismatch|abort|error|assert' "$plog" | head -12 | sed 's/^/          /'
        echo "        --- last 15 lines ---"
        tail -15 "$plog" | sed 's/^/          /'
        [ -s "$phang" ] && cat "$phang"
    fi
    rm -f "$plog" "$phang"
}
# The parity tests choose their own device (gpu_selector_v / a bare
# sycl::queue) and nothing tells them which GPU is under investigation. On a
# host where GPU 0 is a ROCm CPU agent, that is emphatically not the same
# device as `--devices 1`, and "the sort is broken" then means something
# entirely different from "the sort is broken on your card". Mask the visible
# backends down to the one that owns the requested device so the tests and the
# plot probe agree on what they are testing. omp stays visible: it is the host
# device, never enumerated as a GPU, and some paths want it present.
dev_backend=""
PARITY_MASK=""
if [ -n "$DEVICES" ]; then
    dev_backend=$(timeout 60 "$BIN" devices 2>/dev/null \
        | sed -nE "s/^[[:space:]]*\[$DEVICES\][[:space:]].*backend=([a-z_]+).*/\1/p" \
        | head -1)
    case "$dev_backend" in
        level_zero) PARITY_MASK="ze;omp"   ;;
        hip)        PARITY_MASK="hip;omp"  ;;
        cuda)       PARITY_MASK="cuda;omp" ;;
    esac
fi
if [ -n "$PARITY_MASK" ]; then
    echo "-- pinning parity to ACPP_VISIBILITY_MASK=$PARITY_MASK"
    echo "   (device $DEVICES reports backend=$dev_backend)"
    export ACPP_VISIBILITY_MASK="$PARITY_MASK"
else
    echo "-- parity NOT pinned: each test picks its own device."
    echo "   Read the 'device:' line in any failure output below before"
    echo "   attributing a failure to the GPU you are investigating."
fi
echo

# Which ISA were these binaries actually compiled for? A parity test AOT-built
# for the wrong vendor either fails to dispatch or silently exercises a
# different GPU than the one under investigation, and either way its verdict
# is worthless. Never let this be an assumption.
echo "-- what the parity binaries were compiled for --"
for d in "$BUILD_DIR" "$REPO/build"; do
    if [ -f "$d/CMakeCache.txt" ]; then
        printf '  %-28s %s\n' "$(basename "$d"):" \
            "$(grep -E '^ACPP_TARGETS(:[A-Z]+)?=' "$d/CMakeCache.txt" 2>/dev/null \
               | head -1 | cut -d= -f2-)"
    fi
done
echo "  (anything other than 'generic' on a multi-vendor host is suspect —"
echo "   an AOT target speaks one ISA and cannot dispatch to the others)"
echo
echo "-- dispatch smoke test --"
hello=""
for cand in "$BUILD_DIR/tools/sanity/hellosycl" "$BUILD_DIR/hellosycl" \
            "$BUILD_DIR/tools/hellosycl"; do
    [ -x "$cand" ] && { hello="$cand"; break; }
done
if [ -n "$hello" ]; then
    try timeout 120 "$hello"
else
    echo "  hellosycl NOT BUILT"
fi
echo
echo "-- results --"
run_parity sycl_g_x_parity                 "[ORACLE]" "AES core vs host-compiled same source"
run_parity sycl_bucket_offsets_parity      "[ORACLE]" "binary search vs std::lower_bound"
run_parity sycl_sort_parity                "[ORACLE]" "radix sort vs std::sort"
run_parity sycl_sort_u32_u64_parity        "[ORACLE]" "u32/u64 sort vs identity-sort + gather"
run_parity sycl_streaming_partition_parity "[ORACLE]" "partition buckets vs CPU multiset"
run_parity sycl_t1_parity                  "[ORACLE]" "Xs + T1 match vs pos2-chip CPU Table1Constructor"
echo
echo "  sycl_g_x_parity is the one to watch: the AES core is inside every"
echo "  kernel in the pipeline, so if it fails, nothing downstream is"
echo "  meaningful. sycl_t1_parity is the only end-to-end phase check with"
echo "  real CPU ground truth on a non-CUDA host."

sec "7. TOOLCHAIN ISOLATION (only runs if small k failed)"
if [ "$SMALL_OK" = 0 ]; then
    echo "(skipped -- small k passed, so the toolchain is producing runnable code)"
else
    T1P=""
    for cand in "$BUILD_DIR/tools/parity/sycl_t1_parity" "$BUILD_DIR/sycl_t1_parity"; do
        [ -x "$cand" ] && { T1P="$cand"; break; }
    done
    if [ -z "$T1P" ]; then
        echo "(sycl_t1_parity not built -- cannot run the matrix)"
    else
        echo "Same binary, same kernels, different JIT/runtime settings. If any"
        echo "row differs from the others, the fault is in the toolchain rather"
        echo "than in our kernel logic."
        echo
        for combo in "baseline:" \
                     "no-JIT-specialization:ACPP_ADAPTIVITY_LEVEL=0" \
                     "JIT-log:ACPP_DEBUG_LEVEL=2"; do
            lbl="${combo%%:*}"; envv="${combo#*:}"
            printf '  %-24s ' "$lbl"
            if [ -n "$envv" ]; then
                if env "$envv" timeout "$PARITY_LIMIT" "$T1P" >/dev/null 2>&1; then echo PASS; else echo "FAIL"; fi
            else
                if timeout "$PARITY_LIMIT" "$T1P" >/dev/null 2>&1; then echo PASS; else echo "FAIL"; fi
            fi
        done
    fi
fi

sec "8. TARGET k=$K (only if small k passed)"
# Drop the parity mask before plotting again: hiding a backend renumbers the
# device list, so --devices would no longer mean what it meant in section 5.
unset ACPP_VISIBILITY_MASK
if [ "$SMALL_OK" = 0 ]; then
    run_probe "8a. k=$K" "$K"
else
    echo "(skipped -- see section 5)"
fi

sec "9. KERNEL LOG AFTER  (GPU page faults, engine resets, OOM kills)"
ka="$(klog)"
if [ -z "$ka" ]; then
    cat <<'EOM'
  UNREADABLE via dmesg and journalctl -k, and passwordless sudo is not
  available. This section carries more diagnostic weight than any other:
  a GPU page fault, an engine reset and an OOM kill appear ONLY here, and
  they are what distinguish "the kernel computed the wrong answer" from
  "the kernel touched memory it does not own". Please run and include:

      sudo dmesg -T | tail -150
EOM
else
    echo "-- lines matching fault/reset/hang/oom --"
    printf '%s\n' "$ka" \
        | grep -iE 'fault|reset|hang|oom|killed process|CAT error|GPU|xe |i915|amdgpu|nvrm' \
        | tail -60 || echo "  (none)"
    echo
    echo "-- last 40 lines --"
    printf '%s\n' "$ka" | tail -40
fi

sec "DONE"
echo "Bundle written to: $OUT"
echo "Home paths were redacted; skim it before sharing anyway."
