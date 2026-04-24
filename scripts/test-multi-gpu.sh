#!/usr/bin/env bash
#
# test-multi-gpu.sh — smoke test for the --devices flag.
#
# Two passes:
#
#   1. Argument-parsing checks. Runs xchplot2 against an empty manifest
#      (run_batch returns before touching the GPU, so these work on any
#      host including CI with no GPU visible).
#
#   2. Live multi-device plot, runtime-gated. Skipped automatically when
#      < 2 GPUs are enumerable — so single-GPU dev boxes just see the
#      parse checks run green, and a 2+ GPU rig exercises the fan-out.
#
# Usage:
#   scripts/test-multi-gpu.sh [path/to/xchplot2]
#
# If the path is omitted, falls back to `xchplot2` on PATH (so
# `cargo install --path .` followed by this script works out of the
# box).

set -u
XCHPLOT2="${1:-$(command -v xchplot2 || true)}"
if [[ -z "$XCHPLOT2" || ! -x "$XCHPLOT2" ]]; then
    echo "ERROR: xchplot2 not found. Pass path as \$1 or put it on \$PATH." >&2
    exit 1
fi

PASS=0; FAIL=0; SKIP=0
pass() { printf '  \e[32mPASS\e[0m: %s\n' "$1"; PASS=$((PASS+1)); }
fail() { printf '  \e[31mFAIL\e[0m: %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '  \e[33mSKIP\e[0m: %s\n' "$1"; SKIP=$((SKIP+1)); }

EMPTY_TSV=$(mktemp -t xchplot2-empty-XXXXXX.tsv)
TMP_OUT=$(mktemp -d -t xchplot2-multigpu-out-XXXXXX)
trap 'rm -rf "$EMPTY_TSV" "$TMP_OUT"' EXIT

check_accept() {
    local desc="$1"; shift
    if "$XCHPLOT2" batch "$EMPTY_TSV" "$@" >/dev/null 2>&1; then
        pass "accepts $desc"
    else
        fail "accepts $desc (exit $?)"
    fi
}
check_reject() {
    local desc="$1"; shift
    if ! "$XCHPLOT2" batch "$EMPTY_TSV" "$@" >/dev/null 2>&1; then
        pass "rejects $desc"
    else
        fail "rejects $desc (should have exited nonzero)"
    fi
}

echo "==> --devices argument parsing ($XCHPLOT2)"
check_accept "'all'"              --devices all
check_accept "single id '0'"      --devices 0
check_accept "explicit list"      --devices 0,1,2
check_reject "garbage spec"       --devices badspec
check_reject "negative id"        --devices -1
check_reject "empty value"        --devices ""

# --- Live multi-GPU plot (runtime-gated) ---
echo "==> multi-device plot"

# GPU_COUNT source of truth:
#   - Explicit override lets a CI / test runner force-skip or force-run.
#   - nvidia-smi works on both the main (SYCL+CUDA) and cuda-only branches
#     whenever the target GPUs are NVIDIA, which covers every multi-GPU
#     rig we realistically expect to hit. AMD-only multi-GPU can use
#     `XCHPLOT2_TEST_GPU_COUNT=N scripts/test-multi-gpu.sh`.
GPU_COUNT="${XCHPLOT2_TEST_GPU_COUNT:-}"
if [[ -z "$GPU_COUNT" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null \
                    | head -1 | tr -d ' ' || echo 0)
    fi
    GPU_COUNT="${GPU_COUNT:-0}"
fi

if [[ "$GPU_COUNT" -lt 2 ]]; then
    skip "need >=2 GPUs (got $GPU_COUNT); set XCHPLOT2_TEST_GPU_COUNT=N to override"
else
    # Smallest deterministic plot config we can exercise end-to-end.
    # k=22 is the smallest the pipeline supports; two plots give each
    # worker one to process under round-robin.
    FARMER_PK='a1'$(printf '%.0sa' {1..94})  # fixed-ish 96-hex test key
    POOL_PH='b2'$(printf '%.0sb' {1..62})    # fixed-ish 64-hex test key
    SEED='cd'$(printf '%.0sc' {1..62})       # reproducible across runs

    if "$XCHPLOT2" plot \
        --k 22 --num 2 \
        --farmer-pk "$FARMER_PK" \
        --pool-ph "$POOL_PH" \
        --seed "$SEED" \
        --out "$TMP_OUT" \
        --devices 0,1 >"$TMP_OUT/log" 2>&1
    then
        # Two output files expected, each starting with the 'pos2' magic.
        local_ok=1
        shopt -s nullglob
        plots=("$TMP_OUT"/*.plot2)
        if [[ "${#plots[@]}" -ne 2 ]]; then
            fail "expected 2 plots, got ${#plots[@]}"
            local_ok=0
        else
            for p in "${plots[@]}"; do
                magic=$(head -c 4 "$p" | tr -d '\0')
                if [[ "$magic" != "pos2" ]]; then
                    fail "bad magic in $(basename "$p"): '$magic'"
                    local_ok=0
                fi
            done
        fi
        if (( local_ok )); then
            pass "wrote 2 k=22 plots across devices 0,1"
        fi
    else
        fail "plot --devices 0,1 failed (see $TMP_OUT/log)"
        cat "$TMP_OUT/log" | sed 's/^/    /'
    fi
fi

echo
printf '==> %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
exit $(( FAIL > 0 ? 1 : 0 ))
