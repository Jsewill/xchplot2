#!/usr/bin/env bash
# GPU regression: exact peak+buffer admission and a below-floor rejection.
# Defaults are k=28 CUDA/CUB floors; pass tier:peak-MiB pairs for other backends.
set -euo pipefail
if (( $# < 2 )); then
    echo "usage: $0 XCHPLOT2 DEVICE [tier:peak-MiB ...]" >&2
    exit 2
fi
binary=$(realpath "$1")
device=$2
shift 2
if (( $# == 0 )); then set -- tiny:1064 minimal:3640 compact:5200 plain:7290; fi
buffer=${POS2GPU_VRAM_MARGIN_MB:-256}
[[ $buffer =~ ^[1-9][0-9]{0,6}$ ]] || { echo 'invalid buffer MiB' >&2; exit 2; }
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT
log_dir=${XCHPLOT2_TEST_LOG_DIR:-$test_dir}
mkdir -p "$log_dir"
for spec in "$@"; do
    tier=${spec%%:*}
    peak=${spec#*:}
    [[ $tier =~ ^(plain|compact|minimal|tiny|pinned)$ && $peak =~ ^[1-9][0-9]{0,6}$ ]] || {
        echo "invalid tier:peak-MiB: $spec" >&2; exit 2;
    }
    cap=$((peak + buffer))
    run_log="$log_dir/$tier-boundary.log"
    reject_log="$log_dir/$tier-rejection.log"
    echo "Checking $tier at ${cap} MiB free (peak $peak + buffer $buffer)"
    if ! POS2GPU_MAX_VRAM_MB=$cap POS2GPU_VRAM_MARGIN_MB=$buffer POS2GPU_ASSERT_VRAM=1 \
        "$binary" bench --devices "$device" --tier "$tier" -k 28 -n 3 --warmup 0 \
        --out "$test_dir" --config /dev/null > "$run_log" 2>&1; then
        cat "$run_log" >&2
        exit 1
    fi
    awk '/streaming tier:|peak device VRAM|PHYSICAL high|vram:/{print}' "$run_log"
    # Require an independent driver measurement, including on native CUDA where
    # the streaming allocator's own assertion does not cover runtime overhead.
    if ! awk -v cap="$cap" '$2 == "vram:" && $3 == "peak" {
        seen++; if ($4 <= 0 || $4 > cap) bad=1
    } END {exit (!seen || bad)}' "$run_log"; then
        cat "$run_log" >&2
        echo "FAIL: missing driver VRAM peak or peak exceeds ${cap} MiB" >&2
        exit 1
    fi
    if POS2GPU_MAX_VRAM_MB=$((cap - 1)) POS2GPU_VRAM_MARGIN_MB=$buffer \
        "$binary" bench --devices "$device" --tier "$tier" -k 28 -n 1 --warmup 0 \
        --out "$test_dir" --config /dev/null > "$reject_log" 2>&1; then
        echo "FAIL: $tier accepted less than peak+buffer" >&2
        exit 1
    fi
    if ! grep -q 'including the VRAM buffer' "$reject_log"; then
        cat "$reject_log" >&2
        echo 'FAIL: rejection was unrelated to the VRAM floor' >&2
        exit 1
    fi
done
echo 'All requested GPU tier boundaries passed.'
