#!/usr/bin/env bash
# host-tests.sh — build and run every CUDA-free test in the tree.
#
# WHY THIS IS NOT `cmake --build`
# -------------------------------
# Configuring the real CMakeLists needs the CUDA Toolkit and a working nvcc.
# The cuda-build job already pays that cost in a CUDA container, but it only
# COMPILES — nothing runs, because the runner has no GPU. So the host TUs that
# need neither (GpuPipeline's spill plumbing lives in .cu, but TempFile.cpp,
# SpillEngine.hpp, SpillCoverage and HostRamPolicy.cpp do not) had their tests
# built only when someone typed their names. The spill path shipped on this
# branch with no automated check at all.
#
# These translation units need a C++20 compiler and nothing else, so this
# compiles them directly on a bare runner and skips the toolchain problem.
#
# The target list is DERIVED from CMakeLists.txt rather than duplicated here.
# A hardcoded list is the same trap one level down — someone adds a test, CMake
# knows about it, this script does not, and it silently never runs. Anything
# declared with add_executable() that pulls in no .cu and no pos2_gpu_* library
# is picked up automatically. cuda_spill_ops_test is a .cu and drops out here
# by design: it drives the engine over real device memory, so it needs the GPU
# job, not this one.
#
# Usage:
#   scripts/test/host-tests.sh              # build + run all of them
#   scripts/test/host-tests.sh thread       # ... under ThreadSanitizer
#   scripts/test/host-tests.sh address      # ... under AddressSanitizer
#
# A sanitizer argument matters most for spill_engine_test: the spill I/O engine
# is a worker pool over a mutex/condvar ticket protocol, and TSan is the only
# thing here that can see a missing lock on a path the test happened not to
# interleave.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

sanitizer="${1:-}"
out_dir="$(mktemp -d)"
trap 'rm -rf "$out_dir"' EXIT

CXX="${CXX:-g++}"
cxxflags=(-std=c++20 -O2 -g -Wall -Wextra -Isrc -pthread)
ldflags=(-pthread)

case "$sanitizer" in
    "")        label="plain" ;;
    thread)    label="tsan";  cxxflags+=(-fsanitize=thread)  ; ldflags+=(-fsanitize=thread)  ;;
    address)   label="asan";  cxxflags+=(-fsanitize=address -fsanitize=undefined)
               ldflags+=(-fsanitize=address -fsanitize=undefined) ;;
    *)         echo "unknown sanitizer: $sanitizer (want: thread, address)" >&2; exit 2 ;;
esac

# name<TAB>space-separated sources, for every CUDA-free add_executable target.
targets="$(python3 - <<'PY'
import re

src = open('CMakeLists.txt').read()

targets = {}
for m in re.finditer(r'add_executable\(\s*(\w+)([^)]*)\)', src):
    files = [x for x in m.group(2).split() if x.endswith(('.cpp', '.cu'))]
    targets.setdefault(m.group(1), []).extend(files)

linked = {}
for m in re.finditer(r'target_link_libraries\(\s*(\w+)\s+PRIVATE([^)]*)\)', src):
    linked[m.group(1)] = m.group(2).split()

for name, files in sorted(targets.items()):
    if any(f.endswith('.cu') for f in files):               continue
    if any('pos2_gpu' in l for l in linked.get(name, [])):  continue
    if not files:                                           continue
    print(name + '\t' + ' '.join(files))
PY
)"

if [ -z "$targets" ]; then
    echo "host-tests: derived an EMPTY target list from CMakeLists.txt." >&2
    echo "That means the parse broke, not that there is nothing to run." >&2
    exit 1
fi

echo "== host tests ($label, $CXX) =="
failed=()
count=0

while IFS=$'\t' read -r name sources; do
    [ -n "$name" ] || continue
    count=$((count + 1))
    printf '\n--- %s ---\n' "$name"
    # shellcheck disable=SC2086
    if ! "$CXX" "${cxxflags[@]}" -o "$out_dir/$name" $sources "${ldflags[@]}"; then
        echo "BUILD FAILED: $name" >&2
        failed+=("$name (build)")
        continue
    fi
    if ! "$out_dir/$name"; then
        echo "TEST FAILED: $name" >&2
        failed+=("$name (run)")
    fi
done <<< "$targets"

printf '\n=====================================\n'
if [ ${#failed[@]} -ne 0 ]; then
    printf '%d of %d host tests FAILED (%s):\n' "${#failed[@]}" "$count" "$label"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
printf 'all %d host tests passed (%s)\n' "$count" "$label"
