#!/usr/bin/env bash
# Build the shared host-only CMake targets, without GPU toolchains or downloads.
# Usage: scripts/test/host-tests.sh [thread|address]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out_dir="$(mktemp -d)"
trap 'rm -rf "$out_dir"' EXIT
flags='-O2 -g -Wall -Wextra'
case "${1:-}" in
    "") ;;
    thread) flags+=' -fsanitize=thread' ;;
    address) flags+=' -fsanitize=address -fsanitize=undefined' ;;
    *) echo "unknown sanitizer: $1 (want: thread, address)" >&2; exit 2 ;;
esac
cmake -S "$repo_root" -B "$out_dir" -DXCHPLOT2_HOST_TESTS_ONLY=ON \
    -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DCMAKE_CXX_FLAGS="$flags"
cmake --build "$out_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ctest --test-dir "$out_dir" --output-on-failure --no-tests=error
