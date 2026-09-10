#!/usr/bin/env bash
# Build the Linux CUDA archive inside ci/release/Containerfile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
build_dir="${1:-build/release}"
mkdir -p "$build_dir/licenses"
build_dir="$(cd "$build_dir" && pwd)"

cp /usr/share/doc/cuda-cudart-12-9/copyright "$build_dir/licenses/cuda.txt"
curl --proto '=https' --tlsv1.2 -sSfL --retry 5 \
    https://raw.githubusercontent.com/NVIDIA/cccl/v2.8.2/LICENSE \
    -o "$build_dir/licenses/cuda-cccl.txt"
rust_docs="$(rustc --print sysroot)/share/doc/rust"
mkdir -p "$build_dir/licenses/rust-standard-library"
cp "$rust_docs/COPYRIGHT-library.html" "$build_dir/licenses/rust-standard-library/"
cp -r "$rust_docs/licenses" "$build_dir/licenses/rust-standard-library/"
cargo about generate --locked --fail \
    --manifest-path keygen-rs/Cargo.toml --target x86_64-unknown-linux-gnu \
    --output-file "$build_dir/licenses/rust.txt" ci/release/licenses.hbs
cmake -S . -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES='50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120' \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
    -DXCHPLOT2_PACKAGE=ON -DXCHPLOT2_LICENSE_DIR="$build_dir/licenses"
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ctest --test-dir "$build_dir" --output-on-failure --no-tests=error \
    -R '^(bench_stats_test|numa_topology_test|temp_file_test|spill_engine_test|spill_coverage_test|host_guard_test|host_spill_policy_test|vram_budget_test|cli_host_test|plot_file_parity|solver_filter_parity)$'
cpack --config "$build_dir/CPackConfig.cmake" -B "$build_dir/dist"
