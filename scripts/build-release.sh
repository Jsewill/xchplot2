#!/usr/bin/env bash
# Build a SYCL archive inside ci/release/Containerfile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=${XCHPLOT2_RELEASE_GPU:?Run this script in the release build image}
llvm=${XCHPLOT2_RELEASE_LLVM:-20}
case "$gpu" in
    nvidia) backend=cuda; components=core,cuda; build_cuda=ON ;;
    amd) backend=hip; components=core,hip; build_cuda=OFF ;;
    intel) backend=ze; components=core; build_cuda=OFF ;;
    *) echo "Unknown release GPU: $gpu" >&2; exit 1 ;;
esac
build_dir=${1:-build/release-$gpu}
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
runtime="$build_dir/runtime"
licenses="$build_dir/licenses"
rm -rf "$runtime" "$licenses"
mkdir -p "$runtime" "$licenses"
acpp --acpp-deploy="$components:$runtime"
cp /opt/release-licenses/*.txt "$licenses/"
cp /usr/share/doc/libllvm"$llvm"/copyright "$licenses/llvm.txt"
cp /usr/share/doc/libboost1.83-dev/copyright "$licenses/boost.txt"
cp /usr/share/common-licenses/Apache-2.0 "$licenses/Apache-2.0.txt"
# LLVM's distribution libraries have additional ABI-versioned dependencies.
# Bundle their permissively licensed runtimes so other Linux distributions do
# not need Ubuntu's particular libxml2/ICU versions installed system-wide.
for library in libffi.so.8 libedit.so.2 libz.so.1 libzstd.so.1 libxml2.so.2 \
        libtinfo.so.6 libbsd.so.0 libicuuc.so.74 libicudata.so.74 liblzma.so.5 libmd.so.0; do
    cp -L /usr/lib/x86_64-linux-gnu/"$library" "$runtime/$library"
done
for package in libffi8 libedit2 zlib1g libzstd1 libxml2 libtinfo6 libbsd0 libicu74 liblzma5 libmd0; do
    cp /usr/share/doc/"$package"/copyright "$licenses/$package.txt"
done
printf 'AdaptiveCpp: %s\nLLVM: %s\n' \
    "$(cat /opt/release-licenses/adaptivecpp-revision.txt)" \
    "$(/usr/lib/llvm-"$llvm"/bin/llvm-config --version)" > "$build_dir/runtime-info.txt"
dpkg-query -W -f='${Package}: ${Version}\n' "libllvm$llvm" "libomp5-$llvm" libffi8 libedit2 \
    zlib1g libzstd1 libxml2 libtinfo6 libbsd0 libicu74 liblzma5 libmd0 >> "$build_dir/runtime-info.txt"

case "$gpu" in
    nvidia)
        cp /usr/share/doc/cuda-cudart-12-9/copyright "$licenses/cuda.txt"
        curl --proto '=https' --tlsv1.2 -sSfL --retry 5 \
            https://raw.githubusercontent.com/NVIDIA/cccl/v2.8.2/LICENSE \
            -o "$licenses/cuda-cccl.txt"
        ;;
    amd)
        for component in hip amd_comgr hsa-runtime64 hsakmt rocprofiler-register ROCm-Device-Libs rocm-llvm; do
            mkdir -p "$licenses/rocm/$component"
            cp /opt/rocm/share/doc/"$component"/LICENSE* "$licenses/rocm/$component/"
        done
        printf 'ROCm: %s\n' "$(cat /opt/rocm/.info/version)" >> "$build_dir/runtime-info.txt"
        ;;
    intel)
        # AdaptiveCpp 25.10 has no Level Zero deployment component.
        cp /opt/adaptivecpp/lib/hipSYCL/librt-backend-ze.so "$runtime/hipSYCL/"
        cp /opt/adaptivecpp/lib/hipSYCL/llvm-to-backend/libllvm-to-spirv.so "$runtime/hipSYCL/llvm-to-backend/"
        cp /opt/adaptivecpp/lib/hipSYCL/bitcode/libkernel-sscp-spirv-full.bc "$runtime/hipSYCL/bitcode/"
        mkdir -p "$runtime/hipSYCL/ext/llvm-spirv/bin"
        cp /opt/adaptivecpp/lib/hipSYCL/ext/llvm-spirv/bin/llvm-spirv "$runtime/hipSYCL/ext/llvm-spirv/bin/"
        cp -P /usr/lib/x86_64-linux-gnu/libze_loader.so* "$runtime/"
        cp /usr/share/doc/libze1/copyright "$licenses/level-zero.txt"
        printf 'LLVM-SPIRV: %s\n' "$(cat /opt/release-licenses/llvm-spirv-revision.txt)" >> "$build_dir/runtime-info.txt"
        dpkg-query -W -f='Level Zero loader: ${Version}\n' libze1 >> "$build_dir/runtime-info.txt"
        ;;
esac
test -f "$runtime/hipSYCL/librt-backend-$backend.so"
# Core OS libraries remain system prerequisites, including libnuma.
rm -f "$runtime"/libnuma.so*
python3 - "$runtime" <<'PY'
import os
from pathlib import Path
import subprocess
import sys

runtime = Path(sys.argv[1])
for path in runtime.rglob("*"):
    if path.is_symlink() or not path.is_file():
        continue
    with path.open("rb") as file:
        if file.read(4) != b"\x7fELF":
            continue
    # libcudart only needs system libraries; retain NVIDIA's original binary.
    if path.name.startswith("libcudart.so"):
        continue
    relative = os.path.relpath(runtime, path.parent)
    root = "$ORIGIN/" + relative
    subprocess.run(["patchelf", "--set-rpath",
                    "$ORIGIN:" + root + ":" + root + "/hipSYCL/llvm-to-backend", path], check=True)
PY

rust_docs="$(rustc --print sysroot)/share/doc/rust"
mkdir -p "$licenses/rust-standard-library"
cp "$rust_docs/COPYRIGHT-library.html" "$licenses/rust-standard-library/"
cp -r "$rust_docs/licenses" "$licenses/rust-standard-library/"
cargo about generate --locked --fail \
    --manifest-path keygen-rs/Cargo.toml --target x86_64-unknown-linux-gnu \
    --output-file "$licenses/rust.txt" ci/release/licenses.hbs
cmake -S . -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DACPP_TARGETS=generic -DXCHPLOT2_BUILD_CUDA="$build_cuda" \
    -DCMAKE_CUDA_ARCHITECTURES='50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120' \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
    -DXCHPLOT2_PACKAGE=ON -DXCHPLOT2_PACKAGE_GPU="$gpu" \
    -DXCHPLOT2_LICENSE_DIR="$licenses" -DXCHPLOT2_RUNTIME_DIR="$runtime"
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ACPP_VISIBILITY_MASK=omp ctest --test-dir "$build_dir" --output-on-failure --no-tests=error \
    -R '^(bench_stats_test|numa_topology_test|temp_file_test|spill_engine_test|spill_coverage_test|host_guard_test|host_spill_policy_test|vram_budget_test|cli_host_test|pipeline_control_test|plot_file_parity|sycl_twophase_budget_test|solver_filter_parity)$'
cpack --config "$build_dir/CPackConfig.cmake" -B "$build_dir/dist"
