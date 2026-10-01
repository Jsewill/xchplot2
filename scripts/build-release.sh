#!/usr/bin/env bash
# Build a SYCL archive inside ci/release/Containerfile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
llvm=${XCHPLOT2_RELEASE_LLVM:-20}
arch=$(uname -m)
libdir=/usr/lib/"$arch"-linux-gnu
cxx=/usr/bin/g++
components=core,cuda,hip
if [[ $arch == aarch64 ]]; then
    cxx=/usr/bin/g++-14
    components+=,ocl
fi
build_dir=${1:-build/release-linux}
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
runtime="$build_dir/runtime"
licenses="$build_dir/licenses"
rm -rf "$runtime" "$licenses"
mkdir -p "$runtime" "$licenses"
acpp --acpp-deploy="$components:$runtime"
cp /opt/release-licenses/*.txt "$licenses/"
cp /usr/share/doc/libllvm"$llvm"/copyright "$licenses/llvm.txt"
cp /usr/share/doc/libboost1.*-dev/copyright "$licenses/boost.txt"
cp /usr/share/common-licenses/Apache-2.0 "$licenses/Apache-2.0.txt"
# LLVM's distribution libraries have additional ABI-versioned dependencies.
# Bundle their permissively licensed runtimes so other Linux distributions do
# not need Ubuntu's particular libxml2/ICU versions installed system-wide.
libraries=(libffi.so.8 libedit.so.2 libz.so.1 libzstd.so.1 libtinfo.so.6 libbsd.so.0 liblzma.so.5 libmd.so.0)
packages=(libffi8 libedit2 zlib1g libzstd1 libtinfo6 libbsd0 liblzma5 libmd0)
if [[ $arch == aarch64 ]]; then
    libraries+=(libxml2.so.16 libLLVM.so.21.1 libhiprtc-builtins.so.7)
    packages+=(libxml2-16 libllvm21 libhiprtc-builtins7 ocl-icd-libopencl1 libomp5)
else
    libraries+=(libxml2.so.2 libicuuc.so.74 libicudata.so.74)
    packages+=(libxml2 libicu74 "libomp5-$llvm")
fi
for library in "${libraries[@]}"; do
    cp -L "$libdir/$library" "$runtime/$library"
done
for package in "${packages[@]}"; do
    cp /usr/share/doc/"$package"/copyright "$licenses/$package.txt"
done
printf 'AdaptiveCpp: %s\nLLVM: %s\n' \
    "$(cat /opt/release-licenses/adaptivecpp-revision.txt)" \
    "$(/usr/lib/llvm-"$llvm"/bin/llvm-config --version)" > "$build_dir/runtime-info.txt"
dpkg-query -W -f='${Package}: ${Version}\n' "libllvm$llvm" \
    "${packages[@]}" >> "$build_dir/runtime-info.txt"

cp /usr/share/doc/cuda-cudart-12-9/copyright "$licenses/cuda.txt"
curl --proto '=https' --tlsv1.2 -sSfL --retry 5 \
    https://raw.githubusercontent.com/NVIDIA/cccl/v2.8.2/LICENSE \
    -o "$licenses/cuda-cccl.txt"
if [[ $arch == aarch64 ]]; then
    printf 'CUDA headers: glibc math exception specifications backported\n' >> "$build_dir/runtime-info.txt"
    mkdir -p "$licenses/rocm"
    for package in libamdhip64-7 libhiprtc7 libhsa-runtime64-1 libhsakmt1 libamd-comgr3 rocm-device-libs-21; do
        cp /usr/share/doc/"$package"/copyright "$licenses/rocm/$package.txt"
        dpkg-query -W -f='${Package}: ${Version}\n' "$package" >> "$build_dir/runtime-info.txt"
    done
    printf 'Level Zero loader: %s\n' "$(cat /opt/release-licenses/level-zero-revision.txt)" >> "$build_dir/runtime-info.txt"
else
    for component in hip amd_comgr hsa-runtime64 rocprofiler-register ROCm-Device-Libs rocm-llvm; do
        mkdir -p "$licenses/rocm/$component"
        cp /opt/rocm/share/doc/"$component"/LICENSE* "$licenses/rocm/$component/"
    done
    # ROCm 7 merges the HSA thunk into ROCr; retain its separate notice.
    mkdir -p "$licenses/rocm/hsakmt"
    curl --proto '=https' --tlsv1.2 -sSfL --retry 5 \
        https://raw.githubusercontent.com/ROCm/ROCR-Runtime/rocm-7.1.1/libhsakmt/LICENSE.md \
        -o "$licenses/rocm/hsakmt/LICENSE.md"
    printf 'ROCm: %s\n' "$(cat /opt/rocm/.info/version)" >> "$build_dir/runtime-info.txt"
    cp /usr/share/doc/libze1/copyright "$licenses/level-zero.txt"
    dpkg-query -W -f='Level Zero loader: ${Version}\n' libze1 >> "$build_dir/runtime-info.txt"
fi
# AdaptiveCpp 25.10 has no Level Zero deployment component.
cp /opt/adaptivecpp/lib/hipSYCL/librt-backend-ze.so "$runtime/hipSYCL/"
cp /opt/adaptivecpp/lib/hipSYCL/llvm-to-backend/libllvm-to-spirv.so "$runtime/hipSYCL/llvm-to-backend/"
cp /opt/adaptivecpp/lib/hipSYCL/bitcode/libkernel-sscp-spirv-full.bc "$runtime/hipSYCL/bitcode/"
mkdir -p "$runtime/hipSYCL/ext/llvm-spirv/bin"
cp /opt/adaptivecpp/lib/hipSYCL/ext/llvm-spirv/bin/llvm-spirv "$runtime/hipSYCL/ext/llvm-spirv/bin/"
cp -P "$libdir"/libze_loader.so* "$runtime/"
printf 'LLVM-SPIRV: %s\n' "$(cat /opt/release-licenses/llvm-spirv-revision.txt)" >> "$build_dir/runtime-info.txt"
for backend in cuda hip ze; do
    test -f "$runtime/hipSYCL/librt-backend-$backend.so"
done
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
    --manifest-path keygen-rs/Cargo.toml --target "$arch-unknown-linux-gnu" \
    --output-file "$licenses/rust.txt" ci/release/licenses.hbs
cmake -S . -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DACPP_TARGETS=generic -DXCHPLOT2_BUILD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES='50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120' \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
    -DXCHPLOT2_PACKAGE=ON -DXCHPLOT2_PACKAGE_GPU=all \
    -DXCHPLOT2_LICENSE_DIR="$licenses" -DXCHPLOT2_RUNTIME_DIR="$runtime"
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ACPP_VISIBILITY_MASK=omp ctest --test-dir "$build_dir" --output-on-failure --no-tests=error \
    -R '^(bench_stats_test|numa_topology_test|temp_file_test|spill_engine_test|spill_coverage_test|host_guard_test|host_spill_policy_test|vram_budget_test|cli_host_test|pipeline_control_test|plot_file_parity|sycl_twophase_budget_test|solver_filter_parity)$'
cpack --config "$build_dir/CPackConfig.cmake" -B "$build_dir/dist"
