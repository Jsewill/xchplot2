#!/usr/bin/env bash
# The same clean-system install check runs in Linux containers and WSL2.
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu=${1:?usage: native-install.sh nvidia|amd|intel}
shift
case "$gpu" in
    nvidia) backend=cuda; export XCHPLOT2_BUILD_CUDA=ON ;;
    amd)    backend=hip; export XCHPLOT2_BUILD_CUDA=OFF ;;
    intel)  backend=ze; export XCHPLOT2_BUILD_CUDA=OFF ;;
    *) echo "Unknown GPU vendor: $gpu" >&2; exit 1 ;;
esac

if [[ "${1:-}" == --no-acpp ]]; then
    export ACPP_PREFIX="$HOME/.local"
fi
bash scripts/install-deps.sh --gpu "$gpu" "$@"
acpp_prefix=${ACPP_PREFIX:-/opt/adaptivecpp}
export PATH="$HOME/.cargo/bin:$acpp_prefix/bin:$PATH"
# Hosted runners have no GPU to detect. SSCP compilation is portable;
# vendor kernel execution remains covered by the GPU hardware workflow.
export ACPP_TARGETS=generic CUDA_ARCHITECTURES=75
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}

cargo install --path . --locked --root "$PWD/build-native-install/cargo" \
    --target-dir "$PWD/build-native-install/cargo-target"
# cargo install --git can discard its temporary target directory. Runtime
# libraries must remain available after that cleanup.
rm -rf build-native-install/cargo-target
build-native-install/cargo/bin/xchplot2 --help

test -f "$acpp_prefix/lib/hipSYCL/librt-backend-$backend.so"
if [[ "$gpu" == intel ]]; then
    "$acpp_prefix/lib/hipSYCL/ext/llvm-spirv/bin/llvm-spirv" --version
fi

cmake -S . -B build-native-install/cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DXCHPLOT2_BUILD_CUDA="$XCHPLOT2_BUILD_CUDA" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DACPP_TARGETS="$ACPP_TARGETS"
cmake --build build-native-install/cmake --parallel "$CARGO_BUILD_JOBS" \
    --target xchplot2 plot_file_parity sycl_twophase_budget_test solver_filter_parity
build-native-install/cmake/tools/xchplot2/xchplot2 --help
ACPP_VISIBILITY_MASK=omp ctest --test-dir build-native-install/cmake \
    --output-on-failure --no-tests=error \
    -R '^(plot_file_parity|sycl_twophase_budget_test|solver_filter_parity)$'
