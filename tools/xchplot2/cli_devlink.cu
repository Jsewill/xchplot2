// cli_devlink.cu — exists only to make xchplot2_cli a CUDA-language
// target so CMake's CUDA_RESOLVE_DEVICE_SYMBOLS=ON actually triggers
// nvcc --device-link at static-archive creation time.
//
// xchplot2_cli is the static lib that build.rs hands to Rust's
// linker (cargo install). It depends on pos2_gpu (the CUDA library
// with separable compilation) but has no CUDA sources of its own.
// Without this stub, CMake silently treats xchplot2_cli as a pure-
// C++ static lib, skips the device-link step regardless of
// CUDA_RESOLVE_DEVICE_SYMBOLS, and the resulting libxchplot2_cli.a
// has every per-TU `__sti____cudaRegisterAll()` constructor
// referencing an undefined `__cudaRegisterLinkedBinary_*` stub.
// Rust's `cc` host linker has no way to provide those — it doesn't
// know to invoke nvcc — so the final link fails.
//
// Touching this file via add_library(... cli_devlink.cu) flips
// xchplot2_cli to a CUDA-language target, the device-link runs at
// archive creation, the resolution stubs land inside the .a, and
// the host linker finds them with no extra work.
//
// First reported on a Debian/Ubuntu host with a real GTX 1060 +
// `CUDA_ARCHITECTURES=61 cargo install` — the symptom was a cascade
// of "undefined reference to __cudaRegisterLinkedBinary_*" on every
// .cu TU in pos2_gpu.

namespace {

// Anonymous-namespace `__device__` function — nvcc emits it into the
// per-TU device fatbinary, which gives the device-link step at least
// one input from this TU. Never called from anywhere; marked
// __device__ so it's compiled into the device-side fatbinary, not
// the host-side .o.
__device__ int xchplot2_cli_device_link_anchor() noexcept {
    return 0;
}

}  // namespace
