// CudaHalfShim.hpp — conditionally pulls in the CUDA Toolkit headers
// consumed by AdaptiveCpp-compatible SYCL TUs:
//   - cuda_fp16.h       (AdaptiveCpp's libkernel/half_representation.hpp
//                        references __half whenever the CUDA backend is
//                        in scope)
//   - cuda_runtime.h    (our .cuh signatures reference cudaEvent_t /
//                        cudaError_t for signature-only interop)
//
// On NVIDIA builds these headers are on the include path and everything
// "just works". On AMD/ROCm builds they're absent — ROCm's HIP headers
// redefine vector types like uchar1 that CUDA's headers also define, so
// pulling both in blows up with typedef redefinition errors.
//
// Uses __has_include so the CUDA Toolkit is only pulled in when actually
// available. For HIP/Intel backends we provide minimal type stubs — just
// enough for function signatures carrying cudaEvent_t / cudaError_t to
// parse. Those parameters are always nullptr / ignored on non-CUDA paths,
// so the stubs are purely compile-time bookkeeping.
//
// Define XCHPLOT2_SKIP_CUDA_FP16 or XCHPLOT2_SKIP_CUDA_RUNTIME to opt out
// of either include unconditionally (useful when CUDA headers are present
// for an unrelated reason but you want to test the stub path).

#pragma once

#if !defined(XCHPLOT2_SKIP_CUDA_RUNTIME) && __has_include(<cuda_runtime.h>)
  #include <cuda_runtime.h>
#else
  // Opaque stubs for signature-only CUDA types. These only appear in
  // launch_*_profiled parameter lists where non-CUDA callers pass nullptr.
  using cudaEvent_t = void*;
  using cudaError_t = int;
  #ifndef cudaSuccess
    #define cudaSuccess 0
  #endif
  #ifndef cudaErrorInvalidValue
    #define cudaErrorInvalidValue 1
  #endif
#endif

#if !defined(XCHPLOT2_SKIP_CUDA_FP16) && __has_include(<cuda_fp16.h>)
  #include <cuda_fp16.h>
#endif
