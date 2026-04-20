// PortableAttrs.hpp — backend-portable function attribute macros so the
// AES helpers in AesGpu.cuh / AesHashGpu.cuh compile under both nvcc
// (CUDA TU) and acpp/clang (SYCL TU).
//
// Under CUDA the macros expand to the usual __device__ / __host__ / etc.
// markup. Under non-CUDA the markup is dropped and we fall back to plain
// inline (with a force-inline hint where appropriate). The functions
// then compile as ordinary C++ that can be called from a SYCL kernel
// lambda by ADL with no special decoration.

#pragma once

#if defined(__CUDACC__)
  #define POS2_DEVICE_INLINE      __device__ __forceinline__
  #define POS2_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #define POS2_HOST_DEVICE        __host__ __device__
#else
  #define POS2_DEVICE_INLINE      inline __attribute__((always_inline))
  #define POS2_HOST_DEVICE_INLINE inline __attribute__((always_inline))
  #define POS2_HOST_DEVICE
#endif
