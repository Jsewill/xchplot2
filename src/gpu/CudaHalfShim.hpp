// CudaHalfShim.hpp — conditionally pulls in cuda_fp16.h.
//
// AdaptiveCpp's libkernel/detail/half_representation.hpp references
// __half (and friends) from CUDA's cuda_fp16.h whenever the CUDA backend
// path is in scope. So every header that transitively includes
// sycl/sycl.hpp on the CUDA build needs cuda_fp16.h to be visible *first*.
//
// On AMD/ROCm builds the CUDA Toolkit isn't installed and AdaptiveCpp's
// HIP backend doesn't reference __half. Worse, ROCm's HIP headers
// redefine vector types like uchar1 / char1 that CUDA's headers also
// define, so accidentally including both blows up with typedef
// redefinition errors.
//
// Use __has_include so cuda_fp16.h is included only when the CUDA
// Toolkit headers are actually on the search path. Define
// XCHPLOT2_SKIP_CUDA_FP16 to opt out unconditionally (useful when CUDA
// headers are present for an unrelated reason, e.g. a side-by-side
// build, but you want to test the no-CUDA-headers code path).

#pragma once

#if !defined(XCHPLOT2_SKIP_CUDA_FP16) && __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#endif
