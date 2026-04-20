// AesGpu.cu — T-table initialisation. Tables are computed at compile
// time in AesTables.inl (shared with the SYCL backend) and copied here
// into __constant__ memory for the CUDA path.

#include "gpu/AesGpu.cuh"
#include "gpu/AesTables.inl"

namespace pos2gpu {

__device__ __constant__ uint32_t kAesT0[256];
__device__ __constant__ uint32_t kAesT1[256];
__device__ __constant__ uint32_t kAesT2[256];
__device__ __constant__ uint32_t kAesT3[256];

void initialize_aes_tables()
{
    cudaMemcpyToSymbol(kAesT0, aes_tables::T0.data(), sizeof(uint32_t) * 256);
    cudaMemcpyToSymbol(kAesT1, aes_tables::T1.data(), sizeof(uint32_t) * 256);
    cudaMemcpyToSymbol(kAesT2, aes_tables::T2.data(), sizeof(uint32_t) * 256);
    cudaMemcpyToSymbol(kAesT3, aes_tables::T3.data(), sizeof(uint32_t) * 256);
}

} // namespace pos2gpu
