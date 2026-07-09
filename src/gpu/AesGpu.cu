// AesGpu.cu — T-table initialisation. Tables are computed at compile
// time in AesTables.inl (shared with the SYCL backend) and copied here
// into __constant__ memory for the CUDA path.

#include "gpu/AesGpu.cuh"
#include "gpu/AesTables.inl"

#include <stdexcept>
#include <string>

namespace pos2gpu {

__device__ __constant__ uint32_t kAesT0[256];
__device__ __constant__ uint32_t kAesT1[256];
__device__ __constant__ uint32_t kAesT2[256];
__device__ __constant__ uint32_t kAesT3[256];

namespace {

// A failed upload (no context yet, or a prior sticky error) would leave
// the constant tables all-zero and every AES hash silently wrong —
// garbage plots with no diagnostic. Fail loudly instead.
void copy_to_symbol_or_throw(void const* symbol, void const* src,
                             size_t bytes, char const* name)
{
    cudaError_t const err = cudaMemcpyToSymbol(symbol, src, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("initialize_aes_tables: cudaMemcpyToSymbol(") +
            name + ") failed: " + cudaGetErrorString(err));
    }
}

} // namespace

void initialize_aes_tables()
{
    copy_to_symbol_or_throw(kAesT0, aes_tables::T0.data(), sizeof(uint32_t) * 256, "kAesT0");
    copy_to_symbol_or_throw(kAesT1, aes_tables::T1.data(), sizeof(uint32_t) * 256, "kAesT1");
    copy_to_symbol_or_throw(kAesT2, aes_tables::T2.data(), sizeof(uint32_t) * 256, "kAesT2");
    copy_to_symbol_or_throw(kAesT3, aes_tables::T3.data(), sizeof(uint32_t) * 256, "kAesT3");
}

} // namespace pos2gpu
