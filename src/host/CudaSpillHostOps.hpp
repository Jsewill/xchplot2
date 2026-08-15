// CudaSpillHostOps.hpp — the CUDA backing for SpillEngine's SpillHostOps.
//
// SpillEngine.hpp is deliberately free of any GPU backend: everything it needs
// from one is three calls — two staging allocations and a blocking
// device<->host copy. The SYCL branch supplies those over USM
// (SyclSpillHostOps in its GpuPipeline.cpp); spill_engine_test supplies them
// over plain std::memcpy so the ticket protocol can be exercised with no
// device at all. This is the CUDA one.
//
// It is a header rather than a lump inside GpuPipeline.cu for the same reason
// the engine is: a backend that only exists inside a 4500-line translation
// unit cannot be tested on its own, and this is the layer where a port most
// plausibly goes wrong — wrong direction, wrong pinning, or a copy that has
// not landed when it returns.
//
// WHY copy_blocking MUST BE FULLY COMPLETE ON RETURN
// -------------------------------------------------
// The engine hands the window straight to an I/O worker thread (write path) or
// to the consumer (read path) the instant this returns. An outstanding async
// copy would be a data race against both, and the symptom would not be a
// crash: it would be a spill file holding a chunk that was only partly written,
// over a byte range SpillCoverage considers covered. That is a silently wrong
// plot, which for a plotter is the worst failure mode there is.
//
// So this does NOT rely on the copy landing "because it is the null stream".
// It takes the stream it must order against and synchronises on it explicitly.
// cuda-only's streaming pipeline currently runs on the null stream, which would
// make a bare cudaMemcpy correct today — but that is a property of the caller,
// not of this class, and the tiny path already creates cudaStreamNonBlocking
// streams that the null stream does NOT synchronise with.
//
// Direction is resolved with cudaMemcpyDefault rather than an explicit
// H2D/D2H: the engine's contract is "either direction, both pointers valid",
// and under unified virtual addressing the driver infers it from the pointers.
// That removes a whole class of port bug where a direction argument is right
// on the write path and wrong on the read path.

#pragma once

#include "host/SpillEngine.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pos2gpu {

struct CudaSpillHostOps final : SpillHostOps {
    // Stream this backend orders its copies against. Null means the default
    // stream, which is what the streaming pipeline uses today.
    cudaStream_t stream = nullptr;

    // Bytes currently handed out as staging, for the caller's host-RAM
    // accounting. The engine allocates kNumWindows * kStageBytes (64 MiB) and
    // holds it for the life of the pipeline invocation, no matter how many
    // tables spill — so this should read 64 MiB, and a number that grows with
    // the table count means a window leak.
    std::size_t staging_live = 0;

    explicit CudaSpillHostOps(cudaStream_t s = nullptr) : stream(s) {}

    void* alloc_staging(std::size_t bytes, char const* what) override
    {
        void* p = nullptr;
        // Pinned, not pageable: these windows are the source/destination of
        // every device copy the spill makes, and a pageable staging buffer
        // would force the driver to bounce through its own pinned buffer on
        // each one.
        cudaError_t const err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess || !p) {
            throw std::runtime_error(
                std::string("cudaHostAlloc(") + (what ? what : "spill staging") +
                ", " + std::to_string(bytes) + " B) failed: " +
                cudaGetErrorString(err));
        }
        staging_live += bytes;
        return p;
    }

    void free_staging(void* p) override
    {
        if (p) cudaFreeHost(p);
    }

    void copy_blocking(void* dst, void const* src, std::size_t bytes) override
    {
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("spill copy_blocking: cudaMemcpyAsync(") +
                std::to_string(bytes) + " B) failed: " + cudaGetErrorString(err));
        }
        // Load-bearing. See the header comment: the caller hands this window
        // to another thread the moment we return.
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("spill copy_blocking: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
        }
    }
};

}  // namespace pos2gpu
