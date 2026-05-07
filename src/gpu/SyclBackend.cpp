// SyclBackend.cpp — out-of-line implementations for SyclBackend.hpp.
//
// validate_kernel_dispatch lives here (not in the header) because it
// launches a SYCL kernel via parallel_for, and AdaptiveCpp's SSCP IR
// pass runs per-TU. When the function was inline in the header, every
// SYCL TU that included SyclBackend.hpp produced its own HCF object
// containing the same selftest kernel — at runtime, kernel_launcher
// dispatch saw conflicting HCF entries (different object IDs, identical
// kernel name) and fell through with "No kernel launcher is present for
// requested backend" on kernel_launcher.hpp:119. Confining the body to
// a single TU collapses the registration to one HCF entry.

#include "gpu/SyclBackend.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace pos2gpu::sycl_backend {

void validate_kernel_dispatch(sycl::queue& q)
{
    if (char const* v = std::getenv("POS2GPU_SKIP_SELFTEST"); v && v[0] == '1') {
        return;
    }

    constexpr std::size_t   N        = 16;
    constexpr std::uint32_t kPattern = 0xDEADBEEFu;

    std::uint32_t* d = sycl::malloc_device<std::uint32_t>(N, q);
    if (!d) {
        throw std::runtime_error(
            "[selftest] sycl::malloc_device(16 * u32) returned null. "
            "The SYCL runtime can't allocate even tiny device buffers — "
            "device discovery probably failed (check rocminfo / nvidia-smi, "
            "ACPP_VISIBILITY_MASK).");
    }

    // Sentinel-fill: a "no kernel writes landed" outcome shows the
    // sentinel, not random uninitialised bytes that might happen to
    // match the expected pattern by coincidence.
    q.memset(d, 0xCD, N * sizeof(std::uint32_t)).wait();
    q.parallel_for<selftest_dispatch_kernel>(
        sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> it) {
            std::size_t idx = it.get_global_id(0);
            d[idx] = kPattern + static_cast<std::uint32_t>(idx);
        }).wait();

    std::uint32_t host[N] = {};
    q.memcpy(host, d, N * sizeof(std::uint32_t)).wait();
    sycl::free(d, q);

    int fails = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (host[i] != kPattern + static_cast<std::uint32_t>(i)) ++fails;
    }
    if (fails == 0) return;

    char head[64];
    std::snprintf(head, sizeof(head), "0x%08x (expected 0x%08x)",
                  host[0], kPattern);
    std::string msg =
        "[selftest] SYCL kernel writes are not landing on the device. "
        "A trivial parallel_for(16) writing a known pattern produced "
        "host[0]=";
    msg += head;
    msg += ".\n  ";
    if (host[0] == 0xCDCDCDCDu) {
        msg += "The pre-launch sentinel (0xCDCDCDCD) is intact, so the "
               "kernel completed without writing anything. ";
    } else {
        msg += "The sentinel was overwritten but with a wrong value — "
               "the kernel is dispatching but its output is corrupted. ";
    }
    msg += "Most likely AdaptiveCpp's HIP / CUDA backend on this host is "
           "producing a no-op or miscompiled kernel stub at JIT/AOT time. "
           "Diagnose with:\n"
           "  - ACPP_DEBUG_LEVEL=2 ./xchplot2 ...   (shows the JIT log)\n"
           "  - rocminfo / nvidia-smi              (confirm the actual ISA "
           "matches the AOT target — see cargo:warning lines from your "
           "last `cargo install`)\n"
           "  - try ACPP_TARGETS=generic           (forces SSCP JIT instead "
           "of an AOT spoof)\n"
           "Bypass the self-test with POS2GPU_SKIP_SELFTEST=1 if you've "
           "already validated this device this session.";
    throw std::runtime_error(msg);
}

} // namespace pos2gpu::sycl_backend
