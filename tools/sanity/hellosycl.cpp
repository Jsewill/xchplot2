// hellosycl.cpp — minimal SYCL kernel-dispatch sanity check.
//
// Allocates 16 uint32_t on device, sentinel-fills via memset, runs a
// trivial parallel_for that writes a known pattern, copies back, prints
// pass/fail per slot. Exit 0 if all slots match expected values, else
// non-zero with a "FAIL" line for each mismatch.
//
// Used to localize "is AdaptiveCpp's HIP / CUDA backend actually
// dispatching kernels on this host?" before climbing the abstraction
// stack to sycl_t1_parity / xchplot2. If hellosycl FAILs, no
// xchplot2-level fix can recover the device — the issue is below our
// level (driver mismatch, missing libcudart / libamdhip64, AdaptiveCpp
// JIT producing no-op stubs, ACPP_TARGETS pointing at an ISA the
// installed AdaptiveCpp can't lower for, …).
//
// Compile via the project CMake build (rpath + includes set up
// automatically):
//
//   cmake --build build --target hellosycl
//   ./build/tools/sanity/hellosycl
//
// Or standalone, mirroring whatever ACPP_TARGETS the production binary
// is using (see the cargo:warning lines from `cargo install`):
//
//   ACPP_TARGETS=hip:gfx1013 /opt/adaptivecpp/bin/acpp -O2 hellosycl.cpp -o hellosycl
//   LD_LIBRARY_PATH=/opt/rocm/lib ./hellosycl

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>

int main()
{
    sycl::queue q;
    std::printf("Device: %s\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());

    constexpr std::size_t   N        = 16;
    constexpr std::uint32_t kPattern = 0x12340000u;

    std::uint32_t* d = sycl::malloc_device<std::uint32_t>(N, q);
    if (!d) {
        std::printf("FAIL: sycl::malloc_device returned null\n");
        return 1;
    }

    // Sentinel-fill (0xABABABAB): a "kernel didn't write" outcome shows
    // 0xAB, distinct from "kernel wrote a wrong value" (shows something
    // else) and from random uninitialised bytes that might happen to
    // match the expected pattern by coincidence.
    q.memset(d, 0xAB, N * sizeof(std::uint32_t)).wait();
    q.parallel_for(sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> it) {
        std::size_t idx = it.get_global_id(0);
        d[idx] = kPattern | static_cast<std::uint32_t>(idx);
    }).wait();

    std::uint32_t h[N];
    q.memcpy(h, d, N * sizeof(std::uint32_t)).wait();
    sycl::free(d, q);

    int fails = 0;
    for (std::size_t i = 0; i < N; ++i) {
        std::uint32_t want = kPattern | static_cast<std::uint32_t>(i);
        std::printf("[%2zu] got=0x%08x want=0x%08x %s\n",
            i, h[i], want, h[i] == want ? "OK" : "FAIL");
        if (h[i] != want) ++fails;
    }

    if (fails == 0) {
        std::printf("\nALL OK — AdaptiveCpp can dispatch trivial kernels on this device.\n");
    } else {
        std::printf("\nFAIL — %d/%zu slot(s) wrong. Common causes:\n"
                    "  - libcudart / libamdhip64 not in rpath (check ldd of this binary)\n"
                    "  - AdaptiveCpp JIT producing no-op stubs (ACPP_DEBUG_LEVEL=2 to see)\n"
                    "  - ACPP_TARGETS picks an ISA the installed AdaptiveCpp can't lower\n",
                    fails, N);
    }
    return fails == 0 ? 0 : 1;
}
