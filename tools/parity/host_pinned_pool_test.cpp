// host_pinned_pool_test — unit tests for HostPinnedPool.
//
// Validates: acquire returns a valid pinned pointer, repeat-acquire
// at the same size returns the same pointer (no realloc), grow
// reallocates, slot_count + live_bytes track correctly, destructor
// frees everything (no leaks under sanitizer or valgrind).

#include "host/HostPinnedPool.hpp"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {

bool check(bool cond, char const* what)
{
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    return cond;
}

} // namespace

int main()
{
    auto& q = pos2gpu::sycl_backend::queue();

    bool all_ok = true;

    // Test 1: basic acquire returns non-null pinned pointer.
    {
        pos2gpu::HostPinnedPool pool;
        void* p = pool.acquire("test1", 1024, q);
        all_ok = check(p != nullptr, "acquire returns non-null") && all_ok;
        all_ok = check(pool.slot_count() == 1, "slot_count == 1 after first acquire") && all_ok;
        all_ok = check(pool.live_bytes() == 1024, "live_bytes == requested size") && all_ok;
        // The pointer is host-pinned via SYCL — we can write to it.
        std::memset(p, 0xAA, 1024);
    }

    // Test 2: same name + same size returns same pointer (no realloc).
    {
        pos2gpu::HostPinnedPool pool;
        void* p1 = pool.acquire("buf", 4096, q);
        void* p2 = pool.acquire("buf", 4096, q);
        void* p3 = pool.acquire("buf", 1024, q);  // smaller than cached
        all_ok = check(p1 == p2, "same-size acquire returns identical pointer") && all_ok;
        all_ok = check(p1 == p3, "smaller-than-cached acquire returns identical pointer") && all_ok;
        all_ok = check(pool.slot_count() == 1, "slot_count stays 1 across repeat acquires") && all_ok;
    }

    // Test 3: larger acquire grows the slot (returns possibly different pointer).
    {
        pos2gpu::HostPinnedPool pool;
        void* p1 = pool.acquire("growme", 1024, q);
        all_ok = check(pool.live_bytes() == 1024, "live_bytes == 1024 before grow") && all_ok;
        void* p2 = pool.acquire("growme", 8192, q);
        all_ok = check(p2 != nullptr, "grown acquire returns non-null") && all_ok;
        all_ok = check(pool.live_bytes() == 8192, "live_bytes == 8192 after grow") && all_ok;
        all_ok = check(pool.slot_count() == 1, "slot_count stays 1 after grow") && all_ok;
        // p1 has been freed by the grow; we don't read from it.
        (void)p1;
    }

    // Test 4: distinct names produce distinct slots.
    {
        pos2gpu::HostPinnedPool pool;
        void* a = pool.acquire("a", 512, q);
        void* b = pool.acquire("b", 512, q);
        void* c = pool.acquire("c", 1024, q);
        all_ok = check(a != b && b != c && a != c, "distinct names give distinct pointers") && all_ok;
        all_ok = check(pool.slot_count() == 3, "slot_count tracks distinct names") && all_ok;
        all_ok = check(pool.live_bytes() == 512 + 512 + 1024, "live_bytes sums all slots") && all_ok;
    }

    // Test 5: typed acquire_as wrapper.
    {
        pos2gpu::HostPinnedPool pool;
        auto* p = pool.acquire_as<std::uint64_t>("typed", 256, q);
        all_ok = check(p != nullptr, "acquire_as returns non-null") && all_ok;
        all_ok = check(pool.live_bytes() == 256 * sizeof(std::uint64_t),
                       "acquire_as multiplies count * sizeof(T)") && all_ok;
        // Write/read sanity.
        for (int i = 0; i < 256; ++i) p[i] = static_cast<std::uint64_t>(i);
        bool readback_ok = true;
        for (int i = 0; i < 256; ++i) if (p[i] != static_cast<std::uint64_t>(i)) readback_ok = false;
        all_ok = check(readback_ok, "acquire_as memory readback ok") && all_ok;
    }

    // Test 6: acquire(0) throws.
    {
        pos2gpu::HostPinnedPool pool;
        bool threw = false;
        try { (void)pool.acquire("zero", 0, q); }
        catch (std::runtime_error const&) { threw = true; }
        all_ok = check(threw, "acquire(0) throws") && all_ok;
    }

    return all_ok ? 0 : 1;
}
