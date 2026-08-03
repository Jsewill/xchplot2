// GpuBufferPool.cu — queries per-phase scratch sizes once and allocates
// worst-case-sized persistent buffers. Slice 13 migrated the device and
// pinned-host allocations from the cudaMalloc / cudaMallocHost family to
// sycl::malloc_device / sycl::malloc_host on the shared SYCL queue;
// cudaMemGetInfo is left as-is because it's a context-level query that
// works regardless of which runtime is doing the allocations (SYCL +
// CUDA host code share the same primary CUDA context).

#include "host/GpuBufferPool.hpp"

#ifndef _WIN32
#include <dlfcn.h>   // runtime lookup of hipMemGetInfo on AMD — see device_memory_probe
#else
#include <windows.h>  // GlobalMemoryStatusEx — see host_memory_probe
#endif
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "host/PoolSizing.hpp"

#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>      // probe_target_for cache
#include <mutex>    // ditto — the VRAM watchdog polls it from its own thread
#include <stdexcept>
#include <string>

namespace pos2gpu {

namespace {


// Allocate `bytes` of device memory on `q` and check for null. The cap-and-
// throw helpers in GpuPipeline.cu are streaming-pipeline specific; the pool
// just allocates worst-case sizes once at construction so a one-line wrap
// suffices.
// Format a byte count as "<N> bytes (<N.NN> MB)" for diagnostics. The
// raw byte count surfaces sub-MiB requests that would otherwise round
// to "0 MB"; the MB form keeps human readability for the > 1 MiB case.
inline std::string fmt_alloc_bytes(size_t bytes)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%zu bytes (%.2f MB)",
                  bytes, double(bytes) / (1024.0 * 1024.0));
    return std::string(buf);
}

// AdaptiveCpp's CUDA allocator throws sycl::exception on cudaMalloc
// failure (e.g. "cuda_allocator: cudaMalloc() failed (error code =
// CUDA:2)" for cudaErrorMemoryAllocation). Older / non-CUDA backends
// may instead return nullptr. Cover both paths with one diagnostic
// shape so callers see "sycl::malloc_device(d_pair_a, 4690 MB) failed:
// <underlying>" regardless of which branch fired. This also catches
// the throw synchronously so the async error handler doesn't log the
// same CUDA error a second time after caller cleanup.
inline void* sycl_alloc_device_or_throw(size_t bytes, sycl::queue& q,
                                        char const* what)
{
    void* p = nullptr;
    try {
        p = sycl::malloc_device(bytes, q);
    } catch (sycl::exception const& e) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") failed: " + e.what() +
            ". Likely transient OOM — check `nvidia-smi` for other GPU "
            "consumers, or set POS2GPU_MAX_VRAM_MB lower if VRAM is "
            "shared with display/compositor.");
    }
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") returned null (out of device "
            "memory). Likely transient OOM — check `nvidia-smi` for "
            "other GPU consumers, or set POS2GPU_MAX_VRAM_MB lower if "
            "VRAM is shared with display/compositor.");
    }
    return p;
}

inline void* sycl_alloc_host_or_throw(size_t bytes, sycl::queue& q,
                                      char const* what)
{
    host_pinned_reserve_check(bytes, what);
    void* p = nullptr;
    try {
        p = sycl::malloc_host(bytes, q);
    } catch (sycl::exception const& e) {
        throw std::runtime_error(
            std::string("sycl::malloc_host(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") failed: " + e.what());
    }
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_host(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") returned null (out of host pinned memory)");
    }
    return p;
}

} // namespace

GpuBufferPool::GpuBufferPool(int k_, int strength_, bool testnet_)
    : k(k_), strength(strength_), testnet(testnet_)
{
    sycl::queue& q = sycl_backend::queue();

    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    total_xs = 1ULL << k;
    cap      = max_pairs_per_section(k, num_section_bits) * (1ULL << num_section_bits);

    // d_storage must hold EITHER total_xs XsCandidateGpu (8 B each) OR
    // THREE cap-sized uint32 key/val arrays during sort. Only three, not
    // four: the sort API signature takes a (keys_in, keys_out, vals_in,
    // vals_out) quad, but pool-path callers always pass the SoA match-info
    // stream (d_t1_mi / d_t2_mi, living in d_pair_a) as keys_in, so the
    // keys_in slot inside d_storage was never read. Dropping it saves
    // cap·4 B (~1.09 GiB at k=28) — enough to close the 0.71 GiB pool
    // shortfall on 12 GiB cards.
    storage_bytes = std::max(
        static_cast<size_t>(total_xs) * sizeof(XsCandidateGpu),
        static_cast<size_t>(cap) * 3 * sizeof(uint32_t));

    // d_pair_a holds the *match output* of the current phase: T1 SoA
    // (meta·8 B + mi·4 B = 12 B), T2 SoA (meta·8 B + mi·4 B + xbits·4 B =
    // 16 B), then T3 (T3PairingGpu, 8 B). Worst case is T2 at 16 B/entry.
    // It does NOT alias the Xs construction scratch — that's d_pair_b.
    pair_a_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(T1PairingGpu),
        static_cast<size_t>(cap) * sizeof(T2PairingGpu),
        static_cast<size_t>(cap) * sizeof(T3PairingGpu),
        static_cast<size_t>(cap) * sizeof(uint64_t),
    });

    // d_pair_b holds the *sort output* of the current phase (sorted T1
    // meta, sorted T2 meta+xbits, T3 frags) AND the Xs construction
    // scratch. Sized to the max of those.
    //
    // Split-keys_a optimisation: the pool places the Xs sort's keys_a
    // slot (total_xs·u32 = 1 GiB at k=28) in d_storage's tail — idle
    // during Xs gen+sort, and the final pack phase only writes
    // d_storage[0..total_xs·8), leaving the tail region undisturbed.
    // This drops xs_temp_bytes from ~4.36 GB (4·N·u32 + cub) to
    // ~3.22 GB (3·N·u32 + cub). At k=28 pair_b is then bounded by
    // cap·12 (sorted T2 meta+xbits = 3.27 GB) rather than xs scratch,
    // saving ~1.09 GB off the pool's peak VRAM requirement vs the
    // pre-split layout.
    uint8_t dummy_plot_id[32] = {};
    // Non-null sentinel tells launch_construct_xs to report the
    // split-layout size. The sentinel value is read only in sizing
    // mode (d_temp_storage == nullptr), where only its non-null-ness
    // matters.
    void* const xs_split_sentinel = reinterpret_cast<void*>(uintptr_t{1});
    launch_construct_xs(dummy_plot_id, k, testnet,
                                   nullptr, nullptr, &xs_temp_bytes, q,
                                   xs_split_sentinel);
    pair_b_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(uint64_t),                          // sorted T1 meta
        static_cast<size_t>(cap) * (sizeof(uint64_t) + sizeof(uint32_t)),     // sorted T2 meta+xbits
        static_cast<size_t>(cap) * sizeof(uint64_t),                          // T3 frags out
        xs_temp_bytes,                                                        // Xs aliased scratch (3·N·u32 + cub)
    });

    // Query CUB sort scratch sizes (largest across T1/T2/T3 sorts).
    size_t s_pairs = 0;
    launch_sort_pairs_u32_u32(
        nullptr, s_pairs,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        cap, 0, k, q);
    size_t s_keys = 0;
    launch_sort_keys_u64(
        nullptr, s_keys,
        static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap, 0, 2 * k, q);
    sort_scratch_bytes = std::max(s_pairs, s_keys);

    pinned_bytes = cap * sizeof(uint64_t);

    // Check VRAM before attempting allocation so we can give a useful
    // diagnostic instead of a generic allocation failure. The margin covers
    // GPU driver/context state, sort scratch, AES T-tables, and other small
    // runtime allocations.
    //
    // Uses query_device_memory(), which on the NVIDIA path asks the driver via
    // cudaMemGetInfo. This gate used to approximate free_b == total_b ("SYCL
    // has no portable free-memory query"), which meant it compared the pool's
    // size against the card's *total* VRAM and ignored the CUDA context, the
    // display server, and every other process on the device — so it waved cards
    // into the pool path that could not hold it. It also meant
    // POS2GPU_MAX_VRAM_MB, the knob for rehearsing a smaller card, had no
    // effect here at all.
    {
        size_t const required_device =
            storage_bytes + pair_a_bytes + pair_b_bytes + sort_scratch_bytes + sizeof(uint64_t);
        // Shares vram_safety_margin() with the streaming picker; the two used to
        // disagree (256 here, 128 there) for no reason anyone recorded.
        //
        // The margin is NOT an allowance for the CUDA context: free_bytes comes
        // from cudaMemGetInfo, and query_device_memory() builds the SYCL queue
        // before calling it, so the ~390 MiB context is already deducted. Nor is
        // it an allowance for unmodelled allocations — required_device is the
        // whole footprint, and the driver agrees to within 8 MiB. It is headroom
        // against *other tenants* on the card taking VRAM after this gate has
        // waved us through. See vram_safety_margin() for how sizing it against
        // the wrong quantity got it set to 512 and cost every card 384 MiB.
        size_t const margin = vram_safety_margin();
        DeviceMemInfo const mem = query_device_memory();
        size_t const total_b = mem.total_bytes;
        size_t const free_b  = mem.free_bytes;
        if (free_b < required_device + margin) {
            auto to_gib = [](size_t b) { return b / double(1ULL << 30); };
            InsufficientVramError e(
                "GpuBufferPool: insufficient device VRAM for k=" +
                std::to_string(k) + " strength=" + std::to_string(strength) +
                "; need ~" + std::to_string(to_gib(required_device + margin)).substr(0, 5) +
                " GiB (pool " + std::to_string(to_gib(required_device)).substr(0, 5) +
                " GiB + " + std::to_string(to_gib(margin)).substr(0, 4) +
                " GiB runtime), only " +
                std::to_string(to_gib(free_b)).substr(0, 5) +
                " GiB free of " + std::to_string(to_gib(total_b)).substr(0, 5) +
                " GiB total. Falling back to the streaming pipeline.");
            e.required_bytes = required_device + margin;
            e.free_bytes     = free_b;
            e.total_bytes    = total_b;
            throw e;
        }
    }

    if (getenv("POS2GPU_POOL_DEBUG")) {
        DeviceMemInfo const dbg_mem = query_device_memory();
        std::fprintf(stderr,
            "[pool] k=%d strength=%d cap=%llu total_xs=%llu "
            "total=%.2fGB free=%.2fGB\n",
            k, strength, (unsigned long long)cap, (unsigned long long)total_xs,
            dbg_mem.total_bytes/1e9, dbg_mem.free_bytes/1e9);
        std::fprintf(stderr,
            "[pool] sizes: storage=%.2fGB pair_a=%.2fGB pair_b=%.2fGB "
            "xs_temp(alias→pair_b)=%.2fGB sort_scratch=%.2fGB pinned=%.2fGB\n",
            storage_bytes/1e9, pair_a_bytes/1e9, pair_b_bytes/1e9,
            xs_temp_bytes/1e9, sort_scratch_bytes/1e9, pinned_bytes/1e9);
    }

    // Wrap allocations so a mid-sequence failure (e.g. d_pair_b OOM after
    // d_storage + d_pair_a have already succeeded) frees the pre-allocated
    // buffers instead of leaking ~10 GB of device VRAM and ~7 GB of host
    // pinned memory per failed pool ctor across a batch retry loop.
    auto cleanup_partial = [&]{
        if (d_storage)       { sycl::free(d_storage,      q); d_storage      = nullptr; }
        if (d_pair_a)        { sycl::free(d_pair_a,       q); d_pair_a       = nullptr; }
        if (d_pair_b)        { sycl::free(d_pair_b,       q); d_pair_b       = nullptr; }
        if (d_sort_scratch)  { sycl::free(d_sort_scratch, q); d_sort_scratch = nullptr; }
        if (d_counter)       { sycl::free(d_counter,      q); d_counter      = nullptr; }
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            if (h_pinned_t3[i]) { sycl::free(h_pinned_t3[i], q); h_pinned_t3[i] = nullptr; }
        }
    };
    try {
        d_storage      = sycl_alloc_device_or_throw(storage_bytes,      q, "d_storage");
        // d_pair_a is allocated lazily in ensure_pair_a(), called by
        // run_gpu_pipeline's pool path right after submitting Xs gen
        // — the malloc_device then overlaps with Xs GPU execution.
        // Saves ~400-500 ms on first-plot wall vs eager alloc; batch
        // plots 2+ are unaffected (fast-path pointer lookup).
        d_pair_b       = sycl_alloc_device_or_throw(pair_b_bytes,       q, "d_pair_b");
        d_sort_scratch = sycl_alloc_device_or_throw(sort_scratch_bytes, q, "d_sort_scratch");
        d_counter      = static_cast<uint64_t*>(
            sycl_alloc_device_or_throw(sizeof(uint64_t),                q, "d_counter"));
        // h_pinned_t3[] is allocated lazily in ensure_pinned(); see
        // the header comment for why. Single-plot runs only ever
        // touch slot 0 so the other two 2.2 GB malloc_host calls
        // aren't paid at all.
    } catch (...) {
        cleanup_partial();
        throw;
    }
}

void* GpuBufferPool::ensure_pair_a()
{
    if (d_pair_a) return d_pair_a;
    std::lock_guard<std::mutex> lk(pair_a_mu_);
    if (d_pair_a) return d_pair_a;
    sycl::queue& q = sycl_backend::queue();
    d_pair_a = sycl_alloc_device_or_throw(pair_a_bytes, q, "d_pair_a");
    return d_pair_a;
}

void GpuBufferPool::release_pair_a()
{
    std::lock_guard<std::mutex> lk(pair_a_mu_);
    if (!d_pair_a) return;
    sycl::free(d_pair_a, sycl_backend::queue());
    d_pair_a = nullptr;
}

uint64_t* GpuBufferPool::ensure_pinned(int idx)
{
    if (idx < 0 || idx >= kNumPinnedBuffers) {
        throw std::runtime_error("GpuBufferPool::ensure_pinned: idx out of range");
    }
    // Double-checked locking: fast path skips the mutex once the
    // slot's pointer is visible. Writes inside the mutex are
    // release-ordered w.r.t. the mutex release; the unlocked read
    // on the fast path is an acquire (relaxed access is fine here
    // because x86 and arm64 give us acquire ordering for aligned
    // pointer reads; if this ever needs to be portable to weaker
    // architectures, make h_pinned_t3 std::atomic<uint64_t*>[]).
    if (h_pinned_t3[idx]) return h_pinned_t3[idx];
    std::lock_guard<std::mutex> lk(pinned_mu_[idx]);
    if (h_pinned_t3[idx]) return h_pinned_t3[idx];
    sycl::queue& q = sycl_backend::queue();
    h_pinned_t3[idx] = static_cast<uint64_t*>(
        sycl_alloc_host_or_throw(pinned_bytes, q, "h_pinned_t3"));
    return h_pinned_t3[idx];
}

GpuBufferPool::~GpuBufferPool()
{
    sycl::queue& q = sycl_backend::queue();
    if (d_storage)       sycl::free(d_storage,      q);
    if (d_pair_a)        sycl::free(d_pair_a,       q);
    if (d_pair_b)        sycl::free(d_pair_b,       q);
    if (d_sort_scratch)  sycl::free(d_sort_scratch, q);
    if (d_counter)       sycl::free(d_counter,      q);
    for (int i = 0; i < kNumPinnedBuffers; ++i) {
        if (h_pinned_t3[i]) sycl::free(h_pinned_t3[i], q);
    }
}

#ifdef XCHPLOT2_HAVE_CUB
// Defined in DeviceMemCuda.cu. XCHPLOT2_HAVE_CUB is set exactly when the
// nvcc-compiled TUs (POS2_GPU_CUDA_SRC) are linked, which is the condition we
// need here: "is the CUDA runtime available to call".
bool cuda_query_device_memory(int device_ordinal,
                              std::size_t& free_bytes,
                              std::size_t& total_bytes);
#endif

// AMD free-VRAM probe. Resolved at runtime rather than linked, deliberately:
// asking CMake for HIP would add a build-time dependency and a libamdhip64 link
// to every build, including the NVIDIA ones that will never call this. On a
// ROCm build AdaptiveCpp has already loaded the HIP runtime into the process,
// so the symbol is simply there; on any other build the lookup fails once and
// we fall back exactly as before. Nothing about a non-AMD build changes.
//
// Without this an AMD card gets free_bytes = total_bytes (see below), and the
// tier picker sizes against VRAM the driver, the display server and everything
// else on the card have already taken — an 8 GB card reports 8192 MB free,
// picks plain (needs 7802), actually has ~7400, and OOMs mid-plot. Same bug the
// NVIDIA path had.
#ifndef _WIN32
namespace {

bool hip_query_device_memory(int device_ordinal,
                             size_t& free_bytes,
                             size_t& total_bytes)
{
    using MemGetInfoFn = int (*)(size_t*, size_t*);
    using SetDeviceFn  = int (*)(int);
    using GetDeviceFn  = int (*)(int*);

    static void* const handle = [] () -> void* {
        // Already in-process? A ROCm build has it loaded before we get here.
        if (void* self = dlopen(nullptr, RTLD_LAZY | RTLD_LOCAL)) {
            if (dlsym(self, "hipMemGetInfo")) return self;
            dlclose(self);
        }
        for (char const* soname : {"libamdhip64.so", "libamdhip64.so.6",
                                   "libamdhip64.so.5"}) {
            if (void* h = dlopen(soname, RTLD_LAZY | RTLD_LOCAL)) return h;
        }
        return nullptr;
    }();
    if (!handle) return false;

    static auto const mem_get_info =
        reinterpret_cast<MemGetInfoFn>(dlsym(handle, "hipMemGetInfo"));
    static auto const set_device =
        reinterpret_cast<SetDeviceFn>(dlsym(handle, "hipSetDevice"));
    static auto const get_device =
        reinterpret_cast<GetDeviceFn>(dlsym(handle, "hipGetDevice"));
    if (!mem_get_info) return false;

    int  prev     = 0;
    bool switched = false;
    if (device_ordinal >= 0 && get_device && set_device
        && get_device(&prev) == 0 && prev != device_ordinal) {
        if (set_device(device_ordinal) != 0) return false;
        switched = true;
    }
    size_t f = 0, t = 0;
    int const rc = mem_get_info(&f, &t);       // hipSuccess == 0
    if (switched) set_device(prev);
    if (rc != 0 || t == 0) return false;

    free_bytes  = f;
    total_bytes = t;
    return true;
}

} // namespace
#endif  // !_WIN32

// Host RAM. The CPU "device" IS the host, so its streaming tier has to be
// sized against host memory — asking a GPU runtime how much memory the CPU has
// is a category error.
//
// MemAvailable, not MemFree: it is the kernel's own estimate of what a fresh
// allocation can obtain without swapping, and it counts reclaimable page cache.
// MemFree on a box that has been plotting reads near zero (the cache is holding
// the plots just written), which would pin every CPU worker to the smallest
// tier for no reason. This box: MemFree 13 GiB, MemAvailable 77 GiB.
namespace {

// Testing knob: XCHPLOT2_HOST_FREE_MB makes the host look SMALLER than it is,
// so the host-RAM gate and the disk-offload rescue can be exercised on a box
// that has plenty. It CLAMPS — it can only ever lower the figure, never raise
// it — so no setting of it can talk the plotter past a real shortage and into
// the OOM killer. Because every consumer sees the reduced number, including
// host_pinned_reserve_check() at each allocation, a run under this knob is a
// faithful simulation and not just a way to steer the tier picker: the spill
// really does have to fit under the pretend ceiling or the run fails the same
// way it would on the small host.
void apply_host_free_override(size_t& free_bytes)
{
    static size_t const cap = [] () -> size_t {
        if (char const* v = std::getenv("XCHPLOT2_HOST_FREE_MB"); v && v[0]) {
            size_t const mb = size_t(std::strtoull(v, nullptr, 10));
            if (mb > 0) return mb << 20;
        }
        return 0;   // 0 == no override
    }();
    if (cap && cap < free_bytes) free_bytes = cap;
}

bool host_memory_probe(size_t& free_bytes, size_t& total_bytes)
{
#if defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (!::GlobalMemoryStatusEx(&st)) return false;
    free_bytes  = static_cast<size_t>(st.ullAvailPhys);
    total_bytes = static_cast<size_t>(st.ullTotalPhys);
    apply_host_free_override(free_bytes);
    return true;
#else
    std::FILE* fp = std::fopen("/proc/meminfo", "re");
    if (!fp) return false;
    unsigned long long avail_kb = 0;
    unsigned long long total_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), fp)) {
        unsigned long long v = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1)     avail_kb = v;
        else if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1)    total_kb = v;
    }
    std::fclose(fp);
    if (total_kb == 0) return false;
    // MemAvailable landed in Linux 3.14. Older kernels report only MemFree,
    // which understates badly — fall back to MemTotal and let the pool's own
    // allocation failures be the backstop, rather than silently choosing Tiny.
    free_bytes  = static_cast<size_t>(avail_kb ? avail_kb : total_kb) << 10;
    total_bytes = static_cast<size_t>(total_kb) << 10;
    apply_host_free_override(free_bytes);
    return true;
#endif
}

}  // namespace

// Which vendor runtime, if any, owns the SYCL device behind `device_ordinal`,
// and that device's ordinal WITHIN that runtime.
//
// A mixed-vendor host forces this. Observed on an AMD APU (ROCm iGPU) plus a
// discrete Intel Arc: hipMemGetInfo answers whenever libamdhip64 is loaded, no
// matter which device SYCL is actually running on. With kDefaultGpuId the HIP
// probe never even calls hipSetDevice, so it reported the iGPU's system-RAM
// carve-out — 7.35 GiB — while the kernels ran on the Arc. The tier picker
// sized the Arc against that figure, refused the pool path it had room for,
// and dropped to plain streaming: 15.18 s/plot where the same card does 13.53.
// The VRAM watchdog then polled a device nothing was allocated on and reported
// a peak of 0, so the check that exists to catch exactly this was blind to it.
//
// That failure happened to be SAFE (under-reported memory only costs speed).
// The same bug over-reports whenever the iGPU's share of system RAM exceeds
// the discrete card's VRAM, and then the picker hands out a tier that cannot
// fit and the card OOMs mid-plot.
//
// The ordinal matters as much as the runtime: a SYCL device index has no
// relationship to a HIP or CUDA device index once the box has more than one
// vendor in it. SYCL index 1 here is the Arc, and HIP device 1 does not exist.
namespace {

enum class ProbeRuntime { None, Cuda, Hip };

struct ProbeTarget {
    ProbeRuntime runtime = ProbeRuntime::None;
    int          ordinal = 0;   // index within that runtime, not the SYCL index
};

ProbeTarget probe_target_for(int device_ordinal)
{
    // Cached: the watchdog calls this every 20 ms, and the answer cannot
    // change during a run. Resolving it repeatedly would also mean touching
    // SYCL device enumeration from the polling thread on every tick.
    static std::mutex             m;
    static std::map<int, ProbeTarget> cache;
    {
        std::lock_guard<std::mutex> lk(m);
        if (auto it = cache.find(device_ordinal); it != cache.end())
            return it->second;
    }

    ProbeTarget t;
    try {
        auto const devs = sycl_backend::usable_gpu_devices();
        std::size_t idx = 0;
        bool        found = false;
        if (device_ordinal >= 0) {
            idx   = static_cast<std::size_t>(device_ordinal);
            found = idx < devs.size();
        } else {
            // kDefaultGpuId — whatever the default selector picks. Match it
            // back to an index so the per-runtime ordinal below is right.
            sycl::device const d{sycl::gpu_selector_v};
            for (std::size_t i = 0; i < devs.size(); ++i) {
                if (devs[i] == d) { idx = i; found = true; break; }
            }
        }
        if (found) {
            auto const backend = devs[idx].get_backend();
            ProbeRuntime const rt = (backend == sycl::backend::cuda) ? ProbeRuntime::Cuda
                                  : (backend == sycl::backend::hip)  ? ProbeRuntime::Hip
                                                                     : ProbeRuntime::None;
            if (rt != ProbeRuntime::None) {
                // Rank among devices sharing this backend — that IS the
                // runtime's own ordinal, whereas idx counts every vendor.
                int rank = 0;
                for (std::size_t i = 0; i < idx; ++i)
                    if (devs[i].get_backend() == backend) ++rank;
                t.runtime = rt;
                t.ordinal = rank;
            }
        }
    } catch (...) {
        // Device enumeration can throw on a half-broken driver. An
        // unprobeable device falls back to the device total, which is what
        // this function returning None means.
    }

    std::lock_guard<std::mutex> lk(m);
    cache[device_ordinal] = t;
    return t;
}

} // namespace

bool device_memory_probe(int device_ordinal,
                         size_t& free_bytes,
                         size_t& total_bytes)
{
    // The CPU device is the host. Ask the host.
    //
    // This branch did not exist, and the CUDA probe below folds every negative
    // ordinal onto GPU 0 (`device_ordinal < 0 ? 0 : device_ordinal`). That is
    // correct for kDefaultGpuId (-1), which really does mean "GPU 0" — but it
    // also swallowed kCpuDeviceId (-2), so the CPU device sized its streaming
    // tier against the GPU's free VRAM and logged "vram: peak N of 17690 free"
    // on a host with 125 GB of RAM. Nothing shipped on top of it yet; the
    // --cpu-workers RAM gate is about to, so it has to be right first.
    if (is_cpu_device(device_ordinal)) {
        return host_memory_probe(free_bytes, total_bytes);
    }

    // Ask only the runtime that actually owns this device. Asking both and
    // taking the first answer is what let the AMD probe speak for an Intel
    // card; see probe_target_for().
    ProbeTarget const target = probe_target_for(device_ordinal);

#ifdef XCHPLOT2_HAVE_CUB
    if (target.runtime == ProbeRuntime::Cuda) {
        std::size_t f = 0;
        std::size_t t = 0;
        if (cuda_query_device_memory(target.ordinal, f, t)) {
            free_bytes  = f;
            total_bytes = t;
            return true;
        }
    }
#endif
#ifndef _WIN32
    if (target.runtime == ProbeRuntime::Hip
        && hip_query_device_memory(target.ordinal, free_bytes, total_bytes)) {
        return true;
    }
#endif
    (void)target;
    (void)device_ordinal;
    (void)free_bytes;
    (void)total_bytes;
    // No runtime owns this device (Level Zero / OpenCL), or the owning one
    // declined. Caller falls back to the device total — an upper bound, and
    // honest about being one, which the wrong device's figure was not.
    return false;
}

size_t vram_safety_margin()
{
    static size_t const margin = [] () -> size_t {
        if (char const* v = std::getenv("POS2GPU_VRAM_MARGIN_MB"); v && v[0]) {
            size_t const mb = size_t(std::strtoull(v, nullptr, 10));
            if (mb > 0) return mb << 20;
        }
        return 128ULL << 20;
    }();
    return margin;
}

DeviceMemInfo query_device_memory()
{
    sycl::queue& q = sycl_backend::queue();
    DeviceMemInfo info;
    info.total_bytes =
        q.get_device().get_info<sycl::info::device::global_mem_size>();
    // Fallback: SYCL has no portable free-memory query and AdaptiveCpp exposes
    // none, so absent the driver probe below all we have is the device total.
    // Treat it as an upper bound; sycl::malloc_device remains the source of
    // truth.
    info.free_bytes = info.total_bytes;

    // Real free-memory query on the NVIDIA path. Without it free_bytes is just
    // the device total, so the tier picker sizes against memory the CUDA
    // context (~390 MB), the display server, and every other process on the
    // card have already taken — and hands out a tier that cannot fit. This is
    // what the file-header note refers to: the pool used to call cudaMemGetInfo
    // directly, and the call was dropped, not replaced, when GpuBufferPool.cu
    // became GpuBufferPool.cpp and the CUDA runtime became unavailable in a
    // SYCL-only TU.
    {
        size_t f = 0;
        size_t t = 0;
        int const ord = sycl_backend::current_device_id();
        if (device_memory_probe(ord, f, t)) {
            info.free_bytes  = f;
            info.total_bytes = t;
        }
    }

    if (char const* v = std::getenv("POS2GPU_MAX_VRAM_MB"); v && v[0]) {
        size_t const cap = size_t(std::strtoull(v, nullptr, 10)) * (1ULL << 20);
        info.free_bytes  = std::min(info.free_bytes,  cap);
        info.total_bytes = std::min(info.total_bytes, cap);
    }
    return info;
}

namespace {

// CUB's DeviceRadixSort temp_storage_bytes at k=28 with our key/val
// shape lands around 64-128 MB on sm_89; the streaming peak anchors
// below were measured with that overhead already live, so they
// implicitly budget for it. AdaptiveCpp's HIP backend routes the
// same `launch_sort_*` calls through a hand-rolled SYCL radix in
// SortSycl.cpp that uses ping-pong buffers sized to the input —
// multi-GiB at k=28, far exceeding what CUB's in-place radix needs.
// The streaming peak prediction has to add that excess so dispatch
// in BatchPlotter doesn't pick a tier whose "predicted peak" is
// several GiB short of the actual T1-sort live, the way an 8 GiB
// W5700 (gfx1010 → gfx1013 spoof) currently does.
//
// Baseline set at 256 MB at k=28 (a touch over CUB's typical scratch
// on sm_89 to keep headroom on NVIDIA cards near the threshold) and
// scaled 2× per +k step (linear in cap, matching how CUB's actual
// DeviceRadixSort scratch grows). The returned adjustment is
// `max(0, runtime_sort_scratch - baseline)`, so NVIDIA hosts whose
// runtime scratch is at or below the baseline see no change in
// predicted peak.
inline size_t streaming_sort_scratch_adjustment(int k)
{
    constexpr size_t cub_baseline_at_k28_bytes = 256ULL << 20;

    sycl::queue& q = sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    size_t const cap_for_k =
        max_pairs_per_section(k, num_section_bits) * (1ULL << num_section_bits);

    size_t s_pairs = 0;
    launch_sort_pairs_u32_u32(
        nullptr, s_pairs,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        cap_for_k, 0, k, q);
    size_t s_keys = 0;
    launch_sort_keys_u64(
        nullptr, s_keys,
        static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap_for_k, 0, 2 * k, q);
    size_t const actual = std::max(s_pairs, s_keys);

    int const dk = k - 28;
    size_t baseline = cub_baseline_at_k28_bytes;
    if (dk > 0)      baseline <<= dk;
    else if (dk < 0) baseline >>= -dk;

    return (actual > baseline) ? (actual - baseline) : 0;
}

} // namespace

size_t streaming_peak_bytes(int k)
{
    // Anchor: 5200 MB at k=28 (measured post-stage-4e on sm_89).
    // After the full T1/T2/T3 match/sort work (stages 1-4d) + Xs
    // gen+sort+pack inlining (4e), all match + sort phases cap out at
    // cap·sizeof(uint64_t) × ~2.5 aliases = ~5200 MB. Xs peak is 4128,
    // T3 sort 4228, all others ≤ 5200. Dominant terms scale with 2^k.
    constexpr size_t anchor_mb = 5200;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;       // floor for tiny test plots
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;  // cap halves per −1 in k → 2× smaller
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_plain_peak_bytes(int k)
{
    // Anchor: 7290 MB at k=28 (pre-stage-1-4 peak — d_t1_meta +
    // d_t1_keys_merged + d_t2_meta + d_t2_mi + d_t2_xbits all live
    // concurrently during T2 match, no parks). Plain tier skips all
    // park/rehydrate round-trips for ~400 ms/plot over compact at the
    // cost of this higher peak. Scales the same way as compact.
    constexpr size_t anchor_mb = 7290;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_minimal_peak_bytes(int k)
{
    // Anchor: 3900 MB at k=28 (streaming-stats trace reads 3884 MB on sm_89;
    // rounded up for safety). Bottleneck is T3 match where d_t2_keys_merged +
    // d_t2_xbits_sorted + meta-l/r slices + d_t3_stage are co-resident.
    //
    // Was 3760, from a 3754 MB trace taken when the anchor was first set. The
    // tracked peak has since drifted up to 3884 MB and the anchor was never
    // re-measured, so minimal had been quietly over its own budget by ~124 MB
    // independent of the two-phase scratch bug. See the warning on the
    // streaming_*_peak_bytes block in GpuBufferPool.hpp about calibrating
    // against the s_malloc trace.
    //
    // Minimal layers cumulative cuts on top of compact:
    //   1. N=8 T2 match staging (cap/8 ≈ 570 MB vs compact's cap/2).
    //   2. T1 sort gather, T2 sort meta+xbits gathers — tiled output,
    //      D2H per tile to host pinned, rebuild on device after free.
    //   3. T3 match — d_t2_meta_sorted parked on host pinned, sliced
    //      device buffers H2D'd per (section_l, section_r) pass.
    //   4. T1 match — sliced into N passes per section_l, output
    //      accumulated to host pinned.
    //   5. T1, T2, T3 sort CUB sub-phases — per-tile cap/N output
    //      buffers, USM-host accumulation, merges with USM-host inputs.
    //   6. Xs phase — gen+sort tiled in N=2 position halves with
    //      USM-host accumulators; pack tiled with D2H per tile.
    //
    // Cumulative effect at k=28: peak drops from 5200 MB (compact) →
    // 3884 MB (minimal). Trade-off: ~6 extra cap-sized PCIe round-
    // trips per plot (~2.5× wall on NVIDIA — 13 s/plot → 34 s/plot
    // at k=28). Same k-scaling as compact / plain.
    constexpr size_t anchor_mb = 3900;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_tiny_peak_bytes(int k)
{
    // Anchor: 1100 MB at k=28. Tiny absorbed the Phase 1.4 + 1.5
    // algorithms that were originally developed under the "Pinned"
    // tier name. After Phase 1.6 sub-section attacks (per-bucket-pair
    // T1/T2/T3 match + host-side T2/T3 prepare offsets), measured
    // direct on RTX 4090:
    //   k=22:  ~22 MB
    //   k=24:   92 MB
    //   k=26:  288 MB
    //   k=28: 1064 MB (direct measurement, not extrapolated)
    // Set anchor to 1100 MB — 3.4% safety margin above the measured
    // k=28 peak. The k=28 scaling came in 8% under the linear
    // k=26→k=28 extrapolation because fixed-size CUB scratch and
    // staging caps don't fully 4× with cap. The current floor is
    // T2 sort scratch (CUB tile_max-sized workspace).
    //
    // What Tiny now does (all the host-park + streaming techniques):
    //   - Xs: CPU merge+pack to host h_xs, no device d_xs_keys_b/vals_b
    //     intermediate (Phase 1.4a+b)
    //   - T1 match: per-section-pair tile H2D from h_xs, no full-cap
    //     d_xs on device (Phase 1.4c)
    //   - T1 sort: streaming partition (top-bits bucket) + per-bucket
    //     u32_u64 sort, no full-cap d_t1_meta on device (Phase 1.3c-ii)
    //   - T2 sort: streaming partition with triple-val (key/meta/xbits
    //     paired through duplicate keys) + per-bucket sort, no
    //     full-cap d_t2_meta on device (Phase 1.5b)
    //   - T3 match: d_t3_stage allocated as USM-host so device peak
    //     drops by ~200 MB at k=26 / ~800 MB at k=28 (Phase 1.5c-a)
    //   - T3 sort: N=4 tile + multi-way host merge (vs N=2 before)
    //
    // Wall trade vs the original (pre-promotion) Tiny implementation:
    // approximately +16% at k=26 on RTX 4090. Acceptable on target
    // hardware (2-3 GB GPUs) which couldn't run the original Tiny at
    // all. Larger cards should use Plain/Compact/Minimal which are
    // unchanged.
    //
    // Going below ~1.1 GB at k=28 requires attacking the T2 sort
    // CUB scratch (the new floor — 288 MB at k=26 / ~1152 MB at
    // k=28). Options: (a) finer per-bucket sort with smaller cub
    // scratch, (b) host-side merge of pre-sorted partition tiles,
    // (c) Phase 2 Disk tier for spill.
    constexpr size_t anchor_mb = 1100;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_pinned_peak_bytes(int k)
{
    // Anchor: 2900 MB at k=28. After Phase 1.3c-ii + 1.4a/b/c the
    // Pinned tier eliminates the T1-sort-gather floor (d_t1_meta
    // full-cap on device, ~2 GB at k=28) AND the Xs phase floor
    // (d_xs_keys_b + d_xs_vals_b + d_xs_pack_tile, ~3 GB at k=28,
    // plus the d_xs rehydrate ~2 GB). What remains is the T2 sort /
    // T3 match phase peak. Measured at k=26: Pinned 720 MB / Tiny
    // 792 MB → ~91% of Tiny. Extrapolated to k=28 (which scales
    // ~4× from k=26 for the dominant terms): Pinned ≈ 2900 MB.
    //
    // The spec's original 1500 MB-at-k=28 target is unreachable
    // without also streaming T2 sort and T3 match phases — those
    // are now the floor (project_streaming_pinned_disk_spec
    // memory documents the analysis). Phase 1.5+ would attack
    // those; not currently scoped.
    //
    // MEASURED at k=28 on an RTX 4090 (bench VramWatchdog, true process VRAM):
    // Pinned peaks at 1128 MB — against Tiny's 1118 MB on the same card. The
    // two tiers have the same footprint; Pinned is not "2x Tiny", and it is not
    // a rung above it.
    //
    // Every number in the paragraph above this one was derived, never measured
    // at k=28: 2900 was extrapolated from a k=26 reading, then lowered to 2200
    // as "3200 * 0.68" — where 3200 was Tiny's *old* anchor, itself ~3x Tiny's
    // real peak. A ratio applied to a wrong base gave a peak that over-declared
    // by 2x. Nothing caught it: the bench watchdog only fires when the true peak
    // exceeds what the tier declared, so over-declaring sails through silently
    // and merely costs the user card capability (a card that can run Pinned in
    // ~1.6 GB was told it needed 2.7 GB).
    //
    // Anchor 1150 = measured 1128 + a small allowance, matching how Tiny's 1100
    // sits over its measured 1064. Re-derive this ONLY from the watchdog's true
    // peak, never from a ratio against another anchor.
    constexpr size_t anchor_mb = 1150;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

namespace {

// Host bytes per cap entry, per tier. Table and provenance in the header.
//
// Derived by subtracting the fixed term from each measured peak RSS and
// dividing by cap(28) = 272,629,760:
//   plain   (7.35 - 1.27) GiB / cap = 24 B  — exactly the 3-slot D2H ring
//   compact (14.44 - 1.27)          = 52 B
//   minimal (18.52 - 1.27)          = 68 B  — gather-tiling holds h_t2_meta
//                                             live across T3 match
//   tiny    (21.48 - 1.27)          = 80 B
//
// Reconstructing gives 7.36 / 14.47 / 18.54 / 21.58 GiB — each within 0.11 GiB
// of its reading and every one on the high side, which is the safe direction
// for a gate.
//
// CALIBRATE AT n>=3, NOT n=1. Tiny is the only tier whose peak grows with batch
// depth: 19.56 GiB at n=1 against 21.48 at n=3, because the producer runs ahead
// of the file writer and tiny is the tier that also aliases the rotating D2H
// slots as device-visible working buffers. plain / compact / minimal measured
// identical at n=1 and n=3, so a single-plot calibration silently understates
// tiny alone — by 2 GiB, on the tier whose users have the least RAM to spare.
//
// The fixed term is what does not scale with cap: the CUDA/SYCL context, the
// binary, and the file writer's compressed-chunk heap. Keeping it separate
// matters at small k, where it dominates — a pure bytes-per-entry model would
// predict 81 MB for k=20 against an actual ~1.3 GiB, i.e. wrong in the unsafe
// direction for a gate.
constexpr size_t kHostFixedBytes = size_t(1300) << 20;

size_t streaming_host_bytes(int k, unsigned bytes_per_entry)
{
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    std::uint64_t const cap    = match_phase_capacity(k, num_section_bits);
    return size_t(cap) * bytes_per_entry + kHostFixedBytes;
}

}  // namespace

size_t streaming_plain_host_bytes(int k)   { return streaming_host_bytes(k, 24); }
size_t streaming_compact_host_bytes(int k) { return streaming_host_bytes(k, 52); }
size_t streaming_minimal_host_bytes(int k) { return streaming_host_bytes(k, 68); }
size_t streaming_tiny_host_bytes(int k)    { return streaming_host_bytes(k, 80); }

size_t host_memory_reserve()
{
    static size_t const reserve = [] () -> size_t {
        if (char const* v = std::getenv("XCHPLOT2_HOST_RESERVE_MB"); v && v[0]) {
            size_t const mb = size_t(std::strtoull(v, nullptr, 10));
            if (mb > 0) return mb << 20;
        }
        // Enough for the OS, a shell, and the writer's own heap growth after
        // the last pinned buffer lands. Not enough to protect a desktop
        // session — raise it if the plotting host is also driving one.
        return size_t(512) << 20;
    }();
    return reserve;
}

void host_pinned_reserve_check(size_t bytes, char const* what)
{
    size_t free_b = 0, total_b = 0;
    if (!host_memory_probe(free_b, total_b)) return;   // no probe → inert

    size_t const reserve = host_memory_reserve();
    if (bytes + reserve <= free_b) return;

    auto const gib = [](size_t b) { return b / double(1ULL << 30); };
    char msg[768];
    std::snprintf(msg, sizeof(msg),
        "pinned host allocation of %.2f GiB for %s would leave this machine "
        "under its %.2f GiB reserve — %.2f GiB available of %.2f GiB total. "
        "This is HOST memory, not VRAM: --tier bounds the device peak only, "
        "and a LOWER tier costs MORE host RAM, not less, so it will not help. "
        "The requirement is fixed for PoS2's k=28. Close what else is holding "
        "RAM, or plot on a host with more. "
        "(XCHPLOT2_HOST_RESERVE_MB overrides the reserve.)",
        gib(bytes), what, gib(reserve), gib(free_b), gib(total_b));
    throw std::runtime_error(msg);
}

} // namespace pos2gpu
