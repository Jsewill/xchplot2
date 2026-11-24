// The host worker fixture supplies a synthetic device implementation. Its
// recovery child still uses the production executable, IPC, admission and
// reference checking paths. No driver, device reset or plotting disk is needed.
#pragma once
#include <fstream>

namespace worker_gpu_test
{
inline bool child = false;
inline bool shared_load = false;
inline std::atomic<int> usable_mib{32}, resident_mib{0}, probes{0}, setups{0};
inline std::string active_tier;
inline std::atomic<bool> fast_buffers{false}, fail_trim{false};
inline std::atomic<uint64_t> scratch_limit{0};
inline std::atomic<int> tier_queries{0};
inline std::atomic<bool> entered{false}, release{false};
inline constexpr uint64_t failed_host_free = 1ull << 20, failed_device_free = 36ull << 20;
inline constexpr uint64_t async_host_free = 2ull << 20, async_device_free = 4ull << 20;
inline std::atomic<bool> lose_preflight{false};
inline std::filesystem::path directory()
{
    auto const* path = std::getenv("XCHPLOT2_TEST_GPU_STATE");
    if (!path) throw std::runtime_error("missing GPU fixture directory");
    return std::filesystem::path(reinterpret_cast<char8_t const*>(path));
}
inline void mark(char const* name) { std::ofstream(directory() / name).put('1'); }
inline void wait_until(auto condition, char const* message)
{
    auto const end = std::chrono::steady_clock::now() + std::chrono::seconds(25);
    while (!condition() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    group_tools::require(condition(), message);
}
}

namespace pos2gpu
{
std::vector<GpuDeviceInfo> list_gpu_devices()
{
    return {{0,"test GPU","level_zero",false,false,64ull<<20,4,true,"pci:0000:01:00.0"},
            {1,"test GPU","level_zero",false,false,64ull<<20,4,true,"pci:0000:02:00.0"}};
}
int gpu_device_count() { return 2; }
DeviceMemInfo query_device_memory() { return {64ull<<20,64ull<<20}; }
std::optional<size_t> worker_gpu_headroom(int, size_t resident, bool permit_runtime_only) {
    using namespace worker_gpu_test;
    if (!shared_load)
    {
        unsigned room=child?32:64;
        if (child) {std::ifstream change(directory()/"child-room"); if (change) change>>room;}
        return uint64_t(room)<<20;
    }
    ++probes;
    if (usable_mib<0) {
        if (permit_runtime_only) return query_device_memory().free_bytes;
        return {};
    }
    return uint64_t(std::max(0,usable_mib.load()-resident_mib.load()+int(resident>>20)))<<20;
}
}

namespace group_grow
{
struct GpuProducer::Impl
{
    int k, strength, device;
    std::atomic<bool> borrowed{false};
    std::vector<uint64_t> words;
    bool fail_cleanup = false, pending_queue_loss = false;
};
GpuProducer::GpuProducer(int k,int strength,GpuProducerOptions options)
    : impl_(new Impl{k,strength,options.device_id}) {
    if (worker_gpu_test::shared_load) {
        auto profile = preflight(k,strength,options);
        worker_gpu_test::active_tier = profile.tier;
        worker_gpu_test::fast_buffers = worker_budget_class(k,strength,profile,options.max_vram_bytes);
        worker_gpu_test::resident_mib = int(profile.device_bytes >> 20) - 2 + (worker_gpu_test::fast_buffers ? 10 : 0);
        worker_gpu_test::scratch_limit = options.max_vram_bytes - profile.device_bytes - (worker_gpu_test::fast_buffers ? 10ull<<20 : 0);
        ++worker_gpu_test::setups;
    }
}
GpuProducer::~GpuProducer()
{
    if (worker_gpu_test::shared_load) worker_gpu_test::resident_mib = 0;
    if (impl_->pending_queue_loss)
        pos2gpu::record_async_error("ze_node_event: Could not wait for event (error code = ze:1879048193)");
    if (!impl_->fail_cleanup) return;
    auto fail = [] { throw std::runtime_error("injected driver deallocation failure"); };
    pos2gpu::release_gpu_memory("test pinned output", worker_gpu_test::failed_host_free, &pos2gpu::GpuCleanupCharges::host, fail);
    pos2gpu::release_gpu_memory("test device output", worker_gpu_test::failed_device_free, &pos2gpu::GpuCleanupCharges::device, fail);
    // Native allocators can queue an error and return normally from sycl::free.
    auto report = [] { pos2gpu::record_async_error("ze_allocator: zeMemFree() failed (error code = ze:1879048193)"); };
    pos2gpu::release_gpu_memory("test async free", worker_gpu_test::async_host_free, &pos2gpu::GpuCleanupCharges::host, report);
    pos2gpu::release_gpu_memory("test remaining allocation", worker_gpu_test::async_device_free, &pos2gpu::GpuCleanupCharges::device,
        [&] { report(); worker_gpu_test::mark("last-free-attempted"); });
}
GpuProducerPreflight GpuProducer::preflight(int k, int, GpuProducerOptions const& options)
{
    if (worker_gpu_test::lose_preflight.exchange(false))
    {
        pos2gpu::record_async_error("ZE_RESULT_ERROR_DEVICE_LOST");
        throw std::runtime_error("device lost during preflight");
    }
    if (worker_gpu_test::shared_load ||
        (worker_gpu_test::child && std::filesystem::exists(worker_gpu_test::directory()/"child-tiers"))) {
        uint64_t available = pos2gpu::worker_gpu_headroom(options.device_id,0).value_or(options.max_vram_bytes);
        if (options.max_vram_bytes) available = std::min(available, options.max_vram_bytes);
        int const pool = k == 18 ? 40 : k == 20 ? 56 : 96;
        int const plain = k == 18 ? 12 : k == 20 ? 44 : 72;
        if ((options.tier == "auto" || options.tier == "pool") && available >= (uint64_t(pool)<<20))
            return {"pool", 12ull<<20, 16ull<<20, uint64_t(pool)<<20,"gpu"};
        if ((options.tier == "auto" || options.tier == "plain") && available >= (uint64_t(plain)<<20))
            return {"plain", 20ull<<20, 24ull<<20, uint64_t(plain)<<20,"gpu"};
        throw pos2gpu::InsufficientVramError("injected device headroom below selected tier");
    }
    if (k != 18 || options.max_vram_bytes < (16ull<<20)) throw pos2gpu::InsufficientVramError("insufficient test VRAM");
    return {"pool", 12ull<<20, 16ull<<20, 16ull<<20, pos2gpu::list_gpu_devices().at(options.device_id).physical_identity};
}
std::vector<GpuProducerPreflight> GpuProducer::preflight_tiers(int k, int strength, GpuProducerOptions const& options)
{
    ++worker_gpu_test::tier_queries;
    if (!worker_gpu_test::shared_load &&
        !(worker_gpu_test::child && std::filesystem::exists(worker_gpu_test::directory()/"child-tiers")))
    {
        try { return {preflight(k,strength,options)}; }
        catch (pos2gpu::InsufficientVramError const&) { return {}; }
    }
    uint64_t const pool = uint64_t(k == 18 ? 40 : k == 20 ? 56 : 96)<<20;
    uint64_t const plain = uint64_t(k == 18 ? 12 : k == 20 ? 44 : 72)<<20;
    std::vector<GpuProducerPreflight> result;
    if (options.tier=="auto" || options.tier=="pool") result.push_back({"pool",12ull<<20,16ull<<20,pool,"gpu"});
    if (options.tier=="auto" || options.tier=="plain") result.push_back({"plain",20ull<<20,24ull<<20,plain,"gpu"});
    return result;
}
uint64_t GpuProducer::runtime_host_allowance_bytes() noexcept { return 1ull<<20; }
uint64_t GpuProducer::host_generation_peak_bytes() const { return 16ull<<20; }
uint64_t GpuProducer::host_borrowed_generation_peak_bytes() const { return 12ull<<20; }
uint64_t GpuProducer::host_tiled_generation_peak_bytes() const { return 12ull<<20; }
unsigned GpuProducer::worker_budget_class(int, int, GpuProducerPreflight const& profile, uint64_t room)
{ return worker_gpu_test::shared_load && profile.tier=="pool" && room>=profile.device_bytes+(20ull<<20); }
bool GpuProducer::apply_device_budget(uint64_t room)
{
    using namespace worker_gpu_test;
    if (!shared_load) return true;
    uint64_t const floor=uint64_t(impl_->k == 18 ? (active_tier=="pool"?40:12) : (active_tier=="pool"?56:44))<<20;
    bool const next=active_tier=="pool" && room>=floor+(20ull<<20);
    if (room<floor || next!=fast_buffers) return false;
    auto const allowed=room-floor-(next?10ull<<20:0);
    if (allowed<scratch_limit && fail_trim.exchange(false))
        pos2gpu::checked_gpu_free(4ull<<20,&pos2gpu::GpuCleanupCharges::device,
            []{throw std::runtime_error("injected idle scratch release failure");});
    scratch_limit=allowed;
    return true;
}
uint64_t GpuProducer::device_resident_bytes() const { return uint64_t(worker_gpu_test::resident_mib.load()) << 20; }
uint64_t GpuProducer::retained_host_bytes() const { return runtime_host_allowance_bytes(); }
uint64_t GpuProducer::locked_host_output_bytes() const { return 0; }
void GpuProducer::release_host_output() {}
std::string const& GpuProducer::tier() const { static std::string const value = "pool"; return worker_gpu_test::shared_load ? worker_gpu_test::active_tier : value; }
bool GpuProducer::compact_transfer_available() const { return false; }
void GpuProducer::wait_for_output() {}
std::string GpuProducer::device_loss(std::exception_ptr, std::string_view) { return {}; } // fixture injects real async records
GpuProducer::BorrowedFragments::BorrowedFragments(std::span<uint64_t const> words,std::atomic<bool>& held)
    : fragments_(words), borrowed_(&held) { held.store(true); }
GpuProducer::BorrowedFragments::~BorrowedFragments() { if (borrowed_) borrowed_->store(false); }
GpuProducer::BorrowedFragments::BorrowedFragments(BorrowedFragments&& other) noexcept
    : fragments_(other.fragments_), borrowed_(std::exchange(other.borrowed_,nullptr)) {}
std::vector<uint64_t> GpuProducer::generate(std::array<uint8_t,32> const& id)
{
    if (worker_gpu_test::shared_load && id[0]==0xaa) {
        worker_gpu_test::entered=true;
        auto limit=Clock::now()+std::chrono::seconds(8);
        while (!worker_gpu_test::release && Clock::now()<limit) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!impl_->device && !worker_gpu_test::child && id[0] == 0xfe)
    {
        pos2gpu::record_async_error("ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY");
        throw pos2gpu::InsufficientVramError("out of test GPU memory");
    }
    if (!impl_->device && !worker_gpu_test::child && id[0] == 0xfa)
    {
        impl_->pending_queue_loss = true;
        throw std::runtime_error("unexpected generation output");
    }
    if (pos2gpu::async_error_count()) throw std::runtime_error("synthetic device retained an async error");
    if (!impl_->device && !worker_gpu_test::child && id[0] == 0xfd)
    {
        impl_->fail_cleanup = true;
        pos2gpu::record_async_error("ZE_RESULT_ERROR_DEVICE_LOST");
        throw std::runtime_error("generation failed after an asynchronous error");
    }
    return worker_original_reference(uint8_t(impl_->k), uint8_t(impl_->strength), id);
}
GpuProducer::BorrowedFragments GpuProducer::generate_borrowed(std::array<uint8_t,32> const& id)
{
    impl_->words = generate(id); return BorrowedFragments(impl_->words,impl_->borrowed);
}
void GpuProducer::generate_borrowed_into(std::array<uint8_t,32> const& id,
    std::function<void(std::span<uint64_t const>)> sink, std::function<void()> const& before)
{ auto result = generate(id); before(); sink(result); }
std::shared_future<uint64_t> GpuProducer::generate_packed_into_async(std::array<uint8_t,32> const&,
    std::function<void(PackedFragmentTile const&,uint64_t)>)
{ throw std::logic_error("synthetic device has no packed output"); }
uint64_t GpuProducer::generate_into(std::array<uint8_t,32> const& id,
    std::function<void(std::span<uint64_t const>,uint64_t)> const& sink)
{
    group_tools::require(worker_gpu_test::child && impl_->device == 0, "recovery reused the failed process or wrong GPU");
    if (id[0] == 0xfb)
    {
#ifdef _WIN32
        std::_Exit(73);
#else
        std::raise(SIGKILL);
#endif
    }
    if (id[0] == 0xfc)
    { worker_gpu_test::mark("child-started"); std::this_thread::sleep_for(std::chrono::seconds(60)); }
    auto result = generate(id);
    if (id[0] == 0 && !std::filesystem::exists(worker_gpu_test::directory() / "allow-probe"))
    { result.front() ^= 1; worker_gpu_test::mark("bad-probe"); }
    sink(result,result.size()); return result.size();
}
}

namespace worker_gpu_test
{
inline void recovery(bool network)
{
    pool_test::Directory directory;
    auto const path = pool_test::utf8_path(directory.path);
#ifdef _WIN32
    _putenv_s("XCHPLOT2_TEST_GPU_STATE",path.c_str());
#else
    ::setenv("XCHPLOT2_TEST_GPU_STATE",path.c_str(),1);
#endif
    auto options = config(); options.backend = "gpu"; options.capacity.cpu_threads = 4;
    options.capacity.slots = 3; options.capacity.host_bytes = 4ull<<30; options.device = 0;
    default_worker_resources(options);
    options.available_resources = [capacity = options.capacity](unsigned) { return capacity; };
    std::unique_ptr<RunningBroker> broker;
    std::optional<pool::BrokerClient> control;
    pool::OwnerSession session;
    auto sibling = options;
    if (network)
    {
        auto ca = pool_test::issue(directory.path,"gpu-ca");
        auto primary = pool_test::issue(directory.path,"gpu-worker",&ca);
        auto other = pool_test::issue(directory.path,"healthy-worker",&ca);
        auto owner = pool_test::issue(directory.path,"gpu-owner",&ca);
        options.tls = pool_test::tls(ca,primary); sibling.tls = pool_test::tls(ca,other);
        pool::BrokerOptions settings; settings.listen = {"127.0.0.1",0}; settings.tls = pool_test::tls(ca,owner);
        broker = std::make_unique<RunningBroker>(settings);
        options.broker = sibling.broker = {"127.0.0.1",broker->server.endpoint().port,"localhost"};
        options.listen = sibling.listen = {"127.0.0.1",0};
        control.emplace(options.broker,settings.tls); session = control->open_owner("gpu-fixture");
    }
    WorkerService service(options); sibling.device = 1; WorkerService healthy(sibling);
    auto routes = [&]
    {
        pool::Task task; task.task_id = "sample"; task.kind = pool::Kind::GenerateAny;
        task.compatibility = worker_compatibility(); task.k = 18; task.strength = 2;
        pool::BrokerClient::GenerationAvailability info; std::shared_ptr<pool::GenerationRoster const> peers;
        control->generation_hint(session,task,nullptr,&info,worker_compatibility(pool::Kind::Pack),&peers);
        return peers;
    };
    if (network)
    {
        service.start(); healthy.start();
        wait_until([&] { auto peers = routes(); return peers && peers->routes.size() == 2; },
            "both GPU workers did not register");
    }
    std::array<uint8_t,32> id{}; id.front() = 42;
    auto generate = [&](WorkerService& worker, pool::Kind kind, uint8_t value, WorkerCancellation cancel = {})
    {
        auto key = id; key.front() = value;
        uint64_t count = 0; worker.generate_local(18,2,key,[&](auto const&,uint64_t total) { count = total; },cancel,kind);
        require(count, "worker produced no fragments");
    };
    require(pool::worker_diagnostic("lost\n\x1b[2J") == "lost  [2J", "peer GPU diagnostics contain terminal controls");
    auto const kind = pool::Kind::GenerateGpu;
    {
        auto cuda = std::make_shared<pos2gpu::AsyncErrorLog>();
        auto ze = std::make_shared<pos2gpu::AsyncErrorLog>();
        pos2gpu::WorkerErrorRegistration a(*cuda,"cuda",[] {}), b(*ze,"level_zero",[] {});
        cuda->generating = true; ze->generating = true;
        pos2gpu::ScopedAsyncErrors use(cuda);
        pos2gpu::record_async_error("driver event wait failed (error code = ze:1879048193)");
        require(!cuda->count && ze->count == 1, "a foreign queue mistook a Level Zero fault for a CUDA fault");
        auto idle = std::make_shared<pos2gpu::AsyncErrorLog>();
        pos2gpu::WorkerErrorRegistration c(*idle,"level_zero",[] {});
        pos2gpu::ScopedAsyncErrors late(idle);
        pos2gpu::record_async_error("ZE_RESULT_ERROR_DEVICE_LOST");
        require(idle->count == 1 && ze->count == 2, "late device loss skipped its own reporting queue");
    }
    {
        auto isolated = options; isolated.broker = {}; isolated.brokers.clear();
        WorkerService late(isolated);
        bool retry = false;
        try { generate(late,kind,0xfa); }
        catch (pool::TransientTransport const&) { retry = true; }
        catch (std::runtime_error const&) {}
        auto const status = late.status();
        require(retry && !status.capacity.gpu_bytes && status.gpu_recovery.find("ze:1879048193") != std::string::npos,
            "late queue loss did not release the first task for retry with the native GPU withdrawn");
    }
    rejects([&] { generate(service,kind,0xfe); });
    require(service.status().gpu_recovery.empty() && service.status().capacity.gpu_bytes, "an allocation failure withdrew the GPU");
    generate(service,kind,41);
    auto const baseline = service.status().baseline_host_bytes;
    std::atomic<bool> entered{false}, release{false};
    struct Release { std::atomic<bool>& value; ~Release() { value.store(true); } };
    auto pending = std::async(std::launch::async,[&] {
        service.generate_local(18,2,id,[&](auto const&,uint64_t) {
            entered.store(true);
            auto const limit = Clock::now() + std::chrono::seconds(20);
            while (!release.load() && Clock::now() < limit) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        },{},pool::Kind::GenerateCpu);
    });
    Release cleanup{release};
    wait_until([&] { return entered.load(); }, "CPU task never began");
    rejects([&] { generate(service,kind,0xfd); });
    auto status = service.status();
    require(std::filesystem::exists(directory.path / "last-free-attempted") &&
        status.baseline_host_bytes == baseline + failed_host_free + async_host_free, "failed frees skipped cleanup or released uncertain host pages");
    require(status.admission == pool::WorkerAdmission::Accepting && status.accepted_tasks == 1 && !status.capacity.gpu_bytes &&
        status.capacity.cpu_threads > 0 && status.gpu_recovery == "ZE_RESULT_ERROR_DEVICE_LOST" &&
        !service.local_generation_offer(18,2,1).demand.slots, "fault did not withdraw only the GPU");
    if (network)
    {
        auto direct = WorkerService::remote_status(service.endpoint(),options.tls);
        require(direct.gpu_recovery == status.gpu_recovery, "direct authenticated status lost the GPU failure");
        wait_until([&] {
            auto page = control->status({});
            return page.worker_records.size() == 2 && std::any_of(page.worker_records.begin(),page.worker_records.end(),[&](auto const& row)
                { return row.instance == status.instance && !row.capacity.gpu_bytes && row.capacity.cpu_threads && row.gpu_recovery == status.gpu_recovery; });
        }, "broker lost the withdrawn GPU or its CPU partition");
        auto peers = routes();
        require(peers && peers->routes.size() == 2 && peers->recovering_gpus.size() == 1 &&
            peers->recovering_gpus.front().cause == status.gpu_recovery &&
            std::any_of(peers->routes.begin(),peers->routes.end(),[&](auto const& row)
                { return row.instance == status.instance && row.cpu_slots && !row.gpu_slots; }) &&
            std::any_of(peers->routes.begin(),peers->routes.end(),[&](auto const& row)
                { return row.instance != status.instance && row.gpu_slots; }), "dashboard roster lost independent CPU/GPU capacity");
    }
    release.store(true); pending.get();
    generate(healthy,kind,42); generate(service,pool::Kind::GenerateCpu,42);
    require(healthy.status().gpu_recovery.empty() && !pos2gpu::async_error_count(), "device fault contaminated another execution lane");
    wait_until([&] { return std::filesystem::exists(directory.path / "bad-probe"); }, "fresh GPU probe did not run");
    require(!service.status().capacity.gpu_bytes, "worker re-admitted an incorrect GPU result");
    auto paused = service.control(WorkerControl::Pause);
    mark("allow-probe");
    require(paused.admission == pool::WorkerAdmission::Paused && !paused.gpu_recovery.empty(), "operator pause masked device loss");
    service.control(WorkerControl::Resume);
    wait_until([&] { return service.status().capacity.gpu_bytes; }, "verified fresh GPU was never re-admitted");
    require(service.status().capacity.gpu_bytes == (64ull<<20) - failed_device_free - async_device_free &&
        service.status().baseline_host_bytes == baseline + failed_host_free + async_host_free && service.status().gpu_recovery.empty() &&
        service.local_generation_offer(18,2,1).demand.slots == 1, "recovery ignored live GPU capacity");
    if (network)
        wait_until([&] { auto peers = routes(); return peers && peers->recovering_gpus.empty() &&
            std::all_of(peers->routes.begin(),peers->routes.end(),[](auto const& route) { return route.gpu_slots != 0; }); },
            "broker kept the recovery advisory after verified re-admission");
    generate(service,kind,42);
    std::atomic<bool> stop{false};
    auto cancelled = std::async(std::launch::async,[&] { rejects([&] { generate(service,kind,0xfc,{&stop}); }); });
    wait_until([&] { return std::filesystem::exists(directory.path / "child-started"); }, "isolated member did not start");
    stop.store(true);
    require(cancelled.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "cancelled child stayed active"); cancelled.get();
    require(service.status().capacity.gpu_bytes, "cancellation withdrew a healthy GPU");
    rejects([&] { generate(service,kind,0xfb); });
    auto crashed = service.status();
    require(!crashed.capacity.gpu_bytes && (crashed.gpu_recovery.find("signal ") != std::string::npos ||
        crashed.gpu_recovery.find("exit") != std::string::npos), "an exited GPU child stayed advertised");
    wait_until([&] { return service.status().capacity.gpu_bytes; }, "a crashed recovery child was never reprobed");
    { std::ofstream(directory.path/"child-room")<<8; }
    rejects([&]{generate(service,kind,41);});
    require(!service.status().capacity.gpu_bytes && service.status().capacity.cpu_threads,
        "recovered worker repeatedly offered a tier its fresh process could no longer fit");
    { std::ofstream(directory.path/"child-room")<<32; }
    wait_until([&]{return service.status().capacity.gpu_bytes;},"fresh worker did not recover after adapter memory returned");
    generate(service,kind,42);

    auto startup = options; startup.broker = {}; startup.brokers.clear();
    std::atomic<unsigned> before_publish{0}; std::atomic<bool> published{false};
    startup.reserve_gpu_probe = [&](uint64_t) -> std::shared_ptr<void> {
        if (!published.load()) { ++before_publish; return {}; }
        return std::make_shared<int>(1);
    };
    lose_preflight.store(true); WorkerService interrupted(startup);
    require(!interrupted.status().capacity.gpu_bytes && interrupted.status().capacity.cpu_threads,
        "preflight device loss removed CPU capacity");
    wait_until([&] { return before_publish.load(); }, "preflight fault skipped private probe admission");
    require(!interrupted.status().capacity.gpu_bytes, "private probe bypassed the shared RAM grant");
    published.store(true);
    wait_until([&] { return interrupted.status().capacity.gpu_bytes; }, "worker could not recover without an original GPU profile");
    generate(interrupted,kind,42);

    if (network)
    {
        service.stop(); healthy.stop();
        auto same_budget = options; same_budget.capacity.gpu_bytes = 32ull << 20;
        WorkerService late_registration(same_budget);
        auto const original = late_registration.local_generation_offer(18,2,1);
        rejects([&] { generate(late_registration,kind,0xfa); });
        wait_until([&] { return late_registration.status().capacity.gpu_bytes; }, "late registration never recovered");
        auto const current = late_registration.local_generation_offer(18,2,1);
        require(current.capacity == original.capacity && current.demand.host_bytes > original.demand.host_bytes,
            "fixture missed the isolated host allocation at unchanged capacity");
        late_registration.start();
        pool::Task task; task.task_id = "same-budget-recovery"; task.sequence = 1;
        task.compatibility = worker_compatibility(); task.kind = kind; task.k = 18; task.strength = 2;
        task.resources.host_bytes = worker_cpu_task_bytes(18,2,1,same_budget.max_task_bytes);
        std::optional<pool::Lease> lease;
        wait_until([&] { if (!lease) lease = control->acquire(session,task); return bool(lease); },
            "broker did not admit the recovered worker");
        require(lease->worker_instance == late_registration.status().instance && lease->task.resources == current.demand,
            "broker reissued the native grant after recovery changed the required host memory");
        control->cancel(session,task.task_id,task.attempt);
    }
}
}

namespace worker_gpu_test {
inline void shared_capacity()
{
    shared_load = true;
    pool_test::Directory directory;
    auto path = pool_test::utf8_path(directory.path);
#ifdef _WIN32
    _putenv_s("XCHPLOT2_TEST_GPU_STATE", path.c_str());
#else
    ::setenv("XCHPLOT2_TEST_GPU_STATE", path.c_str(), 1);
#endif
    std::ofstream(directory.path / "allow-probe").put('1');
    auto ca = pool_test::issue(directory.path, "shared-ca");
    auto node = pool_test::issue(directory.path, "shared-worker", &ca);
    auto owner = pool_test::issue(directory.path, "shared-owner", &ca);
    auto options = config();
    options.backend = "gpu";
    options.capacity.cpu_threads = 4;
    options.generation_threads = 2;
    default_worker_resources(options);
    options.available_resources = [capacity = options.capacity](unsigned) { return capacity; };
    pool::BrokerOptions settings;
    settings.listen = {"127.0.0.1", 0};
    settings.tls = pool_test::tls(ca, owner);
    settings.worker_ms = 600;
    settings.lease_ms = 12000;
    RunningBroker broker(settings);
    options.broker = {"127.0.0.1", broker.server.endpoint().port, "localhost"};
    options.listen = {"127.0.0.1", 0};
    options.tls = pool_test::tls(ca, node);
    WorkerService worker(options);
    worker.start();
    pool::BrokerClient control(options.broker, options.tls);
    auto session = control.open_owner("shared");
    auto route = [&](uint8_t k)
    {
        pool::Task t;
        t.task_id = "inspect";
        t.kind = pool::Kind::GenerateGpu;
        t.k = k;
        t.strength = 2;
        t.compatibility = worker_compatibility();
        t.generation_output = pool::GenerationOutput::Contributions;
        pool::BrokerClient::GenerationAvailability current;
        std::shared_ptr<pool::GenerationRoster const> peers;
        control.generation_hint(session, t, nullptr, &current, worker_compatibility(pool::Kind::Pack), &peers);
        return std::pair{current, peers};
    };
    auto eventually = [&](auto test)
    {
        auto const until = Clock::now() + std::chrono::seconds(4);
        do
        {
            if (test()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        } while (Clock::now() < until);
        return bool(test());
    };
    unsigned sequence = 0;
    auto request = [&](unsigned width, bool independent = false, bool packed = false)
    {
        pool::Task task;
        task.sequence = ++sequence;
        task.task_id = "gpu-capacity-" + std::to_string(sequence);
        task.k = 18;
        task.strength = 2;
        task.kind = pool::Kind::GenerateGpu;
        task.compatibility = worker_compatibility();
        task.resources = {0, 1, 1, 0};
        task.work_units = width;
        task.independent_members = independent;
        task.generation_output = packed ? pool::GenerationOutput::Packed : pool::GenerationOutput::Contributions;
        return task;
    };
    auto grant = [&](pool::Task const& task, bool expected)
    {
        auto lease = control.acquire(session, task);
        if (expected)
            require(lease.has_value(), "broker did not sign feasible generation shape");
        else
            require(!lease, "broker signed a generation shape unsupported at this tier");
        return lease;
    };
    auto generate = [&](unsigned k)
    {
        std::array<uint8_t, 32> id{};
        id.front() = 42;
        worker.generate_local(k, 2, id, [](auto const&, uint64_t) {}, {}, pool::Kind::GenerateGpu);
    };
    require(eventually([&] { return route(18).second && !route(18).second->routes.empty(); }), "missing initial GPU");
    require(!worker.local_generation_offer(20, 2, 1).demand.slots, "new route already fits at startup");
    auto initial_joint = request(2);
    auto joint = control.acquire(session, initial_joint);
    require(joint.has_value(), "streaming tier lost feasible joined batch");
    control.cancel(session, initial_joint.task_id, initial_joint.attempt);
    generate(18);
    require(active_tier == "plain", "failed to use initial small tier");
    auto lower = worker.local_generation_offer(18, 2, 1);
    usable_mib = 64;
    require(eventually(
                [&]
                { return !route(20).second->routes.empty() && worker.local_generation_offer(20, 2, 1).demand.slots; }),
            "released VRAM did not admit new k on the same worker");
    auto higher = worker.local_generation_offer(18, 2, 1);
    require(higher.demand.host_bytes < lower.demand.host_bytes, "upgraded tier did not reduce host demand");
    require(route(18).first.independent_width(UINT32_MAX) > 1, "pooled tier missing independent batching");
    generate(18);
    require(active_tier == "pool", "existing small producer failed to upgrade");
    auto old_setups = setups.load();
    auto const priced = worker.local_generation_offer(18, 2, 1);
    auto observations = probes.load();
    auto const initial_queries = tier_queries.load();
    usable_mib = 62;
    require(eventually(
                [&]
                {
                    (void)worker.local_generation_offer(18, 2, 1);
                    return probes > observations;
                }),
            "idle observer never sampled shared GPU");
    require(worker.local_generation_offer(18, 2, 1).demand == priced.demand && tier_queries == initial_queries,
            "equivalent headroom churn changed the signed offer or repeated native sizing");
    require(eventually([&] { return worker.local_generation_offer(20, 2, 1).demand.slots; }),
            "retained pool excluded another feasible size");
    generate(18);
    require(setups == old_setups, "owned resident buffers caused needless rebuild");
    require(fast_buffers, "fully available pool did not use faster optional output");
    usable_mib = 59;
    require(eventually(
                [&]
                {
                    auto available = worker.local_generation_offer(18, 2, 1);
                    return available.demand.gpu_bytes == (59ull << 20);
                }),
            "same-tier loss of optional buffers did not reach admission");
    generate(18);
    require(active_tier == "pool" && !fast_buffers && scratch_limit <= (19ull << 20),
            "same-tier reuse exceeded the smaller GPU envelope");
    auto smaller_task = request(1, false, true);
    auto smaller = grant(smaller_task, true);
    require(smaller->task.resources.gpu_bytes == (59ull << 20), "broker missed the smaller signed GPU envelope");
    usable_mib = 64;
    require(eventually(
                [&]
                {
                    auto offer = worker.local_generation_offer(18, 2, 1);
                    return offer.demand.gpu_bytes == (64ull << 20);
                }),
            "same-tier optional buffers never became available again");
    generate(18);
    require(fast_buffers, "recovered memory did not restore optional buffers");
    std::array<uint8_t, 32> old_id{};
    old_id.front() = 42;
    worker.generate_packed_local(
        18, 2, {&old_id, 1}, [](auto const&, uint64_t) {}, {}, pool::Kind::GenerateGpu, {}, &smaller->task.resources);
    require(!fast_buffers && scratch_limit <= (19ull << 20),
            "warm native producer exceeded an older signed smaller grant");
    control.cancel(session, smaller_task.task_id, smaller_task.attempt);
    generate(18);
    require(fast_buffers, "subsequent job retained a previous lease's smaller envelope");
    auto running_task = request(1, false, true);
    auto running = grant(running_task, true);
    std::array<uint8_t, 32> id{};
    id.front() = 42;
    auto run_signed = [&](pool::Lease const& lease, uint8_t first)
    {
        std::array<uint8_t, 32> plot{};
        plot.front() = first;
        auto channel = pool::Channel::connect(lease.endpoint, options.tls, lease.worker_identity);
        pool::Writer body(options.tls.max_frame_bytes);
        pool::write_lease(body, lease);
        body.u32(15000);
        body.u8(uint8_t(lease.task.generation_output == pool::GenerationOutput::Packed ? Operation::Packed
                                                                                       : Operation::Contributions));
        body.u8(lease.task.k);
        body.u8(lease.task.strength);
        body.u16(lease.task.work_units);
        for (unsigned i = 0; i < lease.task.work_units; ++i)
        {
            plot.back() = uint8_t(i);
            body.bytes(plot);
        }
        channel.send(pool::Message::Work, body.data());
        for (;;)
        {
            auto frame = channel.receive();
            if (frame.type == pool::Message::End) return;
            if (frame.type == pool::Message::Error) throw_worker_error(frame.payload);
        }
    };
    auto active = std::async(std::launch::async, [&] { run_signed(*running, 0xaa); });
    require(eventually([&] { return entered.load(); }), "signed GPU lease did not begin");
    usable_mib = 32;
    bool const protected_lease = eventually(
        [&]
        {
            auto p = control.status({});
            auto observed = worker.status();
            return observed.active.gpu_bytes == running->task.resources.gpu_bytes &&
                   observed.capacity.gpu_bytes >= observed.active.gpu_bytes && !p.worker_records.empty() &&
                   p.worker_records.front().reserved.gpu_bytes == running->task.resources.gpu_bytes;
        });
    release = true;
    active.get();
    require(protected_lease, "active GPU lease lost its signed reservation");
    require(eventually(
                [&]
                { return route(20).second->routes.empty() && !worker.local_generation_offer(20, 2, 1).demand.slots; }),
            "external usage did not withdraw newly infeasible k");
    require(route(18).first.independent_width(UINT32_MAX) < 2,
            "obsolete pool remained eligible for independent batching");
    require(worker.local_generation_offer(18, 2, 1).demand.host_bytes >= lower.demand.host_bytes,
            "reduced tier omitted larger host model");
    generate(18);
    require(active_tier == "plain", "ordinary generation refused smaller feasible tier");
    auto down = request(2, false);
    auto arranged = control.acquire(session, down);
    require(arranged && arranged->task.resources == worker.local_generation_offer(18, 2, 2).demand,
            "refreshed streaming joint grant omitted its host model");
    control.cancel(session, down.task_id, down.attempt);
    auto single = request(1, true);
    (void)grant(single, true);
    control.cancel(session, single.task_id, single.attempt);
    auto none = request(2, true);
    (void)grant(none, false);
    control.cancel(session, none.task_id, none.attempt);
    usable_mib = 64;
    require(eventually([&] { return route(18).first.independent_width(UINT32_MAX) > 1; }),
            "independent shape did not recover");
    auto wide = request(2, true);
    auto pooled = grant(wide, true);
    require(pooled->task.resources == worker.local_generation_offer(18, 2, 1).demand,
            "independent batch did not retain singleton grant");
    control.cancel(session, wide.task_id, wide.attempt);
    auto stale_task = request(1, false, true);
    auto stale = grant(stale_task, true);
    usable_mib = 32;
    require(
        eventually(
            [&]
            { return worker.local_generation_offer(18, 2, 1).demand.host_bytes > stale->task.resources.host_bytes; }),
        "worker failed to observe pressure during a granted lease");
    {
        auto until = Clock::now() + std::chrono::milliseconds(1400);
        while (Clock::now() < until)
        {
            auto page = control.status({});
            require(!page.worker_records.empty() &&
                        page.worker_records.front().reserved.gpu_bytes == stale->task.resources.gpu_bytes,
                    "unstarted GPU reservation expired the whole worker");
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
        }
    }
    {
        WorkerClientConfig local;
        local.broker = options.broker;
        local.tls = options.tls;
        local.owner_id = "shared-local-cpu";
        local.generation_kind = pool::Kind::GenerateCpu;
        local.generation_threads = 1;
        local.local_service = &worker;
        bool delivered = false;
        local.observe = [&](auto const& observation) { delivered = observation.local_direct; };
        WorkerClient cpu(local);
        cpu.generate(18, 2, id, [](auto const&, uint64_t) {});
        require(delivered, "CPU fixture did not finish through the embedded worker");
        auto page = control.status({});
        require(!page.worker_records.empty() && page.worker_records.front().reserved == stale->task.resources,
                "a fenced GPU capacity refresh stranded an unrelated completed CPU lease");
    }
    bool declined = false;
    try
    {
        run_signed(*stale, 42);
    }
    catch (pool::UnsupportedResources const&)
    {
        declined = true;
    }
    require(declined, "worker used a lower tier that exceeded its signed host grant");
    ++stale_task.attempt;
    std::optional<pool::Lease> retry;
    require(eventually(
                [&]
                {
                    retry = control.acquire(session, stale_task);
                    return retry.has_value();
                }),
            "a repriced singleton permanently excluded the only GPU for this logical task");
    require(retry->worker_instance == stale->worker_instance &&
                retry->task.resources.host_bytes > stale->task.resources.host_bytes,
            "same-job retry missed the refreshed host price");
    run_signed(*retry, 42);
    usable_mib = -1;
    require(eventually(
                [&]
                {
                    auto state = worker.local_generation_offer(18, 2, 1);
                    return !state.demand.slots && !state.capacity.gpu_bytes;
                }),
            "automatic worker reused an unavailable measurement");
    auto const parked = worker.local_generation_offer(18, 2, 1);
    require(parked.gpu_headroom_limited && parked.demand.host_bytes == higher.demand.host_bytes &&
                !worker.local_generation_offer(22, 2, 1).gpu_headroom_limited,
            "private worker confused temporary headroom with an impossible workload");
    usable_mib = 64;
    require(eventually([&] { return worker.local_generation_offer(18, 2, 1).demand.slots; }),
            "no native GPU after measurement recovered");
    generate(18);
    {
        std::ofstream(directory.path / "child-tiers").put('1');
        std::ofstream(directory.path / "child-room") << 24;
    }
    fail_trim = true;
    usable_mib = 62;
    require(eventually(
                [&]
                {
                    auto h = worker.local_generation_offer(18, 2, 1);
                    return !h.capacity.gpu_bytes && !h.gpu_recovery.empty();
                }),
            "a failed idle scratch release left the damaged producer advertised");
    wait_until(
        [&]
        {
            auto h = worker.local_generation_offer(18, 2, 1);
            return h.capacity.gpu_bytes == (24ull << 20) && h.demand.slots &&
                   route(18).first.independent_width(UINT32_MAX) > 1;
        },
        "fresh streamed child lost the originally supported independent shape");
    auto const low_child = worker.local_generation_offer(18, 2, 1);
    auto independent_child = grant(request(2, true), true);
    require(independent_child->task.resources == low_child.demand,
            "isolated independent contributions were charged for simultaneous members");
    run_signed(*independent_child, 42);
    {
        WorkerClientConfig local;
        local.broker = options.broker;
        local.tls = options.tls;
        local.owner_id = "recovered-independent";
        local.generation_kind = pool::Kind::GenerateGpu;
        local.generation_threads = 2;
        local.local_service = &worker;
        bool direct = false;
        local.observe = [&](auto const& observation)
        {
            direct = observation.local_direct && observation.independent_members &&
                     observation.generation_resources == low_child.demand;
        };
        WorkerClient client(local);
        std::array<std::array<uint8_t, 32>, 2> ids{};
        ids[0].front() = 42;
        ids[1].front() = 43;
        size_t chunks = 0;
        client.contributions(
            18, 2, ids,
            [&](uint64_t ordinal, auto bytes, uint16_t members)
            {
                require(ordinal == chunks++ && !bytes.empty() && members == 1,
                        "isolated independent members lost their individual contribution framing");
            },
            {}, {}, {}, {}, true);
        require(direct && chunks == 2 * PlotGroupFile::getChunkCountForK(18),
                "embedded recovered work lost its independently signed streamed members");
    }
    auto missing = worker.local_generation_offer(20, 2, 1);
    require(!missing.demand.slots && missing.gpu_headroom_limited && missing.demand.host_bytes,
            "isolated admission treated a temporarily unavailable k as permanently impossible");
    auto larger_task = request(1, false, true);
    larger_task.k = 20;
    (void)grant(larger_task, false);
    {
        std::ofstream(directory.path / "child-room") << 64;
    }
    std::optional<pool::Lease> larger;
    wait_until(
        [&]
        {
            larger = control.acquire(session, larger_task);
            return larger.has_value();
        },
        "an idle recovered GPU ignored released memory while only a larger workload was queued");
    auto const restored = worker.local_generation_offer(20, 2, 1);
    require(larger->worker_instance == worker.status().instance && restored.demand == larger->task.resources &&
                restored.demand.gpu_bytes == (60ull << 20) && missing.demand.host_bytes == restored.demand.host_bytes,
            "fresh process lost its unconfirmed allocation charge or its minimum host estimate");
    run_signed(*larger, 42);
    usable_mib = 0;
    worker.stop();
    auto capped = config();
    capped.backend = "gpu";
    capped.capacity.gpu_bytes = 40ull << 20;
    capped.tier = "pool";
    capped.available_resources = [capacity = capped.capacity](unsigned) { return capacity; };
    WorkerService explicit_worker(capped);
    require(!explicit_worker.local_generation_offer(18, 2, 1).demand.slots,
            "explicit cap treated exhausted memory as unknown");
    usable_mib = 32;
    require(!explicit_worker.local_generation_offer(18, 2, 1).demand.slots,
            "explicit pool tier fell back to streaming");
    usable_mib = 64;
    require(eventually(
                [&]
                {
                    auto offer = explicit_worker.local_generation_offer(18, 2, 1);
                    return offer.demand.slots && offer.capacity.gpu_bytes == (40ull << 20);
                }),
            "explicit GPU partition was not restored within its cap");
    usable_mib = -1;
    require(eventually(
                [&]
                {
                    auto offer = explicit_worker.local_generation_offer(18, 2, 1);
                    return offer.demand.gpu_bytes == (40ull << 20);
                }),
            "explicit cap lost the supported ordinary runtime fallback");
    capped.tier = "auto";
    capped.capacity.gpu_bytes = 96ull << 20;
    usable_mib = 32;
    WorkerService oversized(capped);
    require(!oversized.local_generation_offer(22, 2, 1).gpu_headroom_limited &&
                oversized.local_generation_offer(20, 2, 1).gpu_headroom_limited,
            "a cap larger than physical memory turned permanent infeasibility into a wait");
}
}
