// The host worker fixture supplies a synthetic device implementation. Its
// recovery child still uses the production executable, IPC, admission and
// reference checking paths. No driver, device reset or plotting disk is needed.
#pragma once
#include <fstream>

namespace worker_gpu_test
{
inline bool child = false;
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
uint64_t worker_gpu_available_bytes(int) { return uint64_t(worker_gpu_test::child ? 32 : 64) << 20; }
}

namespace group_grow
{
struct GpuProducer::Impl
{
    int k, strength, device;
    std::atomic<bool> borrowed{false};
    std::vector<uint64_t> words;
};
GpuProducer::GpuProducer(int k,int strength,GpuProducerOptions options)
    : impl_(new Impl{k,strength,options.device_id}) {}
GpuProducer::~GpuProducer() = default;
GpuProducerPreflight GpuProducer::preflight(int k, int, GpuProducerOptions const& options)
{
    if (k != 18 || options.max_vram_bytes < (16ull<<20)) throw pos2gpu::InsufficientVramError("insufficient test VRAM");
    return {"pool", 12ull<<20, 16ull<<20, 16ull<<20, pos2gpu::list_gpu_devices().at(options.device_id).physical_identity};
}
uint64_t GpuProducer::runtime_host_allowance_bytes() noexcept { return 1ull<<20; }
uint64_t GpuProducer::host_generation_peak_bytes() const { return 16ull<<20; }
uint64_t GpuProducer::host_borrowed_generation_peak_bytes() const { return 12ull<<20; }
uint64_t GpuProducer::host_tiled_generation_peak_bytes() const { return 12ull<<20; }
uint64_t GpuProducer::retained_host_bytes() const { return runtime_host_allowance_bytes(); }
uint64_t GpuProducer::locked_host_output_bytes() const { return 0; }
void GpuProducer::release_host_output() {}
std::string const& GpuProducer::tier() const { static std::string const value = "pool"; return value; }
bool GpuProducer::compact_transfer_available() const { return false; }
void GpuProducer::wait_for_output() {}
std::string GpuProducer::device_loss(std::exception_ptr) { return {}; } // fixture injects real async records
GpuProducer::BorrowedFragments::BorrowedFragments(std::span<uint64_t const> words,std::atomic<bool>& held)
    : fragments_(words), borrowed_(&held) { held.store(true); }
GpuProducer::BorrowedFragments::~BorrowedFragments() { if (borrowed_) borrowed_->store(false); }
GpuProducer::BorrowedFragments::BorrowedFragments(BorrowedFragments&& other) noexcept
    : fragments_(other.fragments_), borrowed_(std::exchange(other.borrowed_,nullptr)) {}
std::vector<uint64_t> GpuProducer::generate(std::array<uint8_t,32> const& id)
{
    if (!impl_->device && !worker_gpu_test::child && id[0] == 0xfe)
    {
        pos2gpu::record_async_error("ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY");
        throw pos2gpu::InsufficientVramError("out of test GPU memory");
    }
    if (pos2gpu::async_error_count()) throw std::runtime_error("synthetic device retained an async error");
    if (!impl_->device && !worker_gpu_test::child && id[0] == 0xfd)
    {
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
    auto const kind = pool::Kind::GenerateGpu;
    {
        auto cuda = std::make_shared<pos2gpu::AsyncErrorLog>();
        auto ze = std::make_shared<pos2gpu::AsyncErrorLog>();
        pos2gpu::WorkerErrorRegistration a(*cuda,"cuda",[] {}), b(*ze,"level_zero",[] {});
        cuda->generating = true; ze->generating = true;
        pos2gpu::ScopedAsyncErrors use(cuda);
        pos2gpu::record_async_error("driver event wait failed (error code = ze:1879048193)");
        require(!cuda->count && ze->count == 1, "a foreign queue mistook a Level Zero fault for a CUDA fault");
    }
    rejects([&] { generate(service,kind,0xfe); });
    require(service.status().gpu_recovery.empty() && service.status().capacity.gpu_bytes, "an allocation failure withdrew the GPU");
    generate(service,kind,41);
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
    require(service.status().capacity.gpu_bytes == (32ull<<20) && service.status().gpu_recovery.empty() &&
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
}
}
