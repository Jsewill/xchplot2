// Like the grow engine tests, include the implementation to exercise private
// bounded execution paths without adding a public testing-only interface.
#include "host/ComputeWorker.cpp"
#include "compute_pool_fixture.hpp"
#include <iostream>
#ifdef _WIN32
#include <tlhelp32.h>
#else
#include <dirent.h>
#endif

using namespace group_grow;
using group_tools::require;
std::vector<uint64_t> worker_original_reference(uint8_t, uint8_t, std::array<uint8_t, 32> const&);

template <typename Operation> void rejects(Operation&& operation)
{
    bool rejected = false;
    try { operation(); } catch (std::exception const&) { rejected = true; }
    require(rejected, "worker accepted an invalid or over-budget operation");
}

void scoped_threads()
{
    auto const original = worker_detail::cpu_threads();
    {
        worker_detail::CpuLimit limit(1);
        require(worker_detail::cpu_threads() == 1, "worker CPU scope ignored configured bound");
        std::atomic<unsigned> observed{0};
        { WorkerReferenceThread child([&] { observed.store(worker_detail::cpu_threads()); }); }
        require(observed.load() == 1, "reference child thread did not inherit CPU scope");
        { worker_detail::CpuLimit nested(2); require(worker_detail::cpu_threads() == 1, "nested scope widened CPU allowance"); }
    }
    require(worker_detail::cpu_threads() == original, "worker CPU scope leaked into caller");
}

WorkerConfig config()
{
    WorkerConfig options; options.capacity.cpu_threads = 2; options.capacity.slots = 2;
    options.capacity.host_bytes = uint64_t(2) << 30; options.generation_threads = 1;
    options.tls.max_frame_bytes = 4096;
    return options;
}
std::vector<uint64_t> unpack(Member const& member)
{
    std::vector<uint64_t> values; values.reserve(member.size()); auto const view = member.view();
    for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(view.k()); ++chunk)
        for (auto value : view.region(chunk)) values.push_back(value);
    return values;
}
unsigned process_threads()
{
#ifdef _WIN32
    HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    require(snapshot != INVALID_HANDLE_VALUE, "cannot inspect worker CPU fanout");
    struct CloseSnapshot { HANDLE value; ~CloseSnapshot() { ::CloseHandle(value); } } cleanup{snapshot};
    THREADENTRY32 entry{}; entry.dwSize = sizeof(entry); unsigned count = 0;
    for (BOOL found = ::Thread32First(snapshot, &entry); found; found = ::Thread32Next(snapshot, &entry))
        if (entry.th32OwnerProcessID == ::GetCurrentProcessId()) ++count;
    return count;
#else
    struct CloseDirectory { void operator()(DIR* value) const { if (value) ::closedir(value); } };
    std::unique_ptr<DIR, CloseDirectory> directory(::opendir("/proc/self/task"));
    require(bool(directory), "cannot inspect worker CPU fanout"); unsigned count = 0;
    while (auto entry = ::readdir(directory.get())) if (entry->d_name[0] != '.') ++count;
    return count;
#endif
}
void generation()
{
    auto options = config(); WorkerService service(options);
    std::array<uint8_t, 32> id{}; std::iota(id.begin(), id.end(), uint8_t(0));
    worker_detail::CpuLimit limit(1);
    auto expected = worker_original_reference(18, 2, id);
    auto collect = [&]
    {
        std::unique_ptr<CompactMemberReceiver> receiver;
        service.generate_local(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
        {
            if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count));
            require(tile.words.size_bytes() <= options.tls.max_frame_bytes, "worker tile exceeded transport window");
            receiver->append(tile);
        });
        require(bool(receiver), "worker delivered no compact output"); return unpack(receiver->finish());
    };
    // The original reference has already instantiated its inline templates in
    // this executable. Observe the isolated worker while both remain linked.
    std::atomic<bool> sampling{true}, ready{false}, measure{false}; std::atomic<unsigned> peak{0};
    std::thread sampler([&]
    {
        ready.store(true);
        while (!measure.load()) std::this_thread::yield();
        while (sampling.load())
        {
            auto const count = process_threads(); auto previous = peak.load();
            while (previous < count && !peak.compare_exchange_weak(previous, count)) {}
            std::this_thread::yield();
        }
    });
    while (!ready.load()) std::this_thread::yield(); auto const baseline = process_threads();
    peak.store(baseline); measure.store(true);
    std::vector<uint64_t> actual;
    try { actual = collect(); }
    catch (...) { sampling.store(false); sampler.join(); throw; }
    sampling.store(false); sampler.join();
    require(actual == expected, "worker CPU compact stream differs from original reference TU");
    require(peak.load() <= baseline + 1, "linked original reference defeated worker CPU fanout limit");
    rejects([&] { service.generate_local(18, 2, id, [](auto const&, uint64_t) { throw std::runtime_error("owner stopped reading"); }); });
    require(collect() == expected, "worker could not be reused after a failed consumer");
    uint64_t next_chunk = 0;
    ChunkCodec codec(18, uint64_t(1) << 20);
    std::array<std::array<uint8_t, 32>, 2> ids{id, id};
    size_t first = 0;
    service.contributions_local(18, 2, ids, [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
    {
        require(ordinal == next_chunk++ && members == ids.size(), "worker contribution order or batch membership changed");
        size_t last = first;
        uint64_t const end = (ordinal + 1) << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        while (last < expected.size() && expected[last] < end) ++last;
        std::span<uint64_t const> region(expected.data() + first, last - first);
        std::array<std::span<uint64_t const>, 2> regions{region, region};
        auto const canonical = codec.encode(ordinal, {}, 0, regions);
        require(std::ranges::equal(bytes, canonical), "worker coencoded contribution differs from canonical codec");
        first = last;
    });
    require(next_chunk == PlotGroupFile::getChunkCountForK(18) && first == expected.size(), "worker contribution stream was truncated");
    uint64_t maximum_chunk = 0, chunk_count = 0, previous_chunk = UINT64_MAX;
    for (auto value : expected)
    {
        auto const chunk = value >> (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        if (chunk != previous_chunk) { maximum_chunk = std::max(maximum_chunk, chunk_count); chunk_count = 0; previous_chunk = chunk; }
        ++chunk_count;
    }
    maximum_chunk = std::max(maximum_chunk, chunk_count);
    auto bounded = options;
    bounded.max_task_bytes = std::max<uint64_t>(bounded.tls.max_frame_bytes, ChunkCodec::estimate_work(18, 0, maximum_chunk));
    WorkerService scratch_limited(bounded);
    bool emitted = false, scratch_rejected = false;
    try { scratch_limited.contributions_local(18, 2, ids, [&](auto, auto, auto) { emitted = true; }); }
    catch (pool::UnsupportedResources const&) { scratch_rejected = true; }
    require(scratch_rejected && !emitted, "over-limit contribution scratch emitted partial output before rejection");
    next_chunk = 0;
    scratch_limited.contributions_local(18, 2, {&id, 1}, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == next_chunk++ && members == 1 && !bytes.empty(), "split contribution did not fit its admitted scratch"); });
    require(next_chunk == PlotGroupFile::getChunkCountForK(18), "valid split contribution was truncated after scratch rejection");
    std::atomic<bool> stopped{true};
    rejects([&] { service.generate_local(18, 2, id, [](auto const&, uint64_t) {}, {&stopped}); });
    auto too_small = options; too_small.capacity.host_bytes = uint64_t(64) << 20;
    WorkerService rejected(too_small);
    rejects([&] { rejected.generate_local(18, 2, id, [](auto const&, uint64_t) {}); });
    service.stop(); rejects([&] { service.generate_local(18, 2, id, [](auto const&, uint64_t) {}); });
}

struct RunningBroker
{
    pool::BrokerServer server;
    std::atomic<bool> stop{false};
    std::thread thread;
    explicit RunningBroker(pool::BrokerOptions options)
        : server(std::move(options)), thread([this] { server.serve(stop); }) {}
    ~RunningBroker() { stop.store(true); thread.join(); }
};

void multiple_brokers(char const* ca, char const* cert, char const* key)
{
    pool::TlsOptions tls{ca, cert, key, 10000, 65536};
    pool::BrokerOptions broker_options; broker_options.listen = {"127.0.0.1", 0}; broker_options.tls = tls;
    RunningBroker first(broker_options), second(broker_options);
    auto options = config(); options.broker = first.server.endpoint();
    auto other = second.server.endpoint(); other.server_name = "localhost";
    options.brokers = {other}; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    auto first_alias = options.broker; first_alias.server_name = "localhost";
    auto second_alias = other; second_alias.server_name.clear();
    options.brokers.push_back(first_alias); options.brokers.push_back(second_alias);
    options.broker_refresh_parallelism = 1;
    options.capacity.cpu_threads = options.capacity.slots = 1;
    WorkerService service(options); service.start();
    auto const registration_deadline = Clock::now() + std::chrono::seconds(3);
    while ((service.status().known_brokers != 2 || service.status().healthy_brokers != 2) && Clock::now() < registration_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    require(service.status().known_brokers == 2 && service.status().healthy_brokers == 2,
        "alternate routes consumed independent broker registry entries");
    WorkerClientConfig client; client.broker = options.broker; client.tls = tls;
    client.owner_id = "shared-worker"; client.local_service = &service; client.task_timeout_ms = 10000;
    WorkerClient a(client); client.broker = other; WorkerClient b(client);
    std::array<uint8_t, 32> id{}; id.fill(0x47);
    worker_detail::CpuLimit limit(1);
    auto expected = worker_original_reference(18, 2, id);
    auto collect = [&](WorkerClient& owner, std::function<void()> hold = {})
    {
        std::unique_ptr<CompactMemberReceiver> receiver;
        owner.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t total)
        {
            if (!receiver)
            {
                receiver = std::make_unique<CompactMemberReceiver>(18, size_t(total));
                if (hold) hold();
            }
            receiver->append(tile);
        });
        return unpack(receiver->finish());
    };
    std::promise<void> entered, release;
    auto ready = entered.get_future(), finish = release.get_future();
    auto held = std::async(std::launch::async, [&]
    { return collect(a, [&] { entered.set_value(); finish.wait(); }); });
    bool const started = ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (!started) { release.set_value(); held.get(); throw std::runtime_error("first broker made no progress"); }
    auto queued = std::async(std::launch::async, [&] { return collect(b); });
    bool const bounded = queued.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    release.set_value();
    require(held.get() == expected && queued.get() == expected, "independent brokers changed worker output");
    require(bounded, "independent brokers exceeded the worker's one shared task slot");

    client.local_service = nullptr; WorkerClient remote(client);
    auto local = std::async(std::launch::async, [&] { return collect(a); });
    auto network = std::async(std::launch::async, [&] { return collect(remote); });
    require(local.get() == expected && network.get() == expected, "local and remote multi-broker admissions disagreed");
}

void live_controls(pool::TlsOptions tls, pool::TlsOptions const& other)
{
    tls.timeout_ms = 3000;
    pool::BrokerOptions broker_options; broker_options.listen = {"127.0.0.1", 0}; broker_options.tls = tls;
    RunningBroker broker(broker_options);
    auto options = config(); options.broker = broker.server.endpoint(); options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.capacity.cpu_threads = options.capacity.slots = 1;
    WorkerService service(options); service.start();
    auto endpoint = service.endpoint();
    require(WorkerService::remote_status(endpoint, other).admission == pool::WorkerAdmission::Accepting,
        "authenticated member could not read worker status");
    rejects([&] { WorkerService::remote_control(endpoint, other, WorkerControl::Pause); });
    pool::Writer payload(tls.max_frame_bytes); payload.u32(1); payload.u64(0); payload.u16(0); payload.bytes({});
    payload.u16(0); payload.u16(1); payload.u32(3); payload.u64(1); payload.u64(2); payload.u64(3);
    pool::BrokerClient owner(options.broker, tls); auto session = owner.open_owner("control-inflight-codec");
    pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility(); task.kind = pool::Kind::Pack;
    task.k = 18; task.strength = 2; task.work_units = payload.data().size(); task.chunk_count = 1; task.max_output_bytes = 4096;
    task.resources = {1, 1, add(multiply(payload.data().size(), 20), ChunkCodec::estimate_work(18, 0, 0)), 0};
    auto lease = owner.acquire(session, task); require(bool(lease), "control fixture did not acquire codec work");
    auto channel = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
    pool::Writer request(tls.max_frame_bytes); pool::write_lease(request, *lease); request.u32(10000);
    request.u8(uint8_t(Operation::Encode)); request.u8(18); request.u8(2); request.u64(payload.data().size()); request.u64(4096);
    channel.send(pool::Message::Work, request.data());
    auto until = Clock::now() + std::chrono::seconds(3); WorkerStatus busy;
    do { busy = WorkerService::remote_status(endpoint, tls); if (busy.accepted_tasks == 1 && busy.active.slots == 1) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); } while (Clock::now() < until);
    require(busy.accepted_tasks == 1 && busy.active.slots == 1 && busy.active.host_bytes,
        "live worker status blocked behind its occupied execution slot");
    auto paused = WorkerService::remote_control(endpoint, tls, WorkerControl::Pause);
    require(paused.admission == pool::WorkerAdmission::Paused && paused.accepted_tasks == 1 && paused.active.host_bytes == busy.active.host_bytes,
        "pause discarded accepted work or released its memory accounting");
    std::array<uint8_t, 32> id{};
    rejects([&] { service.generate_local(18, 2, id, [](auto const&, auto) {}); });
    auto draining = WorkerService::remote_control(endpoint, tls, WorkerControl::Drain);
    require(draining.admission == pool::WorkerAdmission::Draining && draining.accepted_tasks == 1,
        "drain failed to retain the accepted codec request");
    channel.send(pool::Message::Data, payload.data()); bool ended = false;
    while (!ended)
    {
        auto frame = channel.receive(); require(frame.type != pool::Message::Error, "drain interrupted accepted work");
        ended = frame.type == pool::Message::End;
    }
    until = Clock::now() + std::chrono::seconds(3);
    do { busy = WorkerService::remote_status(endpoint, tls); if (busy.admission == pool::WorkerAdmission::Drained) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); } while (Clock::now() < until);
    require(busy.admission == pool::WorkerAdmission::Drained && !busy.accepted_tasks && !busy.active.slots && !busy.active.host_bytes,
        "drained worker retained task resources or accepted state");
    require(WorkerService::remote_control(endpoint, tls, WorkerControl::Resume).admission == pool::WorkerAdmission::Accepting,
        "resume did not reopen worker admission");
}

void broker_authority(pool::TlsOptions tls, pool::TlsOptions const& other)
{
    tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker real(options);
    pool::BrokerClient probe(real.server.endpoint(), other);
    pool::WorkerOffer offered; offered.instance = "authority-probe"; offered.compatibility = worker_compatibility();
    offered.endpoint = {"127.0.0.1", 9999}; offered.capacity = {1, 1, uint64_t(1) << 20, 0};
    offered.capabilities = pool::capability(pool::Kind::Pack);
    auto const epoch = probe.register_worker(offered); probe.unregister_worker(offered.instance);
    // A different authenticated pool member deliberately asserts the real
    // broker's epoch. An epoch is a fence, not an authenticated identity.
    struct Impersonator
    {
        pool::Listener listener;
        std::atomic<bool> stop{false};
        std::atomic<unsigned> begin_requests{0};
        std::mutex mutex;
        pool::Lease forged;
        std::vector<std::thread> handlers;
        std::thread thread;
        Impersonator(pool::TlsOptions const& tls, std::string epoch)
            : listener({"127.0.0.1", 0}, tls), thread([this, epoch]
        {
            while (!stop.load()) try
            {
                auto channel = listener.accept(50); if (!channel) continue;
                require(handlers.size() < 4, "impersonator connection fixture exceeded its bound");
                handlers.emplace_back([this, epoch, channel = std::move(channel)]() mutable
                {
                    while (!stop.load()) try
                    {
                        auto frame = channel.receive(); pool::Reader request(frame.payload);
                        require(frame.type == pool::Message::Control && request.u32() == pool::protocol_version, "invalid impersonator request");
                        auto const operation = request.u8(); pool::Writer response(65536);
                        if (operation == 5) response.string(epoch); // worker registration
                        else if (operation == 12) response.u8(1); // heartbeat
                        else if (operation == 6) { ++begin_requests; response.u8(0); }
                        else if (operation == 1)
                        { auto const owner = request.string(256); response.string(owner); response.string("forged-session"); response.string(epoch); response.u64(1); }
                        else if (operation == 2)
                        { std::lock_guard lock(mutex); response.u8(1); pool::write_lease(response, forged); }
                        channel.send(pool::Message::Reply, response.data());
                    }
                    catch (...) { break; }
                });
            }
            catch (...) { if (stop.load()) break; }
        }) {}
        ~Impersonator() { stop.store(true); thread.join(); for (auto& handler : handlers) handler.join(); }
    } impostor(other, epoch);
    auto worker = config(); worker.broker = real.server.endpoint(); worker.brokers = {impostor.listener.endpoint()};
    worker.listen = {"127.0.0.1", 0}; worker.tls = tls; worker.capacity.cpu_threads = worker.capacity.slots = 1;
    WorkerService service(worker); service.start();
    require(service.status().known_brokers == 2 && service.status().healthy_brokers == 2,
        "distinct authenticated broker identities were merged by an asserted epoch");
    WorkerClientConfig owner; owner.broker = worker.broker; owner.tls = tls; owner.owner_id = "bound-authority"; owner.local_service = &service;
    WorkerClient client(owner); std::array<uint8_t, 32> id{}; id.fill(0x62);
    std::unique_ptr<CompactMemberReceiver> receiver;
    client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t total)
    { if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(total)); receiver->append(tile); });
    require(receiver && unpack(receiver->finish()) == worker_original_reference(18, 2, id) && !impostor.begin_requests.load(),
        "lease authority routed real work to a different authenticated broker");
    pool::Task task; task.task_id = "forged-authority"; task.compatibility = worker_compatibility(); task.resources = {1, 1, 1024, 0};
    {
        std::lock_guard lock(impostor.mutex); impostor.forged.task = task; impostor.forged.epoch = epoch;
        impostor.forged.token = pool::random_token(); impostor.forged.broker_identity = pool::certificate_identity(tls.cert_file);
        impostor.forged.worker_identity = impostor.forged.owner_identity = pool::certificate_identity(tls.cert_file);
        impostor.forged.worker_instance = service.status().instance; impostor.forged.endpoint = service.endpoint(); impostor.forged.ttl_ms = 10000;
    }
    pool::BrokerClient liar(impostor.listener.endpoint(), tls); auto session = liar.open_owner("forged-authority-owner");
    bool bound = false;
    try { (void)liar.acquire(session, task); }
    catch (std::runtime_error const& error) { bound = std::string_view(error.what()).find("lease authority differs") != std::string_view::npos; }
    require(bound, "broker client accepted a lease authority asserted by a different TLS peer");
}

void broker_rotation(pool::TlsOptions tls)
{
    tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.lease_ms = 10000;
    RunningBroker first(options), second(options);
    std::vector<pool::Endpoint> addresses{first.server.endpoint(), second.server.endpoint()};
    auto worker = config(); worker.tls = tls; worker.listen = {"127.0.0.1", 0}; worker.max_brokers = 1;
    worker.broker_rotation_ms = 1000; worker.discover_brokers = [addresses] { return addresses; };
    WorkerService service(worker); service.start();
    pool::BrokerClient a(addresses[0], tls), b(addresses[1], tls);
    auto session = a.open_owner("pinned-rotation-grant");
    pool::Task task; task.task_id = "pinned-rotation-task"; task.compatibility = worker_compatibility(); task.resources = {1, 1, 1024, 0};
    auto lease = a.acquire(session, task); require(bool(lease), "initial bounded authority was not registered");
    // Leave the grant unbegun: no local task/client copy can pin it. Only the
    // broker's atomic reservation check can preserve this admitted work.
    std::this_thread::sleep_for(std::chrono::milliseconds(1150));
    require(a.status().workers == 1 && b.status().workers == 0 && service.status().broker_saturated,
        "overflow rotation revoked an unused granted lease");
    a.cancel(session, task.task_id, task.attempt);
    auto until = Clock::now() + std::chrono::seconds(3); bool reached = false;
    do
    {
        reached = b.status().workers == 1;
        auto const status = service.status();
        require(status.known_brokers <= worker.max_brokers && status.broker_transport_bytes == broker_registration_bytes(),
            "overflow rotation exceeded its charged control transport budget");
        if (reached) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (Clock::now() < until);
    require(reached, "healthy admitted authorities permanently starved discovered overflow");
}

void many_brokers(pool::TlsOptions tls)
{
    tls.timeout_ms = 3000;
    pool::BrokerOptions broker_options; broker_options.listen = {"127.0.0.1", 0}; broker_options.tls = tls;
    broker_options.worker_ms = 10000; broker_options.lease_ms = 10000;
    std::vector<std::unique_ptr<RunningBroker>> authorities;
    std::vector<pool::Endpoint> addresses;
    pool::Listener stalled({"127.0.0.1", 0}, tls);
    for (unsigned index = 0; index < 4; ++index)
    { auto route = stalled.endpoint(); route.server_name = "stalled-" + std::to_string(index); addresses.push_back(std::move(route)); }
    for (unsigned index = 0; index < 24; ++index)
    { auto broker = std::make_unique<RunningBroker>(broker_options); addresses.push_back(broker->server.endpoint()); authorities.push_back(std::move(broker)); }
    auto options = config(); options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.max_brokers = 32; options.broker_refresh_parallelism = 4; options.broker_refresh_ms = 100;
    options.discover_brokers = [addresses] { return addresses; };
    WorkerService service(options); service.start();
    auto until = Clock::now() + std::chrono::seconds(10);
    while (service.status().healthy_brokers != authorities.size() && Clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto status = service.status(); require(status.healthy_brokers == authorities.size() && status.known_brokers == addresses.size(),
        "failed discovered brokers starved authorities beyond the former broker limit");
    require(status.broker_transport_bytes == multiply(options.max_brokers, broker_registration_bytes()) &&
        !status.broker_saturated && !status.deferred_brokers, "worker did not reserve bounded broker transport capacity");
    std::vector<std::unique_ptr<WorkerClient>> owners;
    for (auto const& broker : authorities)
    {
        WorkerClientConfig client; client.broker = broker->server.endpoint(); client.tls = tls; client.local_service = &service;
        client.owner_id = "many-broker-owner"; client.task_timeout_ms = 10000; owners.push_back(std::make_unique<WorkerClient>(client));
    }
    std::array<uint8_t, 32> id{}; id.fill(0x35); worker_detail::CpuLimit limit(1);
    auto const expected = worker_original_reference(18, 2, id);
    std::promise<void> unblock; auto release = unblock.get_future().share();
    std::vector<std::future<std::vector<uint64_t>>> results;
    for (auto const& owner : owners) results.push_back(std::async(std::launch::async, [&, selected = owner.get()]
    {
        std::unique_ptr<CompactMemberReceiver> receiver;
        selected->generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t total)
        {
            if (!receiver) { receiver = std::make_unique<CompactMemberReceiver>(18, size_t(total)); release.wait(); }
            receiver->append(tile);
        });
        return unpack(receiver->finish());
    }));
    until = Clock::now() + std::chrono::seconds(5);
    do { status = service.status(); if (status.accepted_tasks == owners.size()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); } while (Clock::now() < until);
    bool const shared_gate = status.accepted_tasks == owners.size() && status.active.slots <= options.capacity.slots &&
        status.active.cpu_threads <= options.capacity.cpu_threads && status.active.host_bytes <= status.capacity.host_bytes;
    unblock.set_value();
    for (auto& result : results) require(result.get() == expected, "concurrent broker task changed reference output");
    require(shared_gate, "many independent brokers bypassed the worker's single global resource gate");
    // The configured transport ceiling remains a real admission budget.
    service.stop(); options.max_brokers = 8; WorkerService limited(options); limited.start();
    auto limited_status = limited.status(); require(limited_status.known_brokers == options.max_brokers && limited_status.broker_saturated &&
        limited_status.deferred_brokers, "discovered broker budget saturation was hidden");
}

void network(char const* ca, char const* cert, char const* key)
{
    pool::TlsOptions tls{ca, cert, key, 10000, 65536};
    pool::BrokerOptions broker_options; broker_options.listen = {"127.0.0.1", 0}; broker_options.tls = tls;
    broker_options.lease_ms = 10000; broker_options.worker_ms = 10000;
    RunningBroker broker(broker_options);
    auto options = config(); options.broker = broker.server.endpoint(); options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.capacity.cpu_threads = 1;
    options.capacity.slots = 4;
    options.cpu_scheduler = std::make_shared<CodecScheduler>(4);
    WorkerService service(options); service.start();
    WorkerClientConfig client_options; client_options.broker = options.broker; client_options.tls = tls; client_options.owner_id = "worker-loopback";
    client_options.task_timeout_ms = 10000; client_options.cpu_scheduler = options.cpu_scheduler;
    WorkerClient client(client_options);
    std::array<uint8_t, 32> id{}; id.fill(0x71);
    worker_detail::CpuLimit limit(1);
    auto expected = worker_original_reference(18, 2, id);
    auto collect = [&]
    {
        std::unique_ptr<CompactMemberReceiver> receiver;
        client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t total)
        { if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(total)); receiver->append(tile); });
        return unpack(receiver->finish());
    };
    require(collect() == expected, "leased TLS worker compact stream differs from reference");
    auto a = std::async(std::launch::async, collect), b = std::async(std::launch::async, collect);
    require(a.get() == expected && b.get() == expected, "concurrent owner tasks lost deterministic output");
    auto collect_local = [&]
    {
        std::unique_ptr<CompactMemberReceiver> receiver;
        service.generate_local(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t total)
        { if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(total)); receiver->append(tile); });
        return unpack(receiver->finish());
    };
    auto local_a = std::async(std::launch::async, collect_local);
    auto local_b = std::async(std::launch::async, collect_local);
    auto remote = std::async(std::launch::async, collect);
    require(local_a.get() == expected && local_b.get() == expected && remote.get() == expected,
        "embedded worker lost deterministic output with concurrent local and remote calls");

    // One persistent session carries a bounded window containing multiple
    // chunks, mixing encoded old/contributions with new raw regions.
    ChunkCodec codec(18, uint64_t(1) << 20);
    std::vector<std::vector<uint64_t>> raw(4);
    std::vector<std::vector<uint8_t>> old(4), canonical(4);
    std::vector<std::array<WorkerContribution, 1>> contributions(4);
    std::vector<std::array<std::span<uint64_t const>, 1>> added(4);
    std::vector<WorkerChunkInput> chunks; chunks.reserve(4);
    uint64_t const range = uint64_t(1) << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t i = 0; i < raw.size(); ++i)
    {
        for (uint64_t n = 0; n < 64; ++n) raw[i].push_back(uint64_t(i) * range + n * (range / 64) + 1);
        added[i][0] = raw[i]; auto bytes = codec.encode(i, {}, 0, added[i]); old[i].assign(bytes.begin(), bytes.end());
        contributions[i][0] = {old[i], 1};
        std::array<EncodedChunk, 2> pair{{{old[i], 1}, {old[i], 1}}};
        auto full = codec.join(i, pair, added[i]); canonical[i].assign(full.begin(), full.end());
        chunks.push_back({uint64_t(i), old[i], 1, contributions[i], added[i]});
    }
    auto encoded = client.encode(18, chunks, uint64_t(1) << 20);
    require(encoded == canonical, "TLS multi-chunk codec window differs from local canonical bytes");
    rejects([&] { (void)client.encode(18, chunks, 1); });
    require(client.encode(18, chunks, uint64_t(1) << 20) == canonical, "worker codec failed reuse after owner output-bound rejection");
    uint64_t next_chunk = 0; size_t first = 0;
    client.contributions(18, 2, {&id, 1}, [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
    {
        require(ordinal == next_chunk++ && members == 1, "TLS contribution order changed");
        size_t last = first; while (last < expected.size() && expected[last] < (ordinal + 1) * range) ++last;
        std::span<uint64_t const> values(expected.data() + first, last - first);
        require(std::ranges::equal(bytes, codec.encode(ordinal, {}, 0, {&values, 1})), "TLS initial contribution differs from local codec");
        first = last;
    });
    require(next_chunk == PlotGroupFile::getChunkCountForK(18), "TLS contribution stream was truncated");

    // A stalled consumer retains memory but relinquishes application compute.
    pool::BrokerClient controller(options.broker, tls); auto owner = controller.open_owner("stalled-worker-test");
    pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility(); task.kind = pool::Kind::GenerateCpu;
    task.k = 22; task.strength = 2;
    task.resources = {1, 1, worker_cpu_generation_bytes(22, 2) + boundary_bytes(22) + options.max_task_bytes, 0};
    std::optional<pool::Lease> lease;
    while (!(lease = controller.acquire(owner, task))) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto blocked = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
    // This fixture tests a blocked output consumer after k22 computation.
    // Leave room for instrumented reference generation; short absolute task
    // deadlines are checked separately by grow-worker-client-test.
    pool::Writer request(tls.max_frame_bytes); pool::write_lease(request, *lease); request.u32(120000);
    request.u8(uint8_t(Operation::Packed)); request.u8(22); request.u8(2); request.u16(1); request.bytes(id);
    blocked.send(pool::Message::Work, request.data());
    auto prelude = blocked.receive();
    while (prelude.type == pool::Message::Reply)
    { require(prelude.payload.empty(), "stalled task emitted an invalid keepalive"); prelude = blocked.receive(); }
    require(prelude.type == pool::Message::Data, "stalled task did not produce output");
    // A replay reaches a different handler while the original lease is still
    // live. Rejecting it must not finish or release the original handler's job.
    {
        auto duplicate = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
        duplicate.send(pool::Message::Work, request.data());
        auto rejection = duplicate.receive();
        require(rejection.type == pool::Message::Error, "duplicate running lease was not rejected");
        rejects([&] { (void)duplicate.receive(); }); // Wait for that handler's cleanup.
        controller.renew(*lease);
    }
    auto progress = std::async(std::launch::async, collect_local);
    require(progress.wait_for(std::chrono::seconds(2)) == std::future_status::ready && progress.get() == expected,
        "blocked worker output retained a CPU permit");
    blocked.close();
    service.stop();
}
void budget()
{
    Budget budget(100, 2, 10);
    WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::milliseconds(100);
    rejects([&] { auto impossible = budget.acquire(91, true, cancel); });
    { auto a = budget.acquire(60, true, cancel); rejects([&] { a.resize(91); }); auto b = budget.acquire(10, true, cancel); a.transfer(b, 20); }
    auto all = budget.acquire(90, true, cancel);
}
void fitted_broker_budget()
{
    auto options = config(); options.capacity.host_bytes = uint64_t(256) << 20;
    options.discover_brokers = [] { return std::vector<pool::Endpoint>{}; };
    auto explicit_limit = options;
    rejects([&] { WorkerService rejected(explicit_limit); });
    fit_worker_broker_budget(options);
    require(options.max_brokers && options.max_brokers < explicit_limit.max_brokers,
        "automatic broker fitting did not reduce an infeasible explicit ceiling");
    WorkerService service(options); auto status = service.status();
    require(status.capacity.host_bytes >= add(add(worker_cpu_generation_bytes(18, 2), boundary_bytes(18)), options.max_task_bytes),
        "automatic broker transport fitting consumed supported execution memory");
}

void retained_budget()
{
    Budget budget(100, 4, 10);
    WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::seconds(3);
    auto retained = budget.acquire(0, false, cancel, true);
    auto current = budget.acquire(60, true, cancel);
    std::promise<void> submitted; auto ready = submitted.get_future();
    auto waiting = std::async(std::launch::async, [&]
    {
        submitted.set_value();
        try { auto request = budget.acquire(50, true, cancel); }
        catch (pool::UnsupportedResources const&) { return true; }
        return false;
    });
    ready.wait(); require(waiting.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
        "model request did not wait for transient memory");
    current.transfer(retained, 50);
    require(budget.task_capacity() == 40 && waiting.wait_for(std::chrono::seconds(1)) == std::future_status::ready && waiting.get(),
        "persistent retention did not wake and reject an impossible request");

    std::promise<void> resizing; auto queued = resizing.get_future();
    auto after_resize = std::async(std::launch::async, [&]
    {
        resizing.set_value();
        try { auto request = budget.acquire(35, true, cancel); }
        catch (pool::UnsupportedResources const&) { return true; }
        return false;
    });
    queued.wait(); require(after_resize.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
        "model resize request did not wait for transient memory");
    retained.resize(60);
    require(budget.task_capacity() == 30 && after_resize.wait_for(std::chrono::seconds(1)) == std::future_status::ready && after_resize.get(),
        "retained resize did not atomically update the impossible-request bound");
    current.release(); retained.resize(20);
    { auto request = budget.acquire(70, true, cancel); }
    retained.release(); require(budget.task_capacity() == 90, "released retention remained in the worker baseline");
    auto full = budget.acquire(90, true, cancel);
}

void chunk_headers()
{
    auto rejects_header = [](pool::Writer const& encoded, char const* expected, uint32_t leased = 0)
    {
        pool::Reader input(encoded.data()); bool rejected = false;
        try { (void)read_chunks(input, pool::default_max_codec_chunks, pool::default_codec_task_bytes, leased); }
        catch (std::runtime_error const& error) { rejected = error.what() == std::string(expected); }
        require(rejected, "worker did not reject metadata before reserving its declared arrays");
    };
    pool::Writer chunks; chunks.u32(pool::default_max_codec_chunks);
    rejects_header(chunks, "truncated worker chunk headers");
    pool::Writer contributions; contributions.u32(1); contributions.u64(0); contributions.u16(0);
    contributions.bytes({}); contributions.u16(UINT16_MAX); contributions.u16(0);
    rejects_header(contributions, "truncated worker contribution headers");
    pool::Writer members; members.u32(1); members.u64(0); members.u16(0); members.bytes({});
    members.u16(0); members.u16(UINT16_MAX);
    rejects_header(members, "truncated worker member headers");
    rejects_header(members, "worker chunk count differs from lease", 2);
}

void renewal_errors()
{
    auto const cancelled = std::make_exception_ptr(TaskCancelled("worker task cancelled"));
    auto const transport = std::make_exception_ptr(pool::TransientTransport("broker connection reset"));
    auto const stale = std::make_exception_ptr(pool::StaleLease("broker epoch changed"));
    auto const permanent = std::make_exception_ptr(std::runtime_error("invalid content or TLS identity"));
    auto kind = [](std::exception_ptr error)
    { try { std::rethrow_exception(error); } catch (std::exception const& value) { return error_kind(value); } };
    require(kind(task_failure(cancelled, transport)) == WorkerError::Transport &&
        kind(task_failure(cancelled, stale)) == WorkerError::Stale &&
        kind(task_failure(cancelled, permanent)) == WorkerError::Permanent &&
        kind(task_failure(permanent, transport)) == WorkerError::Permanent,
        "worker renewal cancellation lost its cause or masked a permanent failure");
    for (auto expected : {WorkerError::Permanent, WorkerError::Unsupported, WorkerError::Transport, WorkerError::Stale})
    {
        pool::Writer output; output.u8(uint8_t(expected)); output.string("bounded worker error");
        try { throw_worker_error(output.data()); }
        catch (std::exception const& error)
        { require(error_kind(error) == expected, "worker error frame changed its retry classification"); }
    }
}

void offered_cpus()
{
    CodecScheduler shared(6); CpuAdmission offered(2);
    WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::seconds(3);
    std::promise<void> entered, release; auto held = entered.get_future(); auto finish = release.get_future().share();
    auto first = std::async(std::launch::async, [&]
    { CpuWork work(offered, &shared, cancel, 2); entered.set_value(); finish.wait(); });
    held.wait();
    std::promise<void> submitted; auto started = submitted.get_future();
    auto waiting = std::async(std::launch::async, [&]
    { submitted.set_value(); CpuWork work(offered, &shared, cancel); });
    started.wait(); bool const bounded = waiting.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    auto owner = std::async(std::launch::async, [&] { CodecScheduler::Lease work(&shared, 4); });
    bool const owner_progress = owner.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    release.set_value(); first.get(); waiting.get(); owner.get();
    require(bounded, "worker used the wider owner scheduler to exceed its offered CPUs");
    require(owner_progress, "queued worker work consumed CPU permits reserved for its owner");

    std::promise<void> single_entered, single_release;
    auto one_entered = single_entered.get_future(); auto one_release = single_release.get_future().share();
    auto one = std::async(std::launch::async, [&]
    { CpuWork work(offered, &shared, cancel); single_entered.set_value(); one_release.wait(); });
    one_entered.wait();
    auto other = std::async(std::launch::async, [&] { CpuWork work(offered, &shared, cancel); });
    bool const concurrent = other.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    single_release.set_value(); one.get(); other.get();
    require(concurrent, "worker CPU admission serialized independently fitting compute scopes");

    std::atomic<bool> stopped{false}; WorkerCancellation stop{&stopped, cancel.deadline};
    std::promise<void> holding, unblock; auto occupied = holding.get_future(); auto unblocked = unblock.get_future().share();
    auto block = std::async(std::launch::async, [&]
    { CpuWork work(offered, &shared, cancel, 2); holding.set_value(); unblocked.wait(); });
    occupied.wait();
    auto cancelled = std::async(std::launch::async, [&]
    { try { CpuWork work(offered, &shared, stop); } catch (std::runtime_error const&) { return true; } return false; });
    stopped.store(true);
    bool const cancelled_waiter = cancelled.wait_for(std::chrono::seconds(1)) == std::future_status::ready && cancelled.get();
    unblock.set_value(); block.get();
    require(cancelled_waiter, "worker CPU queue ignored cancellation");
    CpuWork reused(offered, &shared, cancel, 2);
}
int main(int argc, char** argv)
{
    try { scoped_threads(); budget(); fitted_broker_budget(); retained_budget(); chunk_headers(); renewal_errors(); offered_cpus(); generation(); if (argc == 4) network(argv[1], argv[2], argv[3]);
        else if (argc == 2 && std::string_view(argv[1]) == "--network")
        { pool_test::Directory directory; auto ca = pool_test::issue(directory.path, "worker-ca");
          auto identity = pool_test::issue(directory.path, "worker-host", &ca);
          network(pool_test::utf8_path(ca.cert_file).c_str(), pool_test::utf8_path(identity.cert_file).c_str(), pool_test::utf8_path(identity.key_file).c_str());
          multiple_brokers(pool_test::utf8_path(ca.cert_file).c_str(), pool_test::utf8_path(identity.cert_file).c_str(), pool_test::utf8_path(identity.key_file).c_str());
          auto other = pool_test::issue(directory.path, "other-worker-host", &ca);
          live_controls(pool_test::tls(ca, identity), pool_test::tls(ca, other));
          broker_authority(pool_test::tls(ca, identity), pool_test::tls(ca, other));
          broker_rotation(pool_test::tls(ca, identity));
          many_brokers(pool_test::tls(ca, identity)); }
        else require(argc == 1, "worker test expects CA/cert/key or no arguments");
        std::cout << "worker CPU parity, bounds, cancellation and reuse passed\n"; return 0; }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
