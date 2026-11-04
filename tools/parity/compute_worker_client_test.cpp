// As in grow-worker-test, include the implementation to exercise bounded
// private ownership without adding a production cache-testing interface.
#include "host/ComputeWorker.cpp"
#include "host/ComputeSession.hpp"
#include "host/ComputeIdentity.hpp"
#include "compute_pool_fixture.hpp"
#include <iostream>

using namespace group_grow;
using group_tools::require;

namespace
{
template <typename Operation> void rejects(Operation&& operation)
{
    bool rejected = false;
    try { operation(); } catch (std::exception const&) { rejected = true; }
    require(rejected, "client accepted an expired task or invalid retry identity");
}

void retry_pressure()
{
    worker_detail::TaskRegistry registry;
    WorkerTaskIdentity identity("long-running-member");
    auto connection = registry.acquire(identity, "owner-session");
    uint64_t const sequence = connection->sequence;
    ++connection->next_attempt;
    for (size_t i = 0; i <= worker_detail::TaskRegistry::maximum; ++i)
    {
        WorkerTaskIdentity completed("completed-" + std::to_string(i));
        (void)registry.acquire(completed, "owner-session");
    }
    // The old bug evicted this state after connection teardown because its
    // submission timestamp predated all the intervening completed work.
    connection.reset();
    WorkerTaskIdentity following("following");
    (void)registry.acquire(following, "owner-session");
    auto retry = registry.acquire(identity, "owner-session");
    require(retry->sequence == sequence && retry->next_attempt == 2,
        "cache pressure discarded the logical member's retry sequence or attempt");
    auto copied = identity;
    require(registry.acquire(copied, "owner-session") == retry,
        "copying a retry handle detached its pinned task state");
    worker_detail::TaskRegistry another;
    rejects([&] { (void)another.acquire(identity, "owner-session"); });
    copied.task_id = "different-member";
    rejects([&] { (void)registry.acquire(copied, "owner-session"); });
    registry.reset();
    auto restarted = registry.acquire(identity, "new-owner-session");
    require(restarted != retry && restarted->sequence == 1 && restarted->next_attempt == 1,
        "a new owner session retained an obsolete retry sequence");

    worker_detail::TaskRegistry bounded;
    std::vector<WorkerTaskIdentity> active;
    active.reserve(worker_detail::TaskRegistry::maximum);
    for (size_t i = 0; i < worker_detail::TaskRegistry::maximum; ++i)
    {
        active.emplace_back("active-" + std::to_string(i));
        (void)bounded.acquire(active.back(), "bounded-session");
    }
    WorkerTaskIdentity overflow("overflow");
    rejects([&] { (void)bounded.acquire(overflow, "bounded-session"); });
}

void planning_feedback()
{
    using Planning = worker_detail::GenerationPlanning;
    auto const now = Clock::now(); pool::Endpoint route{"127.0.0.1", 49101};
    WorkerTaskObservation fast; fast.worker_identity = "local-worker"; fast.worker_instance = "first";
    fast.kind = pool::Kind::GenerateCpu; fast.k = 18; fast.strength = 2; fast.work_units = 1;
    fast.local_direct = true; fast.execution_ns = 400; fast.transfer_ns = 100;
    auto hint = [&](Planning& planning, uint8_t k = 18, uint8_t strength = 2, pool::Kind kind = pool::Kind::GenerateCpu)
    { return planning.suggested(k, strength, kind, now); };
    auto observe = [&](Planning& planning, WorkerTaskObservation const& value, pool::Endpoint const& endpoint)
    { auto const selected = planning.selected(value, endpoint, now); planning.observed(value, endpoint, selected, now); };
    auto train = [&](Planning& planning, WorkerTaskObservation const& value, pool::Endpoint const& endpoint)
    { for (unsigned i = 0; i < 3; ++i) observe(planning, value, endpoint); };
    Planning planning(1000);
    auto missing = fast; missing.transfer_ns = 0; train(planning, missing, route);
    require(hint(planning).batch == 1 && hint(planning).concurrency == 2,
        "missing delivery phase evidence trained a generation policy");
    observe(planning, fast, route); observe(planning, fast, route);
    require(hint(planning).batch == 1, "one noisy completion changed generation batching");
    observe(planning, fast, route);
    require(hint(planning).batch == 2 && hint(planning).concurrency == 3,
        "consistent generation observations did not train bounded hints");
    auto pressure = fast; pressure.execution_ns = 100; pressure.transfer_ns = 400;
    for (unsigned i = 0; i < 6; ++i) observe(planning, i & 1 ? fast : pressure, route);
    require(hint(planning).batch == 2 && hint(planning).concurrency == 3,
        "alternating result pressure bypassed planning hysteresis");
    require(hint(planning, 20).batch == 1 && hint(planning, 18, 3).batch == 1 &&
        hint(planning, 18, 2, pool::Kind::GenerateGpu).batch == 1,
        "generation hints crossed a plot parameter or backend boundary");
    auto slow = pressure; slow.worker_identity = "remote-worker"; slow.local_direct = false;
    train(planning, slow, route);
    require(hint(planning).batch == 2 && hint(planning).concurrency == 3,
        "a slow remote minority pinned the local worker's learned policy");
    auto gpu = fast; gpu.worker_identity = "gpu-worker"; gpu.kind = pool::Kind::GenerateGpu;
    train(planning, gpu, route);
    planning.failed(fast, route, planning.selected(fast, route, now), now);
    require(hint(planning).batch == 1 && hint(planning, 18, 2, pool::Kind::GenerateGpu).batch == 2,
        "a CPU worker failure changed another backend's policy");
    auto hybrid = gpu; hybrid.kind = pool::Kind::GenerateCpu; train(planning, hybrid, route);
    planning.failed(hybrid, route, planning.selected(hybrid, route, now), now);
    require(hint(planning, 18, 2, pool::Kind::GenerateGpu).batch == 2,
        "one backend's failure changed the same worker's other backend policy");
    require(hint(planning, 18, 2, pool::Kind::GenerateAny).batch == 1,
        "mixed-backend aggregation ignored its active worker majority");

    Planning replacement(1000); train(replacement, fast, route);
    auto changed = fast; changed.worker_instance = "replacement";
    observe(replacement, changed, route);
    require(hint(replacement).batch == 1, "a replacement worker inherited an old incarnation's feedback");
    observe(replacement, changed, route); observe(replacement, changed, route);
    require(hint(replacement).batch == 2, "replacement worker could not learn its own feedback");
    auto relocated = route; ++relocated.port; observe(replacement, changed, relocated);
    require(hint(replacement).batch == 1, "a changed authenticated route inherited old feedback");
    changed.local_direct = false; observe(replacement, changed, relocated);
    require(hint(replacement).batch == 1, "a TLS route inherited local delivery votes");
    require(replacement.suggested(18, 2, pool::Kind::GenerateCpu, now + std::chrono::milliseconds(1001)).concurrency == 2,
        "an idle worker retained stale pressure beyond the task timeout");

    Planning bounded(1000);
    for (unsigned i = 0; i < 256; ++i) observe(bounded, fast, route);
    require(hint(bounded).batch == pool::max_generation_batch_members && hint(bounded).concurrency == 64,
        "generation feedback exceeded or failed to reach its bounded policy");
    auto older_workload = fast; older_workload.k = 20; train(bounded, older_workload, route);
    auto retired_selection = bounded.selected(older_workload, route, now);
    for (size_t i = 0; i < Planning::maximum; ++i)
    {
        auto fresh = fast; fresh.worker_identity = "fresh-" + std::to_string(i);
        observe(bounded, fresh, route);
    }
    bounded.observed(older_workload, route, retired_selection, now);
    bounded.failed(older_workload, route, retired_selection, now);
    require(hint(bounded, 20).batch == 1, "late feedback resurrected a worker after bounded history eviction");
    Planning invalid(1000); auto oversized = fast; oversized.worker_identity.assign(65, 'x');
    train(invalid, oversized, route); auto invalid_route = route; invalid_route.host.assign(254, 'x');
    train(invalid, fast, invalid_route);
    require(hint(invalid).batch == 1, "invalid protocol keys entered fixed planning storage");

    Planning late(1000); auto old_selection = late.selected(fast, route, now);
    train(late, fast, route); changed = fast; changed.worker_instance = "new-process";
    train(late, changed, route);
    for (unsigned i = 0; i < 3; ++i) late.observed(pressure, route, old_selection, now);
    late.failed(fast, route, old_selection, now);
    require(hint(late).batch == 2 && hint(late).concurrency == 3,
        "an old incarnation's late completion or failure replaced current feedback");
    auto before_relocation = late.selected(changed, route, now);
    train(late, changed, relocated); train(late, changed, route);
    late.observed(changed, route, before_relocation, now); late.failed(changed, route, before_relocation, now);
    require(hint(late).batch == 2 && hint(late).concurrency == 3,
        "returning to an earlier route resurrected a fenced selection's feedback");
    auto expired_selection = late.selected(changed, route, now);
    auto const expired = now + std::chrono::milliseconds(1001);
    late.observed(changed, route, expired_selection, expired); late.failed(changed, route, expired_selection, expired);
    require(late.suggested(18, 2, pool::Kind::GenerateCpu, expired).batch == 1,
        "a delayed completion revived a peer after its monotonic observation timeout");
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

struct StalledPeer
{
    pool::Listener listener;
    std::mutex mutex;
    std::condition_variable ready;
    bool release = false;
    std::atomic<bool> accepted{false};
    std::exception_ptr failure;
    std::thread thread;
    explicit StalledPeer(pool::TlsOptions tls)
        : listener({"127.0.0.1", 0}, tls), thread([this]
        {
            try
            {
                for (;;)
                {
                    auto channel = listener.accept();
                    if (!channel) { std::lock_guard lock(mutex); if (release) return; continue; }
                    (void)channel.receive(); accepted.store(true); ready.notify_all();
                    std::unique_lock lock(mutex); ready.wait(lock, [&] { return release; }); return;
                }
            }
            catch (...) { failure = std::current_exception(); ready.notify_all(); }
        }) {}
    void unblock()
    { { std::lock_guard lock(mutex); release = true; } ready.notify_all(); }
    ~StalledPeer() { unblock(); thread.join(); }
};

WorkerCancellation deadline()
{
    WorkerCancellation result;
    result.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
    return result;
}
template <typename Operation> void bounded_failure(Operation&& operation)
{
    auto const begin = std::chrono::steady_clock::now();
    rejects(std::forward<Operation>(operation));
    require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1),
        "client waited for the longer socket timeout after its task deadline");
}

void network_deadlines()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "client-ca");
    auto identity = pool_test::issue(directory.path, "client-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::validate_tls_credentials(tls);
    auto other_key = pool_test::issue(directory.path, "admission-wrong-key", &ca);
    auto mismatched_key = tls; mismatched_key.key_file = pool_test::utf8_path(other_key.key_file);
    rejects([&] { pool::validate_tls_credentials(mismatched_key); });
    {
        StalledPeer peer(tls);
        WorkerClientConfig config; config.broker = peer.listener.endpoint(); config.tls = tls;
        config.owner_id = "client-cpu-wait-deadline";
        config.cpu_scheduler = std::make_shared<CodecScheduler>(1);
        WorkerClient client(config);
        std::array<uint64_t, 3> fragments{1, 2, 3};
        std::span<uint64_t const> region(fragments);
        WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
        std::future<void> pending;
        bool completed;
        {
            CodecScheduler::Lease pressure(config.cpu_scheduler.get());
            auto cancel = deadline();
            pending = std::async(std::launch::async, [&, cancel]
            { bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); }); });
            completed = pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
        }
        pending.get();
        require(completed && !peer.accepted.load(), "client CPU admission ignored the task deadline");
    }
    {
        StalledPeer peer(tls);
        WorkerClientConfig config; config.broker = peer.listener.endpoint(); config.tls = tls;
        config.owner_id = "lazy-open-deadline";
        WorkerClient client(config);
        require(!peer.accepted.load(), "client construction opened an owner session before a task");
        std::array<uint8_t, 32> id{};
        auto cancel = deadline();
        bounded_failure([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, cancel); });
        require(peer.accepted.load(), "lazy-owner deadline test never reached authenticated control");
    }
    {
        StalledPeer peer(tls);
        WorkerClientConfig config; config.broker = peer.listener.endpoint(); config.tls = tls;
        config.owner_id = "sequence-wait-deadline";
        WorkerClient client(config); std::array<uint8_t, 32> id{};
        auto first = std::async(std::launch::async, [&]
        { rejects([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}); }); });
        {
            std::unique_lock lock(peer.mutex);
            require(peer.ready.wait_for(lock, std::chrono::seconds(1), [&] { return peer.accepted.load(); }),
                "first owner-open request did not reach the stalled broker");
        }
        auto cancel = deadline();
        bounded_failure([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, cancel); });
        peer.unblock(); first.get();
    }
    {
        pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
        RunningBroker broker(options);
        StalledPeer peer(tls);
        pool::WorkerOffer offer; offer.instance = pool::random_token(); offer.compatibility = worker_compatibility();
        offer.endpoint = peer.listener.endpoint(); offer.capacity = {1, 1, uint64_t(128) << 20, 0};
        offer.capabilities = pool::capability(pool::Kind::Pack);
        pool::BrokerClient control(broker.server.endpoint(), tls); control.register_worker(offer);
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "direct-result-deadline";
        WorkerClient client(config);
        std::array<uint64_t, 3> fragments{1, 2, 3};
        std::span<uint64_t const> region(fragments);
        WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
        auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        require(peer.accepted.load(), "direct deadline test never reached the leased TLS peer");
    }
}

void assigned_local_and_fencing()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "local-ca");
    auto identity = pool_test::issue(directory.path, "local-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    auto data_tls = tls; data_tls.max_frame_bytes = 4096;
    WorkerConfig offer; offer.broker = broker.server.endpoint(); offer.listen = {"127.0.0.1", 0}; offer.tls = data_tls;
    offer.capacity = {2, 4, uint64_t(2) << 30, 0}; offer.generation_threads = 1;
    WorkerService service(offer); service.start();
    WorkerClientConfig config; config.broker = offer.broker; config.tls = data_tls;
    config.owner_id = "direct-local-owner"; config.local_service = &service;
    std::vector<WorkerTaskObservation> observations;
    config.observe = [&](auto const& value) { observations.push_back(value); };
    WorkerClient client(config);
    std::array<uint8_t, 32> id{};
    std::vector<std::array<uint8_t, 32>> oversized(pool::max_generation_batch_members + 1);
    rejects([&] { client.contributions(18, 2, oversized, [](auto, auto, auto) {}); });
    std::unique_ptr<CompactMemberReceiver> receiver;
    uint64_t sink_ns = 0;
    client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
    {
        if (!receiver)
        {
            auto const begin = Clock::now(); std::this_thread::sleep_for(std::chrono::milliseconds(10));
            sink_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
            receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count));
        }
        receiver->append(tile);
    });
    require(bool(receiver), "assigned local generation produced no member");
    auto generated = receiver->finish();
    require(generated.size(), "assigned local generation produced an empty member");
    auto stats = client.stats();
    require(stats.local_tasks == 1 && !stats.input_bytes && !stats.output_bytes &&
        observations.size() == 1 && observations.front().local_direct && observations.front().execution_ns &&
        sink_ns && observations.front().transfer_ns >= sink_ns,
        "ordinary local assignment used wire payload or omitted its observation");
    WorkerTaskObservation remote_observation;
    auto remote_config = config; remote_config.local_service = nullptr; remote_config.owner_id = "remote-observation-owner";
    remote_config.observe = [&](auto const& value) { remote_observation = value; };
    WorkerClient remote(remote_config); sink_ns = 0;
    remote.generate(18, 2, id, [&](auto const&, uint64_t)
    {
        if (sink_ns) return;
        auto const begin = Clock::now(); std::this_thread::sleep_for(std::chrono::milliseconds(10));
        sink_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
    });
    require(!remote_observation.local_direct && remote_observation.execution_ns &&
        sink_ns && remote_observation.transfer_ns >= sink_ns,
        "TLS result consumption used a different sink observation boundary");

    std::array<uint64_t, 3> fragments{1, 2, 3}; std::span<uint64_t const> region(fragments);
    WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
    ChunkCodec codec(18, uint64_t(1) << 20);
    auto canonical = codec.encode(0, {}, 0, {&region, 1});
    std::vector<uint8_t> expected(canonical.begin(), canonical.end());
    WorkerTaskOptions per_call;
    bool selected = false, observed = false;
    per_call.selected = [&](auto const& value) { selected = value.local_direct; return true; };
    per_call.observe = [&](auto const& value) { observed = value.local_direct && !value.worker_instance.empty(); };
    auto encoded = client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, {}, {}, per_call);
    require(encoded == std::vector<std::vector<uint8_t>>{expected} && selected && observed,
        "direct local codec or per-call worker metadata differs from canonical output");
    stats = client.stats();
    require(stats.local_tasks == 2 && !stats.input_bytes && !stats.output_bytes,
        "direct local codec counted a wire transfer");
    per_call.selected = [](auto const&) { return false; };
    bool declined = false;
    try { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, {}, {}, per_call); }
    catch (WorkerTaskDeclined const&) { declined = true; }
    require(declined && client.stats().failed_tasks == stats.failed_tasks && client.stats().local_tasks == stats.local_tasks,
        "an optional selection decline counted as a worker execution or failure");

    // Fence between successful receipt/validation and its final acknowledgement.
    // A best-effort transport exception must not swallow an explicit rejection.
    pool::BrokerClient controller(offer.broker, tls);
    bool stale = false;
    try
    {
        (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, {}, [&](auto result)
        {
            require(result.front() == expected, "fencing fixture received incorrect local bytes");
            (void)controller.open_owner(config.owner_id);
        });
    }
    catch (pool::StaleDelivery const&) { stale = true; }
    require(stale && client.stats().local_tasks == stats.local_tasks,
        "explicit fenced completion was accepted as a successful local result");
    require(client.encode(18, {&chunk, 1}, uint64_t(1) << 20).front() == expected,
        "client did not refresh its fenced owner session for later work");

    // Changing public generation parameters creates a distinct bounded model.
    uint64_t chunks = 0;
    std::array<std::array<uint8_t, 32>, 2> ids{id, id};
    client.contributions(18, 3, ids, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == chunks++ && !bytes.empty() && members == ids.size(), "changed-workload batch was truncated"); });
    require(chunks == PlotGroupFile::getChunkCountForK(18) && client.suggested_batch_members(18, 3, 2) <= 2 &&
        client.suggested_generation_concurrency(18, 3, 1) == 1,
        "adaptive worker suggestions exceeded the caller's storage or RAM ceiling");
    config.local_service = nullptr; config.owner_id = "minimum-frame-wire-owner";
    WorkerClient wire(config); chunks = 0;
    wire.contributions(18, 3, ids, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == chunks++ && !bytes.empty() && members == ids.size(), "minimum-frame contribution batch was truncated"); });
    require(chunks == PlotGroupFile::getChunkCountForK(18) && wire.stats().output_bytes,
        "minimum data frame prevented the worker's larger control profile registration");
    std::vector<std::array<uint64_t, 3>> full_raw(pool::default_max_codec_chunks);
    std::vector<std::span<uint64_t const>> full_regions(full_raw.size());
    std::vector<WorkerChunkInput> full_window(full_raw.size());
    auto const range = uint64_t(1) << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t i = 0; i < full_raw.size(); ++i)
    { full_raw[i] = {i * range + 1, i * range + 2, i * range + 3}; full_regions[i] = full_raw[i];
      full_window[i] = {i, {}, 0, {}, {&full_regions[i], 1}}; }
    auto full_encoded = wire.encode(18, full_window, uint64_t(1) << 20, {}, {}, [&](auto output)
    {
        require(output.size() == full_window.size(), "full bounded codec window was truncated");
        for (size_t i = 0; i < output.size(); ++i)
            require(std::ranges::equal(output[i], codec.encode(i, {}, 0, {&full_regions[i], 1})),
                "full bounded codec window changed canonical bytes");
    });
    require(full_encoded.size() == pool::default_max_codec_chunks,
        "worker did not honor its advertised full codec chunk limit");
    service.stop();
    uint64_t maximum_chunk = 0; auto const view = generated.view();
    for (uint64_t ordinal = 0; ordinal < PlotGroupFile::getChunkCountForK(18); ++ordinal)
        maximum_chunk = std::max<uint64_t>(maximum_chunk, view.region(ordinal).size());
    auto bounded = offer;
    bounded.max_task_bytes = std::max<uint64_t>(bounded.tls.max_frame_bytes, ChunkCodec::estimate_work(18, 0, maximum_chunk));
    RunningBroker scratch_broker(options); bounded.broker = scratch_broker.server.endpoint();
    WorkerService scratch_limited(bounded); scratch_limited.start();
    config.broker = bounded.broker; config.owner_id = "batch-refusal-owner";
    config.local_service = nullptr; config.task_timeout_ms = 10000;
    WorkerClient refused(config);
    bool unsupported = false, emitted = false;
    try { refused.contributions(18, 2, ids, [&](auto, auto, auto) { emitted = true; }); }
    catch (pool::UnsupportedResources const&) { unsupported = true; }
    require(unsupported && !emitted,
        "refused contribution batch retried its excluded shape instead of returning before output");
    chunks = 0;
    refused.contributions(18, 2, {&id, 1}, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == chunks++ && members == 1 && !bytes.empty(), "singleton split failed after a typed batch refusal"); });
    require(chunks == PlotGroupFile::getChunkCountForK(18), "singleton split was truncated after a typed batch refusal");
    scratch_limited.stop();
}

void expired_cleanup()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "cleanup-ca");
    auto identity = pool_test::issue(directory.path, "cleanup-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    auto options = [&]
    { pool::BrokerOptions value; value.listen = {"127.0.0.1", 0}; value.tls = tls; value.lease_ms = 10000; return value; };
    auto offer = [&]
    {
        pool::WorkerOffer value; value.instance = pool::random_token(); value.compatibility = worker_compatibility();
        value.endpoint = {"127.0.0.1", 1}; value.capacity = {1, 1, uint64_t(128) << 20, 0};
        value.capabilities = pool::capability(pool::Kind::Pack); return value;
    };
    auto following = [&](pool::BrokerClient& controller, uint64_t bytes)
    {
        auto owner = controller.open_owner("following-cleanup-owner");
        pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility();
        task.kind = pool::Kind::Pack; task.k = 18; task.strength = 2; task.resources = {1, 1, bytes, 0};
        task.chunk_count = 1; task.max_output_bytes = 1;
        task.work_units = pool::codec_window_header_bytes + pool::codec_chunk_header_bytes;
        std::optional<pool::Lease> lease;
        auto const end = Clock::now() + std::chrono::milliseconds(500);
        do
        {
            lease = controller.acquire(owner, task);
            if (!lease) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        while (!lease && Clock::now() < end);
        require(bool(lease), "expired compute task retained capacity until its lease timeout");
        controller.cancel(owner, task.task_id, task.attempt);
    };
    std::array<uint64_t, 3> fragments{1, 2, 3}; std::span<uint64_t const> region(fragments);
    WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
    {
        RunningBroker broker(options());
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "queued-cleanup-owner"; WorkerClient client(config);
        auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        // Registration would immediately grant a forgotten queued task before
        // the following owner can use this newly available capacity.
        pool::BrokerClient controller(config.broker, tls); auto worker = offer(); controller.register_worker(worker);
        following(controller, worker.capacity.host_bytes);
    }
    {
        RunningBroker broker(options());
        pool::BrokerClient controller(broker.server.endpoint(), tls); auto worker = offer(); controller.register_worker(worker);
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "declined-cleanup-owner"; WorkerClient client(config);
        auto cancel = deadline(); bool selected = false;
        WorkerTaskOptions selection;
        selection.selected = [&](auto const&)
        { selected = true; std::this_thread::sleep_until(cancel.deadline); return false; };
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel, {}, {}, selection); });
        require(selected, "declined cleanup fixture never received a grant");
        following(controller, worker.capacity.host_bytes);
    }
    {
        RunningBroker broker(options()); StalledPeer peer(tls);
        pool::BrokerClient controller(broker.server.endpoint(), tls); auto worker = offer();
        worker.endpoint = peer.listener.endpoint(); controller.register_worker(worker);
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "abandoned-cleanup-owner"; WorkerClient client(config);
        auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        require(peer.accepted.load(), "abandoned cleanup fixture never contacted its worker");
        following(controller, worker.capacity.host_bytes);
    }
    {
        RunningBroker broker(options());
        WorkerConfig worker; worker.broker = broker.server.endpoint(); worker.listen = {"127.0.0.1", 0}; worker.tls = tls;
        worker.capacity = {1, 1, uint64_t(2) << 30, 0}; worker.cpu_scheduler = std::make_shared<CodecScheduler>(1);
        WorkerService service(worker); service.start();
        WorkerClientConfig config; config.broker = worker.broker; config.tls = tls;
        config.owner_id = "running-cleanup-owner"; WorkerClient client(config);
        {
            CodecScheduler::Lease pressure(worker.cpu_scheduler.get());
            auto cancel = deadline();
            bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        }
        pool::BrokerClient controller(worker.broker, tls); following(controller, 1);
        config.owner_id = "running-direct-cleanup-owner"; config.local_service = &service;
        WorkerClient direct(config);
        std::future<void> pending; bool completed;
        {
            CodecScheduler::Lease pressure(worker.cpu_scheduler.get()); auto cancel = deadline();
            pending = std::async(std::launch::async, [&, cancel]
            { bounded_failure([&] { (void)direct.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); }); });
            completed = pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
        }
        pending.get();
        require(completed, "expired direct task did not leave its shared CPU gate");
        following(controller, 1);
        service.stop();
    }
}

void generation_admission()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "admission-ca");
    auto identity = pool_test::issue(directory.path, "admission-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options); StalledPeer peer(tls);
    pool::WorkerOffer offer; offer.instance = pool::random_token(); offer.compatibility = worker_compatibility();
    offer.endpoint = peer.listener.endpoint(); offer.capacity = {2, 2, uint64_t(2) << 30, 0};
    offer.capabilities = pool::capability(pool::Kind::GenerateCpu);
    pool::BrokerClient controller(broker.server.endpoint(), tls); controller.register_worker(offer);
    WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
    config.owner_id = "generation-admission-owner"; config.max_active_generation = 1;
    config.cpu_scheduler = std::make_shared<CodecScheduler>(1);
    WorkerConfig local; local.capacity = {1, 1, uint64_t(2) << 30, 0}; local.cpu_scheduler = config.cpu_scheduler;
    WorkerService service(local); config.local_service = &service;
    WorkerClient client(config); std::array<uint8_t, 32> id{};
    WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::seconds(2);
    auto first = std::async(std::launch::async, [&, cancel]
    { rejects([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, cancel); }); });
    {
        std::unique_lock lock(peer.mutex);
        require(peer.ready.wait_for(lock, std::chrono::seconds(1), [&] { return peer.accepted.load(); }),
            "generation admission fixture did not reach its first worker");
    }
    auto waiting = deadline();
    bounded_failure([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, waiting); });
    auto fallback = deadline();
    bounded_failure([&] { client.generate_local(18, 2, id, [](auto const&, uint64_t) {}, fallback); });
    require(client.stats().peak_active_generation == 1,
        "concurrent generation requests bypassed their shared in-flight ceiling");
    {
        auto cpu_deadline = deadline();
        CodecScheduler::Lease cpu(config.cpu_scheduler.get(), 1, [cpu_deadline] { cpu_deadline.check(); });
    }
    first.get();
    std::unique_ptr<CompactMemberReceiver> receiver;
    client.generate_local(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
    { if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count)); receiver->append(tile); });
    require(receiver && receiver->finish().size() && !client.stats().local_tasks && !client.stats().completed_generation,
        "fallback did not release/reuse generation admission or changed broker completion counts");
}

void protected_admission()
{
    pool_test::Directory directory; auto ca = pool_test::issue(directory.path, "admission-ca");
    auto identity = pool_test::issue(directory.path, "admission-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls; RunningBroker broker(options);
    auto data_tls = tls; data_tls.max_frame_bytes = 4096;
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0}; offered.tls = data_tls;
    offered.capacity = {1, 2, uint64_t(2) << 30, 0}; offered.generation_threads = 1;
    bool startup_checked = false;
    offered.admit_start = [&](WorkerConfig const& actual)
    { startup_checked = true; require(actual.capacity == offered.capacity && worker_service_baseline_bytes(actual) < actual.capacity.host_bytes,
        "service startup was not checked against its finalized complete partition"); };
    WorkerService service(offered); require(startup_checked, "worker executor started without its embedding admission"); service.start();
    WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = data_tls; config.owner_id = "protected-owner";
    unsigned selected = 0; bool fallback = false;
    config.admit_task = [&](WorkerTaskAdmission const& value)
    {
        ++selected; fallback = value.local_fallback;
        require(value.worker.identity == pool::certificate_identity(tls.cert_file) && value.worker.instance == service.status().instance &&
            value.task.resources.cpu_threads && value.task.resources.host_bytes && value.task.kind == pool::Kind::GenerateCpu,
            "task admission did not receive authenticated status and a concrete grant");
        throw WorkerAdmissionRejected("fixture refuses this compute grant");
    };
    auto rejected = [&](auto operation)
    { bool typed = false; try { operation(); } catch (WorkerAdmissionRejected const&) { typed = true; } require(typed, "policy rejection lost its permanent type"); };
    std::array<uint8_t, 32> id{};
    WorkerClient remote(config);
    rejected([&] { remote.generate(18, 2, id, [](auto const&, auto) {}); });
    require(selected == 1 && !fallback && !remote.stats().input_bytes && !remote.stats().output_bytes && !service.status().accepted_tasks,
        "rejected remote grant submitted Work or data to its executor");
    config.local_service = &service; WorkerClient local(config);
    rejected([&] { local.generate(18, 2, id, [](auto const&, auto) {}); });
    require(selected == 2 && !fallback && !local.stats().local_tasks && !service.status().accepted_tasks,
        "assigned local work bypassed pre-execution admission");
    rejected([&] { local.generate_local(18, 2, id, [](auto const&, auto) {}); });
    require(selected == 3 && fallback && !service.status().accepted_tasks && !service.status().active.host_bytes,
        "outage fallback allocated or executed before policy admission");
    auto denied = offered; denied.backend = "gpu";
    denied.admit_start = [](WorkerConfig const&) { throw WorkerAdmissionRejected("reject before selecting GPU executor"); };
    rejected([&] { WorkerService unreviewed(denied); });
    service.stop();
    auto unregistered_config = offered; unregistered_config.broker = {};
    WorkerService unregistered(unregistered_config); unregistered.start();
    pool::WorkerOffer stale; stale.instance = pool::random_token(); stale.compatibility = worker_compatibility();
    stale.endpoint = unregistered.endpoint(); stale.capacity = unregistered.status().capacity;
    stale.capabilities = pool::capability(pool::Kind::GenerateCpu); stale.generation_threads = 1;
    pool::BrokerClient controller(broker.server.endpoint(), tls); controller.register_worker(stale);
    auto changed = config; changed.local_service = nullptr; changed.owner_id = "stale-instance-owner";
    bool callback = false; changed.admit_task = [&](auto const&) { callback = true; };
    WorkerClient stale_client(changed);
    rejected([&] { stale_client.generate(18, 2, id, [](auto const&, auto) {}); });
    require(!callback && !stale_client.stats().input_bytes && !unregistered.status().accepted_tasks,
        "lease/status instance mismatch reached policy or executor Work");
    controller.unregister_worker(stale.instance); unregistered.stop();
}

void session_memory_ownership()
{
    pool_test::Directory directory;
    auto identity = pool::create_pool(directory.path / "session", "owned-startup-fixture");
    auto defaults = identity.defaults; defaults.discovery = false; defaults.interfaces = {"lo"}; defaults.max_brokers = 16;
    pool::configure_pool(identity, defaults);
    pos2gpu::GroupOptions args; args.pool_directory = pool_test::utf8_path(identity.directory);
    args.max_ram = uint64_t(2) << 30; args.workers = 1; args.backend = "cpu"; args.backend_set = true;
    args.pool_offer_ram = uint64_t(512) << 20; args.pool_offer_cpus = 1; args.pool_offer_backend_set = true;
    args.pool_frame_bytes = args.window = 65536; args.pool_socket_timeout_ms = 500;
    args.protected_resources = true; args.compute_admit_session = [] {};
    pos2gpu::prepare_group_compute(args);
    pos2gpu::ComputeSession session(args);
    require(!session.owned_worker_bytes(), "a session credited worker memory before acquiring its identity");
    auto offer = pos2gpu::make_group_compute_offer(args);
    offer.admit_start = [&](WorkerConfig const& actual)
    {
        require(session.owned_worker_bytes() == actual.capacity.host_bytes &&
            session.owned_worker_bytes() > worker_service_baseline_bytes(actual),
            "identity-reserved startup omitted its total/baseline ownership");
        throw WorkerAdmissionRejected("startup fixture refuses construction");
    };
    rejects([&] { session.offer_local(offer); });
    require(!session.owned_worker_bytes() && !session.local_worker(), "failed worker startup retained a resident-memory grant");
    offer.admit_start = [&](WorkerConfig const& actual)
    { require(session.owned_worker_bytes() == actual.capacity.host_bytes, "worker startup lost its owned partition"); };
    session.offer_local(offer);
    {
        auto local = session.local_worker(); require(bool(local), "automatic worker did not reach its ready lifetime");
        auto const status = local->status();
        require(session.owned_worker_bytes() == status.capacity.host_bytes + status.baseline_host_bytes,
            "ready worker double counted or omitted its service baseline");
    }
    auto foreign_args = args;
    pos2gpu::ComputeSession foreign(foreign_args);
    foreign.offer_local(pos2gpu::make_group_compute_offer(foreign_args));
    require(!foreign.local_worker() && !foreign.owned_worker_bytes(),
        "an already-running worker identity granted its foreign pages to another session");
}

void codec_window_limits()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "codec-limits-ca");
    auto identity = pool_test::issue(directory.path, "codec-limits-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    auto data_tls = tls; data_tls.max_frame_bytes = 4096;
    WorkerConfig worker; worker.broker = broker.server.endpoint(); worker.listen = {"127.0.0.1", 0}; worker.tls = data_tls;
    worker.capacity = {1, 1, uint64_t(128) << 20, 0}; worker.max_window_chunks = 2; worker.max_task_bytes = 4096;
    WorkerService service(worker); service.start();
    bool selected = false;
    WorkerClientConfig config; config.broker = worker.broker; config.tls = data_tls; config.owner_id = "codec-limits-owner";
    config.max_window_chunks = 3; config.max_task_bytes = 8192;
    config.selected = [&](auto const&) { selected = true; };
    WorkerClient client(config);
    require(client.max_window_chunks() == config.max_window_chunks && client.max_task_bytes() == config.max_task_bytes,
        "client did not expose its configured codec limits");
    std::array<std::array<uint64_t, 3>, 3> raw;
    std::array<std::span<uint64_t const>, 3> regions;
    std::array<WorkerChunkInput, 3> chunks;
    auto const range = uint64_t(1) << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t i = 0; i < raw.size(); ++i)
    { raw[i] = {i * range + 1, i * range + 2, i * range + 3}; regions[i] = raw[i];
      chunks[i] = {i, {}, 0, {}, {&regions[i], 1}}; }
    auto refused_before_data = [&](std::span<WorkerChunkInput const> input, uint64_t output)
    {
        selected = false; auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, input, output, cancel); });
        auto const stats = client.stats();
        require(!selected && !stats.input_bytes && !stats.output_bytes,
            "broker granted a codec window exceeding the worker's advertised limits");
    };
    refused_before_data(chunks, worker.max_task_bytes);
    std::vector<uint64_t> larger(worker.max_task_bytes / sizeof(uint64_t), 1);
    std::span<uint64_t const> large_region(larger);
    WorkerChunkInput large{0, {}, 0, {}, {&large_region, 1}};
    refused_before_data({&large, 1}, worker.max_task_bytes);
    refused_before_data({chunks.data(), 1}, config.max_task_bytes);
    auto encoded = client.encode(18, {chunks.data(), worker.max_window_chunks}, worker.max_task_bytes);
    require(selected && encoded.size() == worker.max_window_chunks,
        "a worker's smaller but compatible codec window was not usable after refusal");
    config.owner_id = "smaller-client-codec-owner"; config.max_window_chunks = 1; config.max_task_bytes = worker.max_task_bytes;
    WorkerClient smaller(config); selected = false;
    rejects([&] { (void)smaller.encode(18, {chunks.data(), worker.max_window_chunks}, worker.max_task_bytes); });
    rejects([&] { (void)smaller.encode(18, {&large, 1}, worker.max_task_bytes); });
    require(!selected && !smaller.stats().input_bytes, "client sent a window exceeding its own configured codec limit");
    service.stop();
}

void renewal_fence()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "renewal-ca");
    auto identity = pool_test::issue(directory.path, "renewal-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    WorkerConfig worker; worker.broker = broker.server.endpoint(); worker.listen = {"127.0.0.1", 0}; worker.tls = tls;
    worker.tls.timeout_ms = 1500; worker.capacity = {1, 1, uint64_t(512) << 20, 0};
    worker.cpu_scheduler = std::make_shared<CodecScheduler>(1);
    WorkerService service(worker); service.start();
    pool::BrokerClient controller(worker.broker, tls);
    auto owner = controller.open_owner("renewal-fence-owner");
    pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility();
    task.kind = pool::Kind::GenerateCpu; task.k = 18; task.strength = 2;
    task.resources = {1, 1, worker_cpu_generation_bytes(18, 2) + boundary_bytes(18) + worker.max_task_bytes, 0};
    auto lease = controller.acquire(owner, task); require(bool(lease), "renewal fixture did not obtain its generation lease");
    {
        // Hold the actual shared compute gate so renewal loss is observed at
        // cancellable admission, independently of reference generation speed.
        CodecScheduler::Lease pressure(worker.cpu_scheduler.get());
        auto channel = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
        pool::Writer request(tls.max_frame_bytes); pool::write_lease(request, *lease); request.u32(10000);
        request.u8(uint8_t(Operation::Packed)); request.u8(18); request.u8(2); request.u16(1);
        std::array<uint8_t, 32> id{}; request.bytes(id); channel.send(pool::Message::Work, request.data());
        bool running = false; auto const until = Clock::now() + std::chrono::seconds(3);
        do
        {
            try { controller.renew(*lease); running = true; }
            catch (pool::StaleLease const&) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        } while (!running && Clock::now() < until);
        require(running, "renewal fixture never entered Running state");
        owner = controller.open_owner(owner.owner_id);
        auto frame = channel.receive();
        require(frame.type == pool::Message::Error, "fenced renewal did not return a typed worker failure");
        bool stale = false;
        try { throw_worker_error(frame.payload); } catch (pool::StaleLease const&) { stale = true; }
        require(stale, "worker renewal collapsed stale lease into a permanent cancellation");
    }
    task.task_id = pool::random_token();
    std::optional<pool::Lease> following; auto const until = Clock::now() + std::chrono::seconds(1);
    do
    {
        following = controller.acquire(owner, task);
        if (!following) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (!following && Clock::now() < until);
    require(bool(following), "fenced renewal left compute resources charged after worker completion");
    controller.cancel(owner, task.task_id, task.attempt); service.stop();
}
} // namespace

int main() try
{
    retry_pressure(); planning_feedback(); network_deadlines(); assigned_local_and_fencing(); expired_cleanup(); generation_admission(); protected_admission(); session_memory_ownership(); codec_window_limits(); renewal_fence();
    std::cout << "client retry ownership, bounded cache and absolute task deadlines passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
