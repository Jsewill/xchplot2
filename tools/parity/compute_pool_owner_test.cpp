// Like grow-merge-test, exercise the actual controller producer in this TU.
#include "host/GroupPlotter.cpp"
#include "host/GroupDashboardFeed.hpp"
#include "compute_pool_fixture.hpp"
#include "host/ComputeThreads.hpp"
#ifdef _WIN32
#include <process.h>
#endif

using namespace group_grow;
using group_tools::require;
namespace fs = std::filesystem;
std::vector<uint64_t> worker_original_reference(uint8_t, uint8_t, std::array<uint8_t, 32> const&);

namespace
{
struct RunningBroker
{
    pool::BrokerServer server;
    std::atomic<bool> stop{false};
    std::thread thread;
    explicit RunningBroker(pool::BrokerOptions options)
        : server(std::move(options)), thread([this] { server.serve(stop); }) {}
    ~RunningBroker() { stop.store(true); thread.join(); }
};
struct Directory
{
    fs::path path;
    Directory()
    {
        // External input needs real disk scratch. CTest runs in its build
        // directory, matching the existing construction fixtures.
#ifdef _WIN32
        for (unsigned attempt = 0; attempt < 32; ++attempt)
        {
            path = fs::current_path() / ("grow-pool-owner-" + pool::random_token());
            if (!file_io::make_directory(path)) return;
            require(errno == EEXIST, "cannot create private owner test scratch");
        }
        throw std::runtime_error("cannot create unique owner test scratch");
#else
        std::string name = (fs::current_path() / "grow-pool-owner-XXXXXX").string();
        require(::mkdtemp(name.data()), "cannot create owner test scratch"); path = name;
#endif
    }
    ~Directory() { std::error_code error; fs::remove_all(path, error); }
};
std::vector<uint8_t> contents(fs::path const& path)
{ auto in = group_tools::input(path.string()); return {std::istreambuf_iterator<char>(in), {}}; }

// Queue one real successor generation at the encoder boundary, after source
// preparation has joined. Policy, measured local time, deadlines and all
// remote validation remain the production provider's responsibility.
struct PrefetchEncoder final : RemoteEncoder
{
    std::shared_ptr<PoolRemoteEncoder> remote;
    std::function<std::future<void>()> prefetch;
    std::future<void> pending;
    bool queued = false;
    PrefetchEncoder(std::shared_ptr<PoolRemoteEncoder> provider, std::function<std::future<void>()> start)
        : remote(std::move(provider)), prefetch(std::move(start)) {}
    ~PrefetchEncoder() { if (pending.valid()) pending.wait(); }
    std::optional<RemoteEncodedWindow> encode(uint8_t k, std::span<ChunkInput const> chunks, bool summaries) override
    {
        auto result = remote->encode(k, chunks, summaries);
        if (!result && !queued && !chunks.empty() && !chunks.front().contributions.empty())
        { pending = prefetch(); queued = true; }
        return result;
    }
    void observe_local(uint8_t k, std::span<ChunkInput const> chunks,
        std::span<uint64_t const> sizes, std::span<EncodedBlock const> blocks, uint64_t elapsed) override
    { remote->observe_local(k, chunks, sizes, blocks, elapsed); }
    void join() { if (pending.valid()) pending.get(); }
};

void failed_successor_publication(grow_controller_test::Arguments args, fs::path const& root,
    WorkerService& original, WorkerConfig replacement)
{
    using namespace grow_controller_test;
    // Withdraw the original worker if the next signed grant arrives before
    // publication. Later foreground work must use the still-admitted worker.
    WorkerService survivor(replacement);
    args.target = root / "refused-successor"; fs::create_directory(args.target);
    args.command = "fill"; args.directory_farm = true; args.construction = "incremental";
    args.temporary.clear(); args.temporary_capacity = 0; args.destination_temp = "off";
    args.k = 22; args.workers = 2; args.window = args.checkpoint_bytes = uint64_t(64) << 10;
    args.max_members = args.batch = args.group_members = 1; args.max_total_members = args.max_groups = 2;
    args.capacity = uint64_t(256) << 20; args.headroom = 0; args.farming_detect_local = false;
    args.farming_max_eligible_groups = 16; args.pool_output = "packed"; args.pool_codec_ram = 0;
    args.pool_owner = "failed-successor";
    auto const destination = args.target, first = group_path(destination, 0);
    auto const worker = original.status().identity;
    std::atomic<unsigned> selected{0}, before_publication{0}, after_publication{0};
    args.compute_admit_task = [&](WorkerTaskAdmission const& admission)
    {
        auto const attempt = ++selected;
        if (attempt == 1) return;
        if (fs::exists(first))
        {
            // A short predecessor write can finish before a speculative grant.
            auto const expected = before_publication ? survivor.status().identity : worker;
            require(admission.worker.identity == expected, "successor used an unexpected foreground executor");
            ++after_publication;
            return;
        }
        ++before_publication;
        original.control(WorkerControl::Pause); survivor.start();
        throw WorkerAdmissionRejected("declined provisional executor");
    };
    require(run_farming(args) == 0 && before_publication <= 1 && after_publication == 1 &&
        selected == 2 + before_publication,
        "successor admission lost its predecessor or skipped the surviving executor");
    args.compute_admit_task = {}; args.generation_overlap = false;
    args.target = root / "ordinary-successor"; fs::create_directory(args.target);
    require(run_farming(args) == 0, "ordinary successor reference failed");
    for (unsigned i = 0; i < 2; ++i)
        require(contents(group_path(destination, i)) == contents(group_path(args.target, i)),
            "a failed speculative receipt changed the canonical published prefix");

    WorkerCancellation expired; expired.deadline = std::chrono::steady_clock::now();
    require(!receive_optional_successor([&] { expired.check(); return 1; }), "optional receipt retained an expired task");
    bool interrupted = false, invalid = false;
    try { receive_optional_successor([]() -> size_t { throw Interrupted{}; }); }
    catch (Interrupted const&) { interrupted = true; }
    try { receive_optional_successor([]() -> size_t { throw std::logic_error("invalid receipt"); }); }
    catch (std::logic_error const&) { invalid = true; }
    require(interrupted && invalid, "optional receipt suppressed cancellation or a permanent validation failure");
}

void accepted_remote_publication(std::optional<fs::path> retained = {})
{
    using namespace grow_controller_test;
    Directory directory;
    auto const output = retained.value_or(directory.path);
    require(fs::is_directory(output) && fs::is_empty(output), "codec publication fixture requires an empty output directory");
    auto const scratch = directory.path / "codec-scratch"; fs::create_directory(scratch);
    pool_test::Directory certificates;
    auto ca = pool_test::issue(certificates.path, "publication-ca");
    auto certificate = pool_test::issue(certificates.path, "publication-host", &ca);
    auto tls = pool_test::tls(ca, certificate); tls.timeout_ms = 2000; tls.max_frame_bytes = 65536;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0};
    offered.tls = tls; offered.capacity = {1, 1, uint64_t(512) << 20, 0}; offered.generation_threads = 1;
    offered.max_task_bytes = (uint64_t(1) << 20) + tls.max_frame_bytes;
    WorkerService worker(offered); worker.start();
    Arguments args; args.backend = "cpu"; args.workers = 1; args.pool_inflight = 2;
    args.max_ram = uint64_t(1) << 30; args.capacity = uint64_t(64) << 20;
    args.window = uint64_t(1) << 20; args.window_set = true;
    args.checkpoint_bytes = uint64_t(1) << 20; args.checkpoint_set = true;
    args.batch = args.max_members = 8; args.k = 18; args.strength = 2;
    args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, '4');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    args.construction = "external"; args.temporary.push_back(scratch); args.temporary_capacity = uint64_t(256) << 20;
    args.pool_frame_bytes = tls.max_frame_bytes; args.pool_timeout_ms = 10000; args.pool_socket_timeout_ms = tls.timeout_ms;
    args.pool_broker = "127.0.0.1:" + std::to_string(offered.broker.port);
    args.pool_owner = "owner-codec-publication"; args.pool_ca = tls.ca_file; args.pool_cert = tls.cert_file; args.pool_key = tls.key_file;
    args.pool_codec_ram = uint64_t(64) << 20;
    Keys keys(args); auto const requested = keys.derive(0);
    auto scheduler = std::make_shared<CodecScheduler>(1);
    WorkerConfig pressure_config; pressure_config.capacity = {1, 1, uint64_t(512) << 20, 0};
    pressure_config.generation_threads = 1; pressure_config.cpu_scheduler = scheduler;
    pressure_config.tls.max_frame_bytes = tls.max_frame_bytes; pressure_config.max_task_bytes = offered.max_task_bytes;
    unsigned completed_prefetch = 0;
    WorkerService pressure(pressure_config);
    auto backend = std::make_unique<Backend>(); backend->cpu_scheduler = scheduler;
    backend->initialize(args, requested, 0, false); backend->configure_network(args);
    auto generate_successor = [&]
    {
        auto const id = requested.params.get_plot_params_for_index(uint16_t(args.max_members)).get_plot_id().data();
        WorkerCancellation cancel; cancel.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        pressure.generate_local(18, 2, id, [](auto const&, uint64_t) {}, cancel, pool::Kind::GenerateCpu);
        ++completed_prefetch;
    };
    auto encoder = std::make_shared<PrefetchEncoder>(backend->pool_remote, [&]
    {
        // Generation can leave a cached observation of its occupied slot.
        // Wait for the owner to see the idle worker before timing local queue pressure.
        auto const client = backend->acquire_pool_client();
        auto const until = std::chrono::steady_clock::now() + std::chrono::milliseconds(args.pool_timeout_ms);
        for (;;)
        {
            auto const capacity = client->generation_capacity(18, 2);
            if (capacity.slots && capacity.availability.codec_available) break;
            require(std::chrono::steady_clock::now() < until, "publication worker did not become available after generation");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return scheduler->submit(generate_successor);
    });
    backend->test_remote = encoder;
    std::vector<std::vector<uint64_t>> raw;
    {
        worker_detail::CpuLimit limit(1);
        for (uint16_t member = 0; member < args.max_members; ++member)
            raw.push_back(worker_original_reference(18, 2,
                requested.params.get_plot_params_for_index(member).get_plot_id().data()));
    }
    Config reference; reference.capacity_bytes = args.capacity; reference.window_bytes = args.window;
    reference.codec_bytes = uint64_t(8) << 20; reference.checkpoint_bytes = args.checkpoint_bytes;
    reference.memory_limit_bytes = args.max_ram; reference.io_overlap = false; reference.workers = 1; reference.cpu_scheduler = scheduler;
    auto const canonical = output / "reference.gplot";
    { Engine engine(canonical.string(), reference, false, initial_identity(requested)); engine.confirm_identity();
      require(engine.append(Members(raw)).accepted, "remote codec publication reference did not fit"); }
    args.target = output / "remote.gplot";
    auto const metrics = run_group(args, args.target, nullptr, requested, uint16_t(args.max_members), args.capacity, 0, backend);
    encoder->join();
    require(encoder->queued && completed_prefetch == 1, "codec publication did not run finite actual successor generation");
    require(backend->pool_remote->counters.calibration_windows.load() && backend->pool_remote->counters.accepted_windows.load() &&
        backend->pool_remote->counters.accepted_chunks.load(), "remote final codec was not accepted through complete publication");
    require(metrics.members == args.max_members && contents(args.target) == contents(canonical),
        "accepted remote codec changed complete canonical publication");
    if (retained) print_metrics(args, metrics, *backend);
    else
    {
        auto replacement = pool_test::issue(certificates.path, "successor-worker", &ca);
        offered.tls = pool_test::tls(ca, replacement); offered.tls.timeout_ms = tls.timeout_ms;
        failed_successor_publication(args, directory.path, worker, std::move(offered));
    }
}

void repartition_and_global_admission()
{
    using namespace grow_controller_test;
    Directory directory;
    pool_test::Directory certificates;
    auto ca = pool_test::issue(certificates.path, "owner-ca");
    auto certificate = pool_test::issue(certificates.path, "owner-host", &ca);
    auto tls = pool_test::tls(ca, certificate); tls.timeout_ms = 10000; tls.max_frame_bytes = 4096;
    grow_controller_test::Identity requested{PlotGroupParams(PlotGroupId(std::string(64, '7')), 18, 2, 0), {}};
    std::vector<std::array<uint8_t, 32>> ids;
    std::vector<std::vector<uint64_t>> raw;
    worker_detail::CpuLimit cpu(1);
    for (uint16_t member = 0; member < 4; ++member)
    {
        ids.push_back(requested.params.get_plot_params_for_index(member).get_plot_id().data());
        raw.push_back(worker_original_reference(18, 2, ids.back()));
    }
    auto const chunks = PlotGroupFile::getChunkCountForK(18);
    std::vector<std::vector<uint64_t>> symbols(raw.size(), std::vector<uint64_t>(size_t(chunks)));
    uint64_t single_work = 0;
    for (size_t member = 0; member < raw.size(); ++member)
    {
        for (auto value : raw[member]) ++symbols[member][size_t(value >> (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS))];
        for (auto count : symbols[member]) single_work = std::max(single_work, ChunkCodec::estimate_work(18, 0, count));
    }
    single_work = std::max(single_work, uint64_t(tls.max_frame_bytes));
    for (size_t member = 0; member + 1 < raw.size(); ++member)
    {
        uint64_t combined_work = 0;
        for (size_t chunk = 0; chunk < size_t(chunks); ++chunk)
            combined_work = std::max(combined_work,
                ChunkCodec::estimate_work(18, 0, symbols[member][chunk] + symbols[member + 1][chunk]));
        require(combined_work > single_work, "owner fixture does not distinguish grouped and singleton scratch");
    }

    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.tls.max_frame_bytes = 65536;
    options.lease_ms = options.worker_ms = 10000;
    RunningBroker broker(options);
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0};
    offered.tls = tls; offered.capacity = {2, 2, uint64_t(512) << 20, 0};
    offered.generation_threads = 1; offered.max_batch_members = 4; offered.max_task_bytes = single_work;
    offered.cpu_scheduler = std::make_shared<CodecScheduler>(offered.capacity.cpu_threads);
    WorkerService worker(offered); worker.start();
    Arguments args; args.backend = "cpu"; args.workers = 2; args.pool_inflight = 1;
    args.max_ram = uint64_t(1) << 30; args.window = uint64_t(1) << 20;
    args.pool_frame_bytes = tls.max_frame_bytes; args.pool_timeout_ms = 10000; args.pool_socket_timeout_ms = tls.timeout_ms;
    args.pool_broker = "127.0.0.1:" + std::to_string(offered.broker.port);
    args.pool_owner = "owner-repartition"; args.pool_ca = tls.ca_file; args.pool_cert = tls.cert_file; args.pool_key = tls.key_file;
    Backend backend; backend.cpu_scheduler = std::make_shared<CodecScheduler>(2);
    backend.initialize(args, requested, 0, false);
    {
        Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20};
        auto const serial = Backend::fresh_receive_bytes(args), ahead = Backend::contribution_read_ahead_bytes(args);
        auto const held = args.max_ram - serial;
        require(ahead && Backend::fresh_receive_grant(args, held + 1) == 0 &&
            Backend::fresh_receive_grant(args, held) == serial && Backend::fresh_receive_grant(args, held - ahead + 1) == serial &&
            Backend::fresh_receive_grant(args, held - ahead) == serial + ahead, "fresh receiver lost its serial or read-ahead RAM boundary");
        uint64_t const receiving = producer.pool_receive_bytes(0, raw.size());
        uint64_t const cache = MemberBatchHash::memory_bytes(raw.size());
        producer.budget = producer.fixed + backend.retained() + receiving + cache;
        require(producer.pool_hash_bytes(raw.size(), receiving, 0) == cache &&
                producer.pool_hash_bytes(raw.size(), receiving, 1) == 0,
            "optional hash cache exceeded the pooled batch grant");
    }
    auto client = backend.acquire_pool_client();
    // Warm the real measured policy. A subsequent grouped request is legal
    // under advertised generation RAM but cannot fit this worker's codec cap.
    for (unsigned attempt = 0; attempt < 4 && client->suggested_batch_members(18, 2, 4, pool::GenerationOutput::Contributions) == 1; ++attempt)
        client->contributions(18, 2, {&ids.front(), 1}, [](auto, auto, auto) {});
    require(client->suggested_batch_members(18, 2, 4, pool::GenerationOutput::Contributions) > 1,
        "owner fixture did not learn contribution batching");
    auto const before = client->stats();

    Config cfg; cfg.capacity_bytes = uint64_t(64) << 20; cfg.window_bytes = args.window;
    cfg.codec_bytes = uint64_t(8) << 20; cfg.checkpoint_bytes = 65536;
    cfg.memory_limit_bytes = args.max_ram; cfg.io_overlap = false; cfg.workers = 1; cfg.cpu_scheduler = backend.cpu_scheduler;
    auto const initial = initial_identity(requested);
    auto const reference = directory.path / "reference.gplot";
    { Engine engine(reference.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(Members(raw)).accepted, "owner canonical reference did not fit"); }
    uint16_t const batch = uint16_t(raw.size());
    std::array<std::unique_ptr<Construction>, 2> engines;
    std::array<std::unique_ptr<ExternalEngine::ContributionInput>, 2> inputs;
    std::array<fs::path, 2> paths;
    for (size_t destination = 0; destination < engines.size(); ++destination)
    {
        auto scratch = directory.path / ("scratch-" + std::to_string(destination)); fs::create_directory(scratch);
        paths[destination] = directory.path / ("output-" + std::to_string(destination) + ".gplot");
        ExternalConfig external{{{scratch.string(), uint64_t(256) << 20}}, 65536, 2};
        engines[destination] = std::make_unique<Construction>(paths[destination].string(), cfg, false, initial, "external", external);
        engines[destination]->confirm_identity();
        inputs[destination] = engines[destination]->external->prepare_contributions(batch,
            fragment_bound(requested.params) / sizeof(uint64_t));
        require(bool(inputs[destination]), "owner contribution reservation did not fit");
    }
    std::array<std::future<size_t>, 2> receiving;
    for (size_t destination = 0; destination < receiving.size(); ++destination)
        receiving[destination] = std::async(std::launch::async, [&, destination]
        {
            Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20, 0};
            producer.defer_pool_history = true;
            auto const admitted = producer.contribution_bytes(cfg, batch);
            require(admitted - ExternalEngine::contribution_input_memory_bytes(cfg, batch) ==
                producer.contribution_bytes(cfg, 1) - ExternalEngine::contribution_input_memory_bytes(cfg, 1),
                "serial receivers reserved transport buffers for every member");
            return producer.fill_contributions(*inputs[destination], 0, 0, admitted, "owner scratch fixture member ");
        });
    for (auto& future : receiving) require(future.get() == raw.size(), "owner repartition silently dropped a member");
    auto const stats = client->stats();
    require(stats.peak_generation_members > 1 && stats.failed_tasks > before.failed_tasks && backend.pool_retries.load(),
        "owner never recovered a refused grouped contribution");
    require(stats.peak_active_generation <= args.pool_inflight, "destinations exceeded the shared owner generation cap");
    require(stats.completed_contributions >= before.completed_contributions + inputs.size() * raw.size(),
        "owner repartition did not use separately fenced singleton tasks");
    for (size_t destination = 0; destination < engines.size(); ++destination)
    {
        auto& engine = *engines[destination];
        require(inputs[destination]->sealed() && engine.append(*inputs[destination], batch).accepted,
            "owner repartition did not seal an appendable exact membership");
        inputs[destination].reset();
        if (!destination)
        {
            auto const admitted = engine.memory_bytes(batch, false);
            engine.prepare_finish();
            auto const serial = engine.memory_bytes(batch, false);

            // Keep the TLS worker in its request parser before it can return
            // a first frame. The owner and the remote worker have separate CPU gates.
            grow_controller_test::Identity const peer{PlotGroupParams(PlotGroupId(std::string(64, '8')), 18, 2, 0), {}};
            std::future<Member> receiver;
            std::optional<CodecScheduler::Lease> pressure(std::in_place, offered.cpu_scheduler.get(), offered.cpu_scheduler->workers());
            receiver = std::async(std::launch::async, [&] { return backend.generate_pool(peer, 0, args); });
            auto const worker_waiting = [&]
            { auto const status = worker.status(); return status.accepted_tasks && status.active.cpu_threads; };
            auto const until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!worker_waiting() && std::chrono::steady_clock::now() < until)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            bool const receiving = receiver.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
            require(worker_waiting() && receiving, "the remote generation request did not wait before first output");

            auto const local_consumers = 1u + unsigned(packing_cpu_consumer(receiving, {}));
            auto const workers = backend.cpu_scheduler->workers() / local_consumers;
            auto const workspace = engine.packing_workspace(batch, false, admitted, workers);
            require(workspace > serial && workspace <= admitted, "a waiting remote receiver withheld the packer's CPU workspace");
            engine.set_workers(workers, batch, false, workspace);
            OrderedWriter writer([] {});
            writer.submit([&] { (void)engine.finish(); }).get();
            require(engine.external->stats().packing_workers == workers && workers > 1,
                "final packing did not use its wider worker grant");
            pressure.reset();
            require(receiver.get().size(), "the waiting receiver did not complete after packing");
        }
        else engine.finish();
        require(contents(paths[destination]) == contents(reference), "owner scratch repartition changed canonical publication");
    }
    // Feed the next external group through the same asynchronous contribution
    // receiver as fill. Its full output writer stays fenced until the predecessor releases it.
    {
        auto target = ScratchRoot::inspect(directory.path); target.device = {"target", {"target"}, false, true};
        std::vector<ScratchRoot> scratch;
        for (std::string name : {"left", "right"})
        {
            auto const path = directory.path / name; fs::create_directory(path);
            auto root = ScratchRoot::inspect(path); root.device = {name, {name}, false, true};
            scratch.push_back(std::move(root));
        }
        ScratchPool pool(scratch, 0);
        require(pool.try_reserve_output(0, target, cfg.capacity_bytes) && pool.try_reserve_output(1, target, cfg.capacity_bytes),
            "successor test could not reserve both target files");
        pool.seal_recovery();
        require(pool.grant(0, cfg.capacity_bytes, cfg.capacity_bytes, mib, 2).has_value(), "old writer lost its scratch");
        auto old = pool.try_io(0);
        auto grant = pool.grant(1, cfg.capacity_bytes, cfg.capacity_bytes, mib, 2);
        require(old && grant && !pool.try_io(1), "same-output owners acquired two full writer leases");
        GroupSuccessor next(directory.path / "next-contributions.gplot", requested, batch, cfg.capacity_bytes);
        next.io = pool.try_stage_io(1); require(bool(next.io), "disjoint successor staging was unavailable");
        next.engine = std::make_unique<Construction>(next.path.string(), cfg, false, initial, "external", grant);
        next.engine->confirm_identity(true);
        Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20};
        next.contribution_bytes = producer.contribution_bytes(cfg, batch);
        ControllerWake wake; std::atomic<bool> cancel{true};
        std::unique_ptr<ExternalEngine::ContributionInput> input;
        auto stopped = receive_successor_contributions(next, args, backend, producer.fixed, 0, wake, cancel, input);
        bool cancelled = false;
        try { (void)stopped.get(); } catch (std::exception const&) { cancelled = true; }
        input.reset();
        require(cancelled && !next.engine->external->committed_member_count(), "cancelled provisional input changed the successor");
        cancel.store(false);
        // Hold the predecessor's real writer lease while the normal speculative
        // receiver obtains and declines a signed grant, independent of disk speed.
        Backend refusing; refusing.mode = "pool"; refusing.cpu_scheduler = backend.cpu_scheduler;
        refusing.pool_client_config = backend.pool_client_config;
        refusing.pool_client_config.owner_id += "-refused-successor";
        bool refused = false;
        refusing.pool_client_config.admit_task = [&](WorkerTaskAdmission const& admission)
        {
            refused = !admission.local_fallback && admission.worker.identity == worker.status().identity &&
                admission.owner_id == refusing.pool_client_config.owner_id && !admission.broker_identity.empty();
            throw WorkerAdmissionRejected("declined provisional executor");
        };
        auto declined = receive_successor_contributions(next, args, refusing, producer.fixed, 0, wake, cancel, input);
        require(!declined.get().members && refused && old && next.io && !next.engine->external->committed_member_count(),
            "declined signed successor escaped its predecessor or retained a provisional member");
        input.reset();
        auto receiving = receive_successor_contributions(next, args, backend, producer.fixed, 0, wake, cancel, input);
        auto const received = receiving.get();
        require(received.members == batch && received.peak <= args.max_ram && !next.engine->external->committed_member_count(),
            "successor contributions did not finish privately inside the RAM bound");
        next.io.reset(); require(!pool.try_io(1), "received successor bypassed its predecessor's publication fence");
        old.reset(); auto writing = pool.try_io(1); require(bool(writing), "published predecessor kept the successor fenced");
        require(next.engine->append(*input, batch, true).accepted, "ordinary handoff rejected the sealed successor");
        input.reset(); next.engine->finish();
        require(contents(next.path) == contents(reference), "asynchronous external successor differs from the canonical group");
    }
    // Network receivers already own compact words; they do not pass through
    // CompactMemberBuilder, which alone populates the producer prefix cache.
    // Transfer a real prefetched group and then exercise the single-owner
    // append/prefetch loop that previously truncated an empty cache.
    auto const prefix_reference = directory.path / "packed-reference.gplot";
    { Engine engine(prefix_reference.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(Members(raw).subspan(0, 3)).accepted, "packed owner reference did not fit"); }
    {
        auto const scratch = directory.path / "packed-scratch"; fs::create_directory(scratch);
        auto const path = directory.path / "packed-external.gplot";
        ExternalConfig external{{{scratch.string(), uint64_t(256) << 20}}, 65536, 2};
        ExternalEngine engine(path.string(), cfg, external, false, initial); engine.confirm_identity();
        Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20};
        require(!prepare_pool_input(engine, cfg, requested, 3, 1), "packed input bypassed the exact capacity probe");
        auto input = prepare_pool_input(engine, cfg, requested, 3, cfg.capacity_bytes);
        require(input && producer.fill_pool_input(*input, 0, 0, producer.packed_input_bytes(cfg, 3)) == 3 &&
            input->sealed() && engine.append(*input).accepted, "bounded packed scratch receive did not append");
        input.reset(); engine.finish();
        require(backend.pool_scratch_packed_members == 3 && contents(path) == contents(prefix_reference),
            "packed scratch changed the canonical group");
    }
    GroupPrefetch prefetched;
    Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20, 0};
    require(prefetch_group(prefetched, producer, requested, 0, 3, 0,
        "owner packed prefetch member ", [] { return false; }) == 3 && !prefetched.hash,
        "packed pool prefetch retained an unpopulated producer hash");
    std::vector<Member> incoming;
    std::unique_ptr<MemberBatchHash> hash;
    prefetched.take(incoming, hash, 3, 3);
    require(incoming.size() == 3 && !hash, "packed pool prefetch handoff lost its owning members");
    auto const transferred = directory.path / "packed-prefetch.gplot";
    { Engine engine(transferred.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(incoming, hash.get()).accepted, "packed pool prefetch append did not fit"); }
    require(contents(transferred) == contents(prefix_reference), "packed prefetch changed canonical bytes");
    auto packed_args = args; packed_args.pool_owner = "owner-packed-prefetch";
    packed_args.batch = 2; packed_args.construction = "incremental";
    auto const packed = directory.path / "packed-pipeline.gplot";
    std::unique_ptr<Backend> packed_backend;
    auto const packed_result = run_group(packed_args, packed, nullptr, requested, 3, cfg.capacity_bytes, 0, packed_backend);
    require(packed_result.members == 3 && contents(packed) == contents(prefix_reference),
        "single-owner packed pipeline prefetch changed canonical publication");
}

void retained_terminal_grants(WorkerGenerationCapacity const& capacity)
{
    // No first grant with an ordinary continuation can reach the 40-unit
    // physical bound. Permit the exact, revalidated continuation on seven
    // writers only when the complete input roster is available.
    std::array<unsigned, 7> packed{3, 1, 1, 2, 3, 3, 1}, write{27, 21, 29, 26, 36, 31, 37};
    std::array<ExecutionReady, 7> ready;
    std::array<ExecutionLane, 7> lanes;
    std::array<TerminalGenerationBatch, 7> inputs{};
    for (size_t i = 0; i < lanes.size(); ++i)
    {
        auto& work = lanes[i].work;
        work.generation = packed[i]; work.cpu = 1; work.destination = write[i];
        work.batches = 1; work.known = work.io_overlap = true; lanes[i].destination = i;
        ready[i] = {i, 0, work.cpu + work.destination, true, uint16_t(packed[i]), false, work.generation};
        inputs[i] = {i, i, 1, 0, uint16_t(packed[i]), {}, {work.generation, work.cpu, 0, work.destination},
            ready[i].time_to_write};
    }
    require(capacity.slots == 1 && capacity.roster && capacity.roster->complete,
        "the authenticated terminal fixture lost its sole worker");
    GenerationSupply supply{capacity.slots, capacity.roster}; TerminalGenerationPlan saved;
    require(saved.choose(ready, lanes, {}, 7, 7, 1, supply, true, 1, 100) == 4 && !saved.count,
        "the single-grant fallback claimed an unvalidated terminal continuation");
    require(saved.choose(ready, lanes, inputs, 7, 7, 1, supply, true, 1, 100) == 6 && saved.count == 7,
        "the seven-writer plan did not attain the shorter durable bound");
    saved.grant(6, inputs[6].count);
    require(saved.next(std::span(inputs).first(6), std::span(ready).first(6), supply, 2) == 4,
        "the second necessary grant lost its qualified writer");
    inputs[3].writer = 99;
    require(!saved.next(std::span(inputs).first(6), std::span(ready).first(6), supply, 2),
        "a changed large cohort kept its old continuation");
}

void late_packed_capacity()
{
    using namespace grow_controller_test;
    pool_test::Directory certificates;
    auto ca = pool_test::issue(certificates.path, "packed-start-ca");
    auto certificate = pool_test::issue(certificates.path, "packed-start-host", &ca);
    auto tls = pool_test::tls(ca, certificate); tls.timeout_ms = 5000; tls.max_frame_bytes = 65536;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0};
    offered.tls = tls; offered.capacity = {1, 1, uint64_t(512) << 20, 0}; offered.generation_threads = 1;
    offered.available_resources = [capacity = offered.capacity](unsigned) { return capacity; };
    WorkerService worker(offered);
    Arguments args; args.backend = "cpu"; args.pool_inflight = 2; args.pool_timeout_ms = 15000;
    args.pool_frame_bytes = tls.max_frame_bytes; args.max_ram = uint64_t(1) << 30;
    Backend backend; backend.mode = "pool"; backend.cpu_scheduler = std::make_shared<CodecScheduler>(2);
    auto& config = backend.pool_client_config; config.tls = tls; config.broker = offered.broker;
    config.owner_id = "packed-start-owner"; config.generation_kind = pool::Kind::GenerateCpu;
    config.max_active_generation = args.pool_inflight; config.task_timeout_ms = args.pool_timeout_ms;
    std::mutex mutex; std::vector<WorkerTaskObservation> receipts;
    config.observe = [&](auto const& value) { std::lock_guard lock(mutex); receipts.push_back(value); };
    std::function<void()> on_selected;
    config.selected = [&](auto const&) { if (on_selected) on_selected(); };
    auto client = backend.acquire_pool_client();
    auto empty = client->generation_capacity(18, 2);
    require(empty.roster && empty.roster->complete && !empty.slots, "packed owner did not start with an empty broker");
    grow_controller_test::Identity requested{PlotGroupParams(PlotGroupId(std::string(64, '7')), 18, 2, 0), {}};
    Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20};
    MemberBatchHash hash(3); std::vector<Member> members;
    auto filling = std::async(std::launch::async, [&] { return producer.fill_pool(members, 0, 3, 0, "", {}, nullptr, &hash, 2); });
    pool::BrokerClient control(offered.broker, tls);
    auto const until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!control.status().queued && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    require(control.status().queued, "cold packed owner did not submit useful work");
    worker.start();
    require(filling.get() == 3 && members.size() == 3 && hash.size() == 3 && receipts.size() == 2 &&
        hash.digest_for(Members(members)) == members_digest(18, Members(members), nullptr, [] {}),
        "newly available packed capacity did not batch and hash the unsent pair");
    retained_terminal_grants(client->generation_capacity(18, 2));
    auto const canonical_packed = members_digest(18, Members(members), nullptr, [] {});
    worker_detail::CpuLimit cpu(1);
    for (unsigned i = 0; i < 3; ++i)
    {
        auto const id = requested.params.get_plot_params_for_index(uint16_t(i)).get_plot_id().data();
        auto const receipt = std::find_if(receipts.begin(), receipts.end(), [&](auto const& value) { return value.work_units == (i ? 2 : 1); });
        require(receipt != receipts.end() && receipt->member_id_count == receipt->work_units && receipt->member_ids[i ? i - 1 : 0] == id,
            "packed rescheduling changed the assigned plot IDs");
        std::vector<uint64_t> raw; auto view = members[i].view();
        for (size_t c = 0; c < PlotGroupFile::getChunkCountForK(18); ++c)
            for (auto value : view.region(c)) raw.push_back(value);
        require(raw == worker_original_reference(18, 2, id), "packed rescheduling changed ordered CPU-reference output");
    }
    {
        // A waiting independent receiver must fit with every extant byte grant.
        // Holding the last foreground CPU keeps the final digest on its receiver.
        uint64_t const held_bytes = producer.pool_receive_bytes(0, 3) + MemberBatchHash::memory_bytes(3);
        auto const full = producer.budget;
        producer.budget = producer.fixed + backend.retained() + 2 * held_bytes - 1;
        require(!producer.fits_waiting_receiver(0, 3, held_bytes), "independent receiver exceeded its RAM grant");
        ++producer.budget;
        require(producer.fits_waiting_receiver(0, 3, held_bytes), "independent receiver lost its exact RAM grant");
        producer.budget = full;
        std::atomic<bool> waiting{true}; producer.waiting_receiver = &waiting;
        {
            CodecScheduler::Lease foreground(backend.cpu_scheduler.get(), 1);
            auto check = std::async(std::launch::async, [&] { return producer.writer_can_hash(); });
            require(!check.get(), "handed a digest to a writer without spare CPU");
        }
        // Preserve an already computed prefix while the final authenticated pair
        // goes to the writer. A second real lease must finish before it resumes.
        std::vector<Member> handed;
        handed.push_back(backend.generate_pool(requested, 0, args));
        MemberBatchHash partial_hash(3); partial_hash.append(handed.front().view(), [] {});
        std::promise<void> selected, release;
        auto selected_future = selected.get_future(); auto resume = release.get_future().share();
        std::once_flag first_lease;
        on_selected = [&] { std::call_once(first_lease, [&]
        {
            selected.set_value();
            require(resume.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "handed receipt never resumed");
        }); };
        waiting.store(false);
        auto pair = std::async(std::launch::async, [&] { return producer.fill_pool(handed, 0, 3, held_bytes, "", {}, nullptr, &partial_hash); });
        require(selected_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "handed pair did not receive an authenticated grant");
        waiting.store(true); release.set_value();
        require(pair.get() == 2 && partial_hash.size() == 1, "receiver consumed or lost the final writer digest");
        on_selected = {}; producer.waiting_receiver = nullptr;
        std::promise<void> hashing, received_peer;
        auto hashing_future = hashing.get_future(); auto peer_done = received_peer.get_future().share();
        auto writer = std::async(std::launch::async, [&]
        {
            CodecScheduler::Lease cpu(backend.cpu_scheduler.get(), 1);
            bool started = false;
            return members_digest(18, Members(handed), &partial_hash, [&]
            {
                if (std::exchange(started, true)) return;
                hashing.set_value();
                require(peer_done.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "next receiver waited for writer hashing");
            });
        });
        require(hashing_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "writer did not resume its digest");
        grow_controller_test::Identity peer_identity{PlotGroupParams(PlotGroupId(std::string(64, '9')), 18, 2, 0), {}};
        auto peer_member = backend.generate_pool(peer_identity, 0, args); received_peer.set_value();
        require(peer_member.view().size() && writer.get() == hash.digest_for(Members(members)),
            "overlapping independent receipt changed the canonical writer digest");
    }
    // A successor hashes only with spare owner CPU. Its complete cache or
    // the ordinary writer fallback must bind the very same sealed members.
    std::atomic<bool> stop{false};
    std::future<std::unique_ptr<MemberBatchHash>> skipped;
    {
        CodecScheduler::Lease writer(backend.cpu_scheduler.get());
        skipped = std::async(std::launch::async, [&] { return producer.hash_pool_successor(members, stop); });
        require(skipped.wait_for(std::chrono::seconds(2)) == std::future_status::ready && !skipped.get(),
            "optional successor hash queued for the last foreground CPU");
    }
    unsigned checks = 0; args.cancelled = [&] { return ++checks == 3; };
    bool interrupted = false;
    try { (void)producer.hash_pool_successor(members, stop); }
    catch (Interrupted const&) { interrupted = true; }
    require(interrupted, "optional packed successor hash lost cancellation");
    args.cancelled = {};
    std::promise<void> started_hash, foreground_ready, release_foreground;
    auto foreground = std::async(std::launch::async, [&]
    {
        require(started_hash.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "successor did not start hashing its first member");
        CodecScheduler::Lease writer(backend.cpu_scheduler.get());
        foreground_ready.set_value();
        require(release_foreground.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "optional hash blocked the foreground permit");
    });
    bool first = true;
    args.cancelled = [&]
    {
        if (std::exchange(first, false))
        {
            started_hash.set_value();
            require(foreground_ready.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "writer did not receive its reserved CPU during the optional hash");
        }
        return false;
    };
    GroupPrefetch partial; partial.capacity = 3; partial.ram_rollover = true;
    partial.hash = producer.hash_pool_successor(members, stop);
    release_foreground.set_value(); foreground.get(); args.cancelled = {};
    require(partial.hash && partial.hash->size() == 1, "CPU pressure discarded a completed successor digest prefix");
    partial.members = std::move(members);
    std::unique_ptr<MemberBatchHash> prefix;
    partial.take(members, prefix, 3, 3);
    size_t suffix_polls = 0, writer_polls = 0;
    (void)members_digest(18, Members(members).subspan(1), nullptr, [&] { ++suffix_polls; });
    require(prefix && members_digest(18, Members(members), prefix.get(), [&] { ++writer_polls; }) ==
        hash.digest_for(Members(members)) && writer_polls == suffix_polls,
        "writer reread the confirmed prefix or changed the canonical digest");
    interrupted = false;
    try { (void)members_digest(18, Members(members), prefix.get(), [] { throw Interrupted{}; }); }
    catch (Interrupted const&) { interrupted = true; }
    require(interrupted && prefix->digest_for(Members(members).subspan(0, 1)) == hash.digest_for(Members(members).subspan(0, 1)),
        "cancelling suffix hashing changed the accepted digest prefix");
    std::swap(members[0], members[1]);
    require(members_digest(18, Members(members), prefix.get(), [] {}) ==
        members_digest(18, Members(members), nullptr, [] {}), "writer trusted another member's digest prefix");
    std::swap(members[0], members[1]);
    GroupPrefetch split; split.capacity = 3; split.ram_rollover = true;
    split.members = std::move(members); split.hash = std::move(prefix);
    split.take(members, prefix, 1, 3);
    require(members.size() == 1 && !prefix && !split.hash && split.first == 1 && split.members.size() == 2,
        "short adoption exposed the original SHA state to a differently aligned batch");
    std::vector<Member> remainder;
    split.take(remainder, prefix, 2, 3);
    require(!prefix && remainder.size() == 2, "remaining members inherited another range's digest");
    for (auto& member : remainder) members.push_back(std::move(member));
    require(bool(hash.digest_for(Members(members))), "short adoption reordered the immutable member identities");
    for (uint16_t const cap : {4, 3, 2})
    {
        auto const count = std::min<size_t>(cap, members.size());
        auto const expected = members_digest(18, Members(members).subspan(0, count), nullptr, [] {});
        GroupPrefetch successor; successor.capacity = uint16_t(members.size()); successor.ram_rollover = true;
        successor.hash = producer.hash_pool_successor(members, stop);
        require(successor.hash && successor.hash->digest_for(Members(members)) == hash.digest_for(Members(members)),
            "spare CPU failed to prepare the complete ordinary packed digest");
        successor.members = std::move(members);
        std::unique_ptr<MemberBatchHash> adopted;
        successor.take(members, adopted, 3, cap);
        require(members.size() == count && bool(adopted) == (cap == 3) &&
            members_digest(18, Members(members), adopted.get(), [] {}) == expected,
            "changed RAM grant reused an unadmitted cache or changed its member digest");
    }
    auto second_cert = pool_test::issue(certificates.path, "packed-start-second", &ca);
    offered.tls = pool_test::tls(ca, second_cert); offered.tls.max_frame_bytes = tls.max_frame_bytes;
    WorkerService second(offered); second.start();
    Backend probing; probing.mode = "pool"; probing.cpu_scheduler = backend.cpu_scheduler;
    probing.pool_client_config = config; probing.pool_client_config.owner_id = "packed-probes-owner";
    std::promise<void> relief, release; auto resume = release.get_future().share();
    std::string held; bool relieved = false;
    probing.pool_client_config.selected = [&](auto const& task)
    {
        {
            std::lock_guard lock(mutex);
            if (!held.empty()) return;
            held = task.worker_identity;
        }
        require(resume.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "first packed probe was never released");
    };
    probing.pool_client_config.observe = [&](auto const& task)
    {
        std::lock_guard lock(mutex);
        if (task.worker_identity != held && !std::exchange(relieved, true)) relief.set_value();
    };
    auto const roster = probing.acquire_pool_client()->generation_capacity(18, 2);
    require(roster.roster && roster.roster->complete && roster.roster->routes.size() == 2 && roster.slots == 2,
        "second compatible packed worker was unavailable");
    Producer competing{requested, nullptr, probing, args, args.max_ram, uint64_t(64) << 20};
    std::vector<Member> progress;
    auto filling_two = std::async(std::launch::async, [&]
    { return competing.fill_pool(progress, 0, 3, 0, "", {}, nullptr, nullptr, 2); });
    auto const concurrent = relief.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    release.set_value();
    require(filling_two.get() == 3 && concurrent, "a held first probe kept another compatible worker idle");
    // Train an exact delivery on the direct data route, then start a fresh
    // owner and add a third, unpriced worker. Use actual owner/worker receipts.
    auto settings = config;
    settings.selected = {}; settings.observe = {};
    settings.generation_path = [](auto const&, auto const&, uint32_t) { return std::string(64, 'b'); };
    settings.owner_id = "training-batch"; settings.ephemeral_owner = true; settings.max_active_generation = 3;
    auto admission = [&](WorkerService& service, WorkerControl operation)
    {
        auto const expected = service.control(operation);
        auto const end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;)
        {
            auto const status = control.status();
            if (std::any_of(status.worker_records.begin(), status.worker_records.end(), [&](auto const& row)
                { return row.identity == expected.identity && row.instance == expected.instance && row.admission == expected.admission; })) return;
            require(std::chrono::steady_clock::now() < end, "signed worker admission did not reach the broker");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    admission(second, WorkerControl::Pause);
    std::array<std::array<uint8_t, 32>, 3> ids;
    for (unsigned i = 0; i < ids.size(); ++i) ids[i] = requested.params.get_plot_params_for_index(i).get_plot_id().data();
    {
        WorkerClient training(settings);
        for (unsigned i = 0; i < 2; ++i) training.generate_packed(18, 2, ids, [](auto const&, uint64_t) {});
    }
    admission(second, WorkerControl::Resume);
    auto third_cert = pool_test::issue(certificates.path, "packed-third", &ca);
    offered.tls = pool_test::tls(ca, third_cert); offered.tls.max_frame_bytes = tls.max_frame_bytes;
    WorkerService third(offered); third.start();

    auto const fast = worker.status().identity;
    std::promise<WorkerTaskObservation> first_grant; auto first_future = first_grant.get_future().share();
    std::promise<void> release_batch; auto batch_done = release_batch.get_future().share();
    std::once_flag initial_selection;
    std::vector<WorkerTaskObservation> delivered;
    settings.owner_id = "independent-batches";
    settings.selected = [&](auto const& value)
    {
        bool first = false;
        std::call_once(initial_selection, [&] { first = true; first_grant.set_value(value); });
        if (first) require(batch_done.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
            "the independent packed batch never resumed");
    };
    settings.observe = [&](auto const& value) { std::lock_guard lock(mutex); delivered.push_back(value); };
    auto fenced_settings = settings; fenced_settings.owner_id = "changed-packed-worker"; fenced_settings.selected = {};
    std::vector<WorkerTaskObservation> fenced_receipts;
    fenced_settings.observe = [&](auto const& value) { fenced_receipts.push_back(value); };
    WorkerClient changed(fenced_settings);
    WorkerPackedRoute fenced_route; unsigned fenced_limit = 0;
    auto const fenced_capacity = changed.generation_capacity(18, 2);
    auto const priced_width = changed.packed_generation_width(18, 2, 3, 3, &fenced_route, &fenced_limit,
        std::chrono::steady_clock::now() + std::chrono::seconds(10));
    require(fenced_capacity.slots == 3 && priced_width == 3 && fenced_route.route.identity == fast,
        "the established route was not priced before withdrawing it");
    std::promise<void> pricing, other_sizing;
    auto price_started = pricing.get_future(); auto other_started = other_sizing.get_future().share();
    std::once_flag first_price;
    std::atomic<bool> grant_before_price{false};
    settings.generation_path = [&](auto const&, auto const&, uint32_t)
    {
        std::call_once(first_price, [&]
        {
            pricing.set_value();
            require(other_started.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "no independent pricing peer");
            grant_before_price = first_future.wait_for(std::chrono::milliseconds(75)) == std::future_status::ready;
        });
        return std::string(64, 'b');
    };
    Backend parallel; parallel.mode = "pool"; parallel.cpu_scheduler = backend.cpu_scheduler;
    parallel.pool_client_config = settings;
    auto const ready = parallel.acquire_pool_client()->generation_capacity(18, 2);
    require(ready.roster && ready.roster->complete && ready.roster->routes.size() == 3,
        "the third compatible worker did not join the authenticated roster");
    auto parallel_args = args; parallel_args.pool_inflight = 3;
    Producer established{requested, nullptr, parallel, parallel_args, args.max_ram, uint64_t(64) << 20};
    std::vector<Member> first_members, peer_members;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto first_task = std::async(std::launch::async, [&]
        { return established.fill_pool(first_members, 0, 3, 0, "", {}, nullptr, nullptr, 3, deadline); });
    bool const pricing_seen = price_started.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    grow_controller_test::Identity peer_ids{PlotGroupParams(PlotGroupId(std::string(64, '8')), 18, 2, 0), {}};
    Producer newcomer{peer_ids, nullptr, parallel, parallel_args, args.max_ram, uint64_t(64) << 20};
    auto peer_task = std::async(std::launch::async, [&]
        { other_sizing.set_value(); return newcomer.fill_pool(peer_members, 0, 3, 0, "", {}, nullptr, nullptr, 3, deadline); });
    auto first_ready = first_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    auto chosen = first_ready ? first_future.get() : WorkerTaskObservation{};
    if (!pricing_seen || grant_before_price || chosen.worker_identity != fast || chosen.work_units != 3)
    {
        release_batch.set_value(); first_task.get(); peer_task.get();
        throw std::runtime_error("an independent probe preempted the established exact packed batch");
    }
    bool const peer_finished = peer_task.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    release_batch.set_value();
    require(first_task.get() == 3 && peer_task.get() == 3 && peer_finished &&
        std::any_of(delivered.begin(), delivered.end(), [&](auto const& value)
            { return value.worker_identity == third.status().identity && value.work_units == 1; }),
        "a paid batch on the busy route suppressed a newly eligible worker's independent delivery");
    require(members_digest(18, Members(first_members), nullptr, [] {}) == canonical_packed,
        "the paid route changed the ordinary canonical packed digest");

    // Size from the same verified price, then retire its actual worker before
    // the request. No ID was sent under the earlier offer. Receive the three
    // originals as ordinary grants under their original deadline.
    settings.selected = {}; settings.owner_id = "limited-packed-owner";
    settings.max_active_generation = 1;
    {
        WorkerClient serial(settings);
        WorkerPackedRoute route; unsigned singles = 0;
        require(serial.packed_generation_width(18, 2, 3, 3, &route, &singles, deadline) == 1,
            "one owner receiver claimed an independent packed batch");
    }
    settings.max_active_generation = 3; delivered.clear();
    // Isolate the already measured worker from the now-paid helpers using a
    // new route fingerprint for each helper, as on an owner interface change.
    settings.generation_path = [fast](auto const& route, auto const&, uint32_t)
        { return std::string(64, route.identity == fast ? 'b' : 'c'); };
    Backend constrained; constrained.mode = "pool"; constrained.cpu_scheduler = parallel.cpu_scheduler;
    constrained.pool_client_config = settings; constrained.pool_client_config.owner_id = "constrained-paid-batch";
    Producer constrained_producer{requested, nullptr, constrained, parallel_args, args.max_ram, uint64_t(64) << 20};
    constrained_producer.budget = constrained_producer.fixed + constrained.retained() +
        constrained_producer.pool_receive_bytes(0, 3) - 1;
    std::vector<Member> limited;
    auto count = constrained_producer.fill_pool(limited, 0, 3, 0, "", {}, nullptr, nullptr, 3, deadline);
    require(count && !delivered.empty() && delivered.front().work_units == 1 &&
        constrained_producer.peak <= constrained_producer.budget,
        "an unfunded exact width committed its unpriced remainder or exceeded owner RAM");
    constrained_producer.budget = args.max_ram;
    if (count < 3) constrained_producer.fill_pool(limited, 0, 3, 0, "", {}, nullptr, nullptr, 3, deadline);
    require(members_digest(18, Members(limited), nullptr, [] {}) == canonical_packed,
        "the smaller receive grant did not preserve the canonical member set");

    admission(worker, WorkerControl::Pause);
    changed.generate_packed(18, 2, ids, [](auto const&, uint64_t) {}, {}, {},
        std::chrono::steady_clock::now() + std::chrono::seconds(10), &fenced_route);
    require(fenced_receipts.size() == ids.size(), "withdrawing a priced worker issued unmeasured wide work elsewhere");
    for (size_t i = 0; i < ids.size(); ++i)
        require(fenced_receipts[i].worker_identity != fast && fenced_receipts[i].work_units == 1 &&
            fenced_receipts[i].member_id_count == 1 && fenced_receipts[i].member_ids[0] == ids[i],
            "a changed packed grant lost or duplicated an ordered member");

    // Only the first worker has confirmed this pair and triple on the owner's
    // path. A second owner reserves it between pricing and the real grant.
    admission(third, WorkerControl::Pause); admission(second, WorkerControl::Pause);
    admission(worker, WorkerControl::Resume);
    auto calibrated = settings;
    calibrated.generation_path = [](auto const&, auto const&, uint32_t) { return std::string(64, 'd'); };
    calibrated.owner_id = "tail-prices"; calibrated.selected = {}; calibrated.observe = {};
    {
        WorkerClient training(calibrated);
        for (unsigned width : {3u, 3u, 2u, 2u})
            training.generate_packed(18, 2, std::span(ids).first(width), [](auto const&, uint64_t) {});
    }
    for (unsigned mode : {0u, 3u, 2u, 1u})
    {
        bool const stop_on_decline = mode == 0, expire_on_decline = mode == 3;
        unsigned const receivers = mode == 2 ? 2 : 1;
        auto trial = calibrated;
        // Each completion test starts with an unpriced alternative. The two
        // cancelled cases leave no delivery sample on their worker.
        auto& other = mode == 1 ? third : second;
        if (mode == 1) admission(second, WorkerControl::Pause);
        admission(other, WorkerControl::Resume);
        auto competing = control.open_owner("other-tail-owner-" + std::to_string(mode));
        pool::Task hold; hold.task_id = "competing"; hold.sequence = 1; hold.kind = pool::Kind::GenerateCpu;
        hold.compatibility = worker_compatibility(); hold.k = 18; hold.strength = 2; hold.resources = {1,1,1,0};
        std::optional<pool::Lease> held;
        std::promise<void> prefix, suffix, resume;
        auto prefix_at = prefix.get_future().share(), suffix_at = suffix.get_future().share(), release = resume.get_future().share();
        std::atomic<unsigned> visits{0}; std::atomic<bool> stop{false};
        trial.owner_id = "tail-reshape-" + std::to_string(mode);
        trial.generation_path = [&](auto const& route,auto const&,uint32_t)
        {
            if (route.identity == fast)
            {
                unsigned const check = ++visits;
                if (check == 2)
                {
                    admission(other, WorkerControl::Pause);
                    held = control.acquire(competing, hold);
                    require(held && held->worker_identity == fast, "competing owner did not acquire the priced worker");
                    admission(other, WorkerControl::Resume);
                }
                // Make the release observable to the concurrently priced
                // suffix, without letting the prefix start data transfer.
                if (check == 3 && receivers == 2)
                    require(prefix_at.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                        "missing prefix grant during independent suffix pricing");
            }
            return std::string(64, 'd');
        };
        std::vector<WorkerTaskObservation> grants, successes;
        std::once_flag first_single;
        trial.selected = [&](auto const& value)
        {
            { std::lock_guard lock(mutex); grants.push_back(value); }
            if (value.work_units == 3 && stop_on_decline) { stop = true; return; }
            if (value.work_units == 3 && expire_on_decline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            if (value.work_units == 1) std::call_once(first_single,[&]
            {
                auto same = control.acquire(competing,hold);
                require(held && same && same->token == held->token, "a batch retry preempted another owner's grant");
                control.cancel(competing,hold.task_id,hold.attempt); prefix.set_value();
                require(release.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "prefix gate did not resume");
            });
            if (value.work_units == 2 && value.worker_identity == fast)
            {
                suffix.set_value();
                require(release.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "suffix gate did not resume");
            }
        };
        trial.observe = [&](auto const& value) { std::lock_guard lock(mutex); successes.push_back(value); };
        Backend tails; tails.mode = "pool"; tails.cpu_scheduler = backend.cpu_scheduler; tails.pool_client_config = trial;
        auto local_args = args; local_args.pool_inflight = receivers; local_args.command = "fill";
        if (expire_on_decline) local_args.pool_timeout_ms = 1000;
        Producer tail{requested,nullptr,tails,local_args,args.max_ram,uint64_t(64)<<20};
        auto cli = tails.acquire_pool_client();
        require(cli->generation_capacity(18,2).slots == 2, "missing eligible workers before pricing the suffix");
        std::vector<Member> output; MemberBatchHash digest(3);
        auto future = std::async(std::launch::async,[&] {
            return tail.fill_pool(output,0,3,0,"",{},&stop,&digest,3,
                std::chrono::steady_clock::now()+std::chrono::seconds(10),receivers);
        });
        if (stop_on_decline || expire_on_decline)
        {
            bool cancelled = false;
            try { (void)future.get(); } catch (std::exception const&) { cancelled = true; }
            auto same = control.acquire(competing, hold);
            require(cancelled && (stop || expire_on_decline) && held && same && same->token == held->token &&
                grants.size() == 1 && successes.empty() && output.empty(),
                "a stopped or expired paid decline issued more Work or exposed a provisional member");
            control.cancel(competing,hold.task_id,hold.attempt);
            continue;
        }
        bool const started = prefix_at.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        bool const overlap = started && suffix_at.wait_for(receivers > 1 ? std::chrono::milliseconds(2000) :
            std::chrono::milliseconds(0)) == std::future_status::ready;
        auto const live = control.status({});
        resume.set_value();
        require(future.get() == 3 && started && grants.front().work_units == 3 && grants.front().worker_identity != fast,
            "the ordinary producer did not lose and reshape the signed triple");
        require(overlap == (receivers == 2) && live.granted == receivers && !live.running &&
            cli->stats().peak_active_generation == receivers, "the packed suffix exceeded or failed to use the funded receivers");
        require(digest.digest_for(Members(output)) == canonical_packed && !successes.empty(),
            "repricing changed ordered canonical packed bytes");
        if (receivers == 2)
        {
            auto pair = std::ranges::find_if(successes,[&](auto const& item){return item.worker_identity==fast && item.work_units==2;});
            auto one = std::ranges::find_if(successes,[&](auto const& item){return item.worker_identity!=fast && item.work_units==1;});
            require(successes.size() == 2 && one != successes.end() && pair != successes.end() &&
                one->member_ids[0] == ids[0] && pair->member_ids[0] == ids[1] && pair->member_ids[1] == ids[2],
                "actual signed retries did not cover the original prefix and priced suffix exactly once");
        }
    }

}

void owner_broker_restart()
{
    using namespace grow_controller_test;
    Directory directory;
    pool_test::Directory certificates;
    auto ca = pool_test::issue(certificates.path, "restart-ca");
    auto certificate = pool_test::issue(certificates.path, "restart-host", &ca);
    auto tls = pool_test::tls(ca, certificate); tls.timeout_ms = 500; tls.max_frame_bytes = 4096;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.tls.max_frame_bytes = 65536;
    options.lease_ms = options.worker_ms = 3000;
    auto broker = std::make_unique<RunningBroker>(options);
    options.listen = broker->server.endpoint();
    WorkerConfig offered; offered.broker = options.listen; offered.listen = {"127.0.0.1", 0};
    offered.tls = tls; offered.capacity = {2, 2, uint64_t(512) << 20, 0}; offered.generation_threads = 1;
    WorkerService worker(offered); worker.start();
    grow_controller_test::Identity requested{PlotGroupParams(PlotGroupId(std::string(64, '6')), 18, 2, 0), {}};
    Arguments args; args.backend = "cpu"; args.workers = 2; args.pool_inflight = 2;
    args.max_ram = uint64_t(1) << 30; args.window = uint64_t(1) << 20;
    args.pool_frame_bytes = tls.max_frame_bytes; args.pool_timeout_ms = 5000; args.pool_socket_timeout_ms = tls.timeout_ms;
    args.pool_broker = "127.0.0.1:" + std::to_string(options.listen.port);
    args.pool_owner = "owner-restart"; args.pool_ca = tls.ca_file; args.pool_cert = tls.cert_file; args.pool_key = tls.key_file;
    Backend backend; backend.cpu_scheduler = std::make_shared<CodecScheduler>(2);
    backend.initialize(args, requested, 0, false);
    (void)backend.generate_pool(requested, 0, args); // Establish the old epoch and a reusable control channel.
    Config cfg; cfg.capacity_bytes = uint64_t(64) << 20; cfg.window_bytes = args.window;
    cfg.codec_bytes = uint64_t(8) << 20; cfg.checkpoint_bytes = 65536; cfg.memory_limit_bytes = args.max_ram;
    cfg.io_overlap = false; cfg.workers = 1; cfg.cpu_scheduler = backend.cpu_scheduler;
    auto const initial = initial_identity(requested);
    worker_detail::CpuLimit cpu(1);
    std::vector<std::vector<uint64_t>> expected;
    for (uint16_t member = 0; member < 2; ++member)
        expected.push_back(worker_original_reference(18, 2,
            requested.params.get_plot_params_for_index(member).get_plot_id().data()));
    auto const raw_reference = directory.path / "restart-raw-reference.gplot";
    { Engine engine(raw_reference.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(Members(expected).subspan(0, 1)).accepted, "restart raw reference did not fit"); }
    auto restart_after_retries = [&](auto& task, uint64_t before)
    {
        // A refused connection can take a full socket timeout. Bring the
        // broker back after two observed retries with the task still pending.
        auto const until = std::chrono::steady_clock::now() + std::chrono::milliseconds(args.pool_timeout_ms);
        while (backend.pool_retries.load() < before + 2 && std::chrono::steady_clock::now() < until &&
            task.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout) {}
        bool const pending = backend.pool_retries.load() >= before + 2 &&
            task.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
        broker = std::make_unique<RunningBroker>(options);
        return pending;
    };
    uint64_t const raw_retries = backend.pool_retries.load();
    broker.reset();
    auto raw = std::async(std::launch::async, [&] { return backend.generate_pool(requested, 0, args); });
    require(restart_after_retries(raw, raw_retries), "raw owner did not stay pending across two broker retries");
    require(raw.wait_for(std::chrono::seconds(8)) == std::future_status::ready,
        "raw owner did not recover within its original task deadline");
    std::vector<Member> member; member.push_back(raw.get());
    auto const raw_path = directory.path / "restart-raw.gplot";
    { Engine engine(raw_path.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(member).accepted, "restarted raw owner append did not fit"); }
    require(contents(raw_path) == contents(raw_reference), "broker restart changed packed-member publication");

    auto scratch = directory.path / "restart-scratch"; fs::create_directory(scratch);
    auto const contribution_path = directory.path / "restart-contributions.gplot";
    ExternalConfig external{{{scratch.string(), uint64_t(128) << 20}}, 65536, 2};
    ExternalEngine engine(contribution_path.string(), cfg, external, false, initial); engine.confirm_identity();
    auto input = engine.prepare_contributions(2, fragment_bound(requested.params) / sizeof(uint64_t));
    require(bool(input), "restart contribution reservation did not fit");
    uint64_t const contribution_retries = backend.pool_retries.load();
    broker.reset();
    auto contribution = std::async(std::launch::async, [&]
    {
        Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20, 0};
        producer.defer_pool_history = true;
        return producer.fill_contributions(*input, 0, 0, producer.contribution_bytes(cfg, 2), "restart member ");
    });
    require(restart_after_retries(contribution, contribution_retries),
        "contribution owner did not stay pending across two broker retries");
    require(contribution.wait_for(std::chrono::seconds(8)) == std::future_status::ready && contribution.get() == 2,
        "contribution owner did not recover its complete membership");
    require(engine.append(*input).accepted, "restarted contribution append did not fit"); input.reset(); engine.finish();
    auto const contribution_reference = directory.path / "restart-contributions-reference.gplot";
    { Engine reference(contribution_reference.string(), cfg, false, initial); reference.confirm_identity();
      require(reference.append(Members(expected)).accepted, "restart contribution reference did not fit"); }
    require(contents(contribution_path) == contents(contribution_reference), "broker restart changed contribution publication");

    auto invalid_args = args; invalid_args.pool_owner = "owner-permanent-error";
    invalid_args.pool_ca = (directory.path / "missing-ca.crt").string();
    Backend invalid; invalid.cpu_scheduler = backend.cpu_scheduler; invalid.initialize(invalid_args, requested, 0, false);
    auto permanent = std::async(std::launch::async, [&]
    { try { (void)invalid.generate_pool(requested, 0, invalid_args); return false; } catch (std::exception const&) { return true; } });
    require(permanent.wait_for(std::chrono::seconds(1)) == std::future_status::ready && permanent.get(),
        "permanent TLS configuration failure entered the transient retry loop");
    broker.reset();
    auto deadline_args = args; deadline_args.pool_timeout_ms = 200;
    std::atomic<bool> stop{false};
    auto expired = std::async(std::launch::async, [&]
    { try { (void)backend.generate_pool(requested, 0, deadline_args, &stop); return false; } catch (std::exception const&) { return true; } });
    bool const bounded = expired.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    stop.store(true); // Also drain safely if a regression extends the deadline.
    require(expired.get() && bounded, "transient owner retry extended its original absolute deadline");
}

void private_local_delegation(std::optional<fs::path> retained = {})
{
    using namespace grow_controller_test;
    PrivateLocalSchedule model(3, 100);
    // A previous GPU lane may already be retrying while the next lane is
    // still being constructed. No probe may inspect this partial schedule.
    model.devices.emplace_back();
    require(!model.reserve_probe(1), "a recovery probe used an unpublished private schedule");
    model.devices.clear();
    model.devices.resize(2);
    auto& fast = model.devices[0]; auto& slow = model.devices[1];
    PrivateLocalSchedule::Shape shape{18, 2, 1, pool::GenerationOutput::Packed};
    using Clock = std::chrono::steady_clock;
    auto const now = Clock::now();
    fast.offer = {{3, 1, 90, 1}, {3, 1, 55, 1}, 10};
    slow.offer = {{3, 1, 90, 0}, {3, 1, 20, 0}, 10};
    fast.recent[0] = {shape, 2, 4};
    require(model.choose(shape, now, 100).device == 1, "explicit unmeasured local device never received a trial");
    slow.recent[0] = {shape, 1, 9};
    auto choice = model.choose(shape, now, 100);
    require(choice.device == 0 && !choice.wait_ns && choice.grant.cpu_threads == 2, "fast local GPU did not leave CPU room for its peer");
    fast.started = now - std::chrono::nanoseconds(2); fast.expected_ns = 4;
    fast.held = choice.grant; fast.standing_at_start = 10;
    choice = model.choose(shape, now, 100);
    require(choice.device == 0 && choice.wait_ns == 2, "idle slow local route displaced an earlier fast-device release");
    slow.recent[0].ns = 5;
    choice = model.choose(shape, now, 100);
    require(choice.device == 1 && !choice.wait_ns && choice.grant.cpu_threads == 1, "committed local work was ignored or double-booked CPUs");
    fast.offer.standing_host_bytes = 70;
    require(model.choose(shape, now, 100).device == 1, "retained fast-device buffers hid the only usable local executor");
    slow.offer.demand.host_bytes = 30;
    require(!model.choose(shape, now, 100).possible, "idle persistent GPU RAM was spent a second time");
    fast.offer.standing_host_bytes = 10; slow.offer.demand.host_bytes = 20;
    fast.offer.demand.slots = 0; fast.held = {0, 0, 0, 0};
    fast.offer.gpu_headroom_limited = true;
    choice = model.choose(shape, now, 100);
    require(choice.device == 1 && choice.grant.cpu_threads == 3, "sole eligible slow CPU did not receive the available grant");
    require(model.choose(shape, now, 30).device == SIZE_MAX, "physical host pressure was ignored");
    slow.offer.demand.slots = 0;
    choice = model.choose(shape, now, 100);
    require(choice.device == SIZE_MAX && choice.possible, "temporary GPU memory pressure ended private acquisition");
    fast.offer.demand.host_bytes = 81;
    require(!model.choose(shape, now, 100).possible, "waiting for GPU memory ignored the permanent shared host limit");
    fast.offer.demand.host_bytes = 55;
    fast.offer.gpu_headroom_limited = false;
    require(!model.choose(shape, now, 100).possible, "permanently unsupported private shape waited for GPU memory");
    slow.offer.demand.slots = 1;
    fast.offer.demand = {3, 1, 55, 0};
    choice = model.choose(shape, now, 100);
    require(choice.device == 0 && choice.grant.cpu_threads == 2, "one CPU service occupied every permit despite a fitting peer");
    choice = model.choose(shape, now, 85);
    require(choice.device == 0 && choice.grant.cpu_threads == 3, "a peer blocked by RAM reduced the only useful CPU grant");

    Arguments args; args.local_compute = true; args.quiet = true; args.window = uint64_t(1) << 20;
    args.pool_frame_bytes = 4096; args.pool_inflight = 2; args.pool_offer_cpus = 1;
    args.pool_offer_ram = uint64_t(240) << 20; args.pool_timeout_ms = 10000;
    args.k = 18; args.strength = 2; args.seed = std::string(64, '5'); args.pool_ph = std::string(64, '4');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    auto const requested = Keys(args).derive(0);
    // Neither of these services could fit a member under the old equal split.
    args.local_devices = {{pos2gpu::cpu_device_id(0), "auto", {}}, {pos2gpu::cpu_device_id(1), "auto", {}}};
    args.dashboard_job = std::make_shared<DashboardJob>();
    Backend backend; backend.cpu_scheduler = std::make_shared<CodecScheduler>(1);
    backend.initialize(args, requested, 0, false);
    auto client = backend.acquire_pool_client();
    auto const hint = client->generation_capacity(18, 2);
    require(hint.slots == 1 && hint.batch >= 2 && !backend.fresh_group_slots(requested, 2),
        "private hints lost a fitting batch, advertised duplicate host capacity or offered a network-only manifest");
    std::array<std::array<uint8_t, 32>, 2> ids;
    for (uint16_t i = 0; i < ids.size(); ++i) ids[i] = requested.params.get_plot_params_for_index(i).get_plot_id().data();
    std::vector<std::unique_ptr<CompactMemberReceiver>> received;
    client->generate_packed(18, 2, ids, [&](PackedFragmentTile const& tile, uint64_t count)
    {
        if (!tile.boundaries.empty()) received.push_back(std::make_unique<CompactMemberReceiver>(18, size_t(count)));
        require(!received.empty(), "private packed batch omitted the member boundary"); received.back()->append(tile);
    });
    require(received.size() == ids.size(), "private packed batch lost a member");
    client->generate_packed(18, 2, ids, [](PackedFragmentTile const&, uint64_t) {});
    DashboardFrame frame; args.dashboard_job->fill(frame, false);
    for (auto const* selector : {"Local cpu0", "Local cpu1"})
        require(std::any_of(frame.rows[DashboardFrame::workers].begin(), frame.rows[DashboardFrame::workers].end(),
            [&](auto const& row) { return row.value.starts_with(selector) && row.value.find("2 delivered") != row.value.npos; }),
            "private selector did not receive its own displayed delivery count");
    worker_detail::CpuLimit limit(1);
    std::vector<std::vector<uint64_t>> reference;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        auto member = received[i]->finish(); auto view = member.view();
        std::vector<uint64_t> values;
        for (size_t c = 0; c < PlotGroupFile::getChunkCountForK(18); ++c)
            for (auto value : view.region(c)) values.push_back(value);
        reference.push_back(worker_original_reference(18, 2, ids[i]));
        require(values == reference.back(), "private shared grant changed CPU-reference output");
    }
    Directory scratch; auto const output = retained.value_or(scratch.path);
    require(fs::is_directory(output) && fs::is_empty(output), "private owner fixture requires an empty output directory");
    args.target = output / "private.gplot"; args.workers = 1; args.max_ram = uint64_t(512) << 20;
    args.capacity = uint64_t(64) << 20; args.max_members = args.batch = ids.size();
    Config cfg; cfg.capacity_bytes = args.capacity; cfg.window_bytes = args.window; cfg.codec_bytes = uint64_t(8) << 20;
    cfg.checkpoint_bytes = args.checkpoint_bytes; cfg.memory_limit_bytes = args.max_ram; cfg.workers = 1;
    auto const cpu = output / "reference.gplot";
    { Engine engine(cpu.string(), cfg, false, initial_identity(requested)); engine.confirm_identity();
      require(engine.append(Members(reference)).accepted, "reference failed to append private test group"); }
    std::unique_ptr<Backend> constructing;
    auto result = run_group(args, args.target, nullptr, requested, uint16_t(ids.size()), args.capacity, 0, constructing);
    require(result.members == ids.size() && !publication::pending(args.target) && contents(args.target) == contents(cpu),
        "complete private local publication differs from the independently generated CPU-reference group");
}

#if defined(__linux__) || defined(_WIN32)
std::string const multi_observer_job(64, 'e');
int dashboard_reader(char** argv)
{
    FarmingOwnerHealth owner; owner.lock_held = owner.local_process_proven = owner.heartbeat_current = true;
    owner.local_pid = std::stoull(argv[2]); owner.local_process_start = std::stoull(argv[3]);
    auto const marker = file_io::native_path(argv[4]);
    (void)DashboardFeed::read(multi_observer_job, owner);
    { std::ofstream ready(marker); require(bool(ready), "could not signal observer readiness"); }
    auto const until = std::chrono::steady_clock::now() + std::chrono::seconds(6);
    do
    {
        auto frame = DashboardFeed::read(multi_observer_job, owner);
        if (frame && frame->title == "Shared owner" && frame->rows[DashboardFrame::resources].size() == 1 &&
            frame->rows[DashboardFrame::resources].front().value == "resource visible") return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (std::chrono::steady_clock::now() < until);
    return 1;
}
int run_dashboard_reader(FarmingOwnerHealth const& owner, fs::path const& marker)
{
    auto const self = group_grow::calibration_process::executable();
#ifdef _WIN32
    auto const pid = std::to_wstring(owner.local_pid), start = std::to_wstring(owner.local_process_start);
    // _wspawnl passes its arguments through the Windows command-line parser.
    auto const exe = profile_detail::windows_argument(self.native());
    auto const ready = profile_detail::windows_argument(marker.native());
    return int(::_wspawnl(_P_WAIT, self.c_str(), exe.c_str(), L"--dashboard-reader", pid.c_str(),
        start.c_str(), ready.c_str(), static_cast<wchar_t const*>(nullptr)));
#else
    std::array<std::string, 5> args{self.string(), "--dashboard-reader", std::to_string(owner.local_pid),
        std::to_string(owner.local_process_start), marker.string()};
    std::array<char*, 6> command{};
    for (size_t i = 0; i < args.size(); ++i) command[i] = args[i].data();
    pid_t child = -1;
    if (::posix_spawn(&child, self.c_str(), nullptr, nullptr, command.data(), environ)) return -1;
    int status = 0;
    while (::waitpid(child, &status, 0) < 0) if (errno != EINTR) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}
#endif
void owner_dashboard_feed()
{
    DashboardJob progress;
    auto gpu_status = [&](std::string const& cause, bool waiting) {
        progress.local_gpu_health("gpu0", cause, waiting);
        DashboardFrame frame; progress.fill(frame, false);
        auto const& workers = frame.rows[DashboardFrame::workers];
        auto const row = std::find_if(workers.begin(), workers.end(), [](auto const& entry)
            { return entry.value.starts_with("Local gpu0   "); });
        return row == workers.end() ? DashboardFrame::Row{} : *row;
    };
    auto waiting = gpu_status({}, true);
    require(waiting.value == "Local gpu0   Waiting for GPU memory" && waiting.detail.empty(),
        "private GPU memory pressure was missing from the worker view");
    auto recovering = gpu_status("device lost", false);
    require(recovering.value == "Local gpu0   GPU recovering" && recovering.detail == "device lost",
        "private GPU recovery lost its driver cause after waiting for memory");
    require(gpu_status({}, false).value.empty(), "available GPU retained its previous worker status");
    progress.roster(18, 2, 1, true, true, true, {{"id", "process", "helper", "GPU", "endpoint", 1}});
    progress.failed_delivery("id", "process", "insufficient memory");
    DashboardFrame snapshot;
    progress.fill(snapshot, true);
    auto const& worker_rows = snapshot.rows[DashboardFrame::workers];
    require(std::any_of(worker_rows.begin(), worker_rows.end(), [](auto const& row)
        { return row.value.find("helper   1 failed assignment") != row.value.npos && row.detail == "insufficient memory"; }) &&
        std::none_of(worker_rows.begin(), worker_rows.end(), [](auto const& row) { return row.value.find(" delivered") != row.value.npos; }),
        "an unsuccessful helper did not show its failure separately from delivered work");
    require(progress.delivery("id", "process", 1, 100, 10, 4, 2, 3, 0) == "helper",
        "a failed assignment consumed the helper's first validated delivery announcement");
#if defined(__linux__) || defined(_WIN32)
    std::string const job(64, 'f');
    DashboardFeed feed(job);
    require(!feed.wants_frame(), "inactive dashboard sampled a frame without a reader");
    FarmingOwnerHealth owner; owner.lock_held = owner.local_process_proven = owner.heartbeat_current = true;
#ifdef _WIN32
    owner.local_pid = ::GetCurrentProcessId();
#else
    owner.local_pid = uint64_t(::getpid());
#endif
    owner.local_process_start = std::stoull(farming_owner_detail::process_start(owner.local_pid));
    require(!DashboardFeed::read(job, owner) && feed.wants_frame(), "reader did not signal a new local attachment");
    DashboardFrame sent; sent.title = "Local owner"; sent.summary = "Publishing";
    sent.add(DashboardFrame::workers, "ready", "registered"); sent.add(DashboardFrame::decisions, "packed", "RAM admitted");
    for (size_t row = 0; row < 96; ++row)
        sent.add(DashboardFrame::resources, "HDD " + std::to_string(row), "Read and write service on the selected farm volume");
    feed.publish(sent);
    uint64_t age = UINT64_MAX;
    auto got = DashboardFeed::read(job, owner, &age);
    require(got && got->title == sent.title && got->summary == sent.summary && age < 5000 &&
        got->rows[DashboardFrame::workers].at(0).detail == "registered" &&
        got->rows[DashboardFrame::decisions].at(0).value == "packed" &&
        got->rows[DashboardFrame::resources].size() == 96 && got->rows[DashboardFrame::resources].back().value == "HDD 95" &&
        got->rows[DashboardFrame::resources].back().detail == sent.rows[DashboardFrame::resources].back().detail,
        "attached local snapshot lost its job, age, data or full 96-volume resource inventory");
    for (size_t row = 96; row < 310; ++row) sent.add(DashboardFrame::resources, "HDD " + std::to_string(row), std::string(900, 'x'));
    sent.add(DashboardFrame::farm, "one late lookup");
    feed.publish(sent); got = DashboardFeed::read(job, owner);
    require(got && got->rows[DashboardFrame::farm].at(0).value == "one late lookup",
        "busy resources crowded out another dashboard view");
    auto const& rows = got->rows[DashboardFrame::resources];
    require(rows.size() > 96 && rows.size() < 256 && rows.back().value ==
        std::to_string(310 - (rows.size() - 1)) + " additional rows on the plotting host",
        "bounded packed snapshot lost the total rows omitted by its source or transfer");
    require(!DashboardFeed::read(std::string(64, 'a'), owner), "snapshot from another job was attached");
    auto stale = owner; ++stale.local_process_start;
    require(!DashboardFeed::read(job, stale), "reused owner PID supplied another generation's snapshot");
    stale = owner; stale.heartbeat_current = false;
    require(!DashboardFeed::read(job, stale), "stale owner heartbeat supplied a live snapshot");
    stale = owner; stale.local_process_proven = false;
    require(!DashboardFeed::read(job, stale), "remote owner was treated as a proven local dashboard");

    Directory scratch; DashboardFeed shared(multi_observer_job);
    auto const first_path = scratch.path / "first", second_path = scratch.path / "second";
    auto first = std::async(std::launch::async, run_dashboard_reader, owner, first_path);
    auto second = std::async(std::launch::async, run_dashboard_reader, owner, second_path);
    auto const until = std::chrono::steady_clock::now() + std::chrono::seconds(6);
    while ((!fs::exists(first_path) || !fs::exists(second_path) || !shared.wants_frame()) &&
        std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    require(fs::exists(first_path) && fs::exists(second_path) && shared.wants_frame(),
        "separate local observers could not attach to their live job");
    DashboardFrame common; common.title = "Shared owner"; common.add(DashboardFrame::resources, "resource visible");
    // Readers or writer may deliberately skip a busy frame.
    while (std::chrono::steady_clock::now() < until)
    {
        shared.publish(common);
        if (first.wait_for(std::chrono::milliseconds(20)) == std::future_status::ready &&
            second.wait_for(std::chrono::milliseconds(20)) == std::future_status::ready) break;
    }
    require(first.get() == 0 && second.get() == 0, "separate observers did not both receive the owner's frame");
#ifdef __linux__
    require(!shared.wants_frame(), "owner kept sampling after both monitoring processes exited");
#endif
#endif
}
}

int main(int argc, char** argv)
try
{
#if defined(__linux__) || defined(_WIN32)
    if (argc == 5 && std::string_view(argv[1]) == "--dashboard-reader") return dashboard_reader(argv);
#endif
    if (argc == 3 && std::string_view(argv[1]) == "--codec-publication")
    { accepted_remote_publication(fs::path(argv[2])); return 0; }
    if (argc == 3 && std::string_view(argv[1]) == "--private-publication")
    { private_local_delegation(fs::path(argv[2])); return 0; }
    if (argc == 2 && std::string_view(argv[1]) == "--private-local") { private_local_delegation(); owner_dashboard_feed(); return 0; }
    require(argc == 1, "usage: grow_pool_owner_test [--private-local | --private-publication EMPTY_DIRECTORY | --codec-publication EMPTY_DIRECTORY]");
    private_local_delegation();
    owner_dashboard_feed();
    repartition_and_global_admission();
    late_packed_capacity();
    owner_broker_restart();
    accepted_remote_publication();
    std::cout << "owner scratch recovery, global admission, packed prefetch, broker restart and remote publication passed\n";
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
