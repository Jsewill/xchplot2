// Like grow-merge-test, exercise the actual controller producer in this TU.
#include "host/GroupPlotter.cpp"
#include "compute_pool_fixture.hpp"
#include "host/ComputeThreads.hpp"

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
    std::shared_ptr<CodecScheduler> scheduler;
    std::function<void()> generation;
    std::future<void> pending;
    bool queued = false;
    PrefetchEncoder(std::shared_ptr<PoolRemoteEncoder> provider,
                    std::shared_ptr<CodecScheduler> cpu, std::function<void()> work)
        : remote(std::move(provider)), scheduler(std::move(cpu)), generation(std::move(work)) {}
    ~PrefetchEncoder() { if (pending.valid()) pending.wait(); }
    std::optional<RemoteEncodedWindow> encode(uint8_t k, std::span<ChunkInput const> chunks, bool summaries) override
    {
        auto result = remote->encode(k, chunks, summaries);
        if (!result && !queued && !chunks.empty() && !chunks.front().contributions.empty())
        { pending = scheduler->submit(generation); queued = true; }
        return result;
    }
    void observe_local(uint8_t k, std::span<ChunkInput const> chunks,
        std::span<uint64_t const> sizes, std::span<EncodedBlock const> blocks, uint64_t elapsed) override
    { remote->observe_local(k, chunks, sizes, blocks, elapsed); }
    void join() { if (pending.valid()) pending.get(); }
};

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
    auto encoder = std::make_shared<PrefetchEncoder>(backend->pool_remote, scheduler, [&]
    {
        auto const id = requested.params.get_plot_params_for_index(uint16_t(args.max_members)).get_plot_id().data();
        WorkerCancellation cancel; cancel.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        pressure.generate_local(18, 2, id, [](auto const&, uint64_t) {}, cancel, pool::Kind::GenerateCpu);
        ++completed_prefetch;
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
}

void repartition_and_global_admission()
{
    using namespace grow_controller_test;
    Directory directory;
    pool_test::Directory certificates;
    auto ca = pool_test::issue(certificates.path, "owner-ca");
    auto certificate = pool_test::issue(certificates.path, "owner-host", &ca);
    auto tls = pool_test::tls(ca, certificate); tls.timeout_ms = 2000; tls.max_frame_bytes = 4096;
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
    WorkerService worker(offered); worker.start();
    Arguments args; args.backend = "cpu"; args.workers = 2; args.pool_inflight = 1;
    args.max_ram = uint64_t(1) << 30; args.window = uint64_t(1) << 20;
    args.pool_frame_bytes = tls.max_frame_bytes; args.pool_timeout_ms = 10000; args.pool_socket_timeout_ms = tls.timeout_ms;
    args.pool_broker = "127.0.0.1:" + std::to_string(offered.broker.port);
    args.pool_owner = "owner-repartition"; args.pool_ca = tls.ca_file; args.pool_cert = tls.cert_file; args.pool_key = tls.key_file;
    Backend backend; backend.cpu_scheduler = std::make_shared<CodecScheduler>(2);
    backend.initialize(args, requested, 0, false);
    auto client = backend.acquire_pool_client();
    // Warm the real measured policy. A subsequent grouped request is legal
    // under advertised generation RAM but cannot fit this worker's codec cap.
    for (unsigned attempt = 0; attempt < 4 && client->suggested_batch_members(18, 2, 4) == 1; ++attempt)
        client->contributions(18, 2, {&ids.front(), 1}, [](auto, auto, auto) {});
    require(client->suggested_batch_members(18, 2, 4) > 1, "owner fixture did not learn contribution batching");
    auto const before = client->stats();

    Config cfg; cfg.capacity_bytes = uint64_t(64) << 20; cfg.window_bytes = args.window;
    cfg.codec_bytes = uint64_t(8) << 20; cfg.checkpoint_bytes = 65536;
    cfg.memory_limit_bytes = args.max_ram; cfg.io_overlap = false; cfg.workers = 1; cfg.cpu_scheduler = backend.cpu_scheduler;
    auto const initial = initial_identity(requested);
    auto const reference = directory.path / "reference.gplot";
    { Engine engine(reference.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(Members(raw)).accepted, "owner canonical reference did not fit"); }
    std::array<std::unique_ptr<ExternalEngine>, 2> engines;
    std::array<std::unique_ptr<ExternalEngine::ContributionInput>, 2> inputs;
    std::array<fs::path, 2> paths;
    for (size_t destination = 0; destination < engines.size(); ++destination)
    {
        auto scratch = directory.path / ("scratch-" + std::to_string(destination)); fs::create_directory(scratch);
        paths[destination] = directory.path / ("output-" + std::to_string(destination) + ".gplot");
        ExternalConfig external{{{scratch.string(), uint64_t(256) << 20}}, 65536, 2};
        engines[destination] = std::make_unique<ExternalEngine>(paths[destination].string(), cfg, external, false, initial);
        engines[destination]->confirm_identity();
        inputs[destination] = engines[destination]->prepare_contributions(uint16_t(raw.size()),
            fragment_bound(requested.params) / sizeof(uint64_t));
        require(bool(inputs[destination]), "owner contribution reservation did not fit");
    }
    std::array<std::future<size_t>, 2> receiving;
    for (size_t destination = 0; destination < receiving.size(); ++destination)
        receiving[destination] = std::async(std::launch::async, [&, destination]
        {
            Producer producer{requested, nullptr, backend, args, args.max_ram, uint64_t(64) << 20, 0};
            producer.defer_pool_history = true;
            return producer.fill_contributions(*inputs[destination], 0, 0,
                producer.contribution_bytes(cfg, uint16_t(raw.size())), "owner scratch fixture member ");
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
        require(inputs[destination]->sealed() && engines[destination]->append(*inputs[destination]).accepted,
            "owner repartition did not seal an appendable exact membership");
        inputs[destination].reset(); engines[destination]->finish();
        require(contents(paths[destination]) == contents(reference), "owner scratch repartition changed canonical publication");
    }
    // Network receivers already own compact words; they do not pass through
    // CompactMemberBuilder, which alone populates the producer prefix cache.
    // Transfer a real prefetched group and then exercise the single-owner
    // append/prefetch loop that previously truncated an empty cache.
    auto const prefix_reference = directory.path / "packed-reference.gplot";
    { Engine engine(prefix_reference.string(), cfg, false, initial); engine.confirm_identity();
      require(engine.append(Members(raw).subspan(0, 3)).accepted, "packed owner reference did not fit"); }
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
    uint64_t const raw_retries = backend.pool_retries.load();
    broker.reset();
    auto raw = std::async(std::launch::async, [&] { return backend.generate_pool(requested, 0, args); });
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    bool const waiting_raw = raw.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    broker = std::make_unique<RunningBroker>(options);
    require(waiting_raw, "raw owner exhausted immediate retries before broker restart");
    require(raw.wait_for(std::chrono::seconds(8)) == std::future_status::ready,
        "raw owner did not recover within its original task deadline");
    std::vector<Member> member; member.push_back(raw.get());
    require(backend.pool_retries.load() > raw_retries + 1, "raw owner did not survive the old retry limit");
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
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    bool const waiting_contribution = contribution.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    broker = std::make_unique<RunningBroker>(options);
    require(waiting_contribution, "contribution owner exhausted immediate retries before broker restart");
    require(contribution.wait_for(std::chrono::seconds(8)) == std::future_status::ready && contribution.get() == 2,
        "contribution owner did not recover its complete membership");
    require(backend.pool_retries.load() > contribution_retries + 1, "contribution owner did not survive the old retry limit");
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
}

int main(int argc, char** argv)
try
{
    if (argc == 3 && std::string_view(argv[1]) == "--codec-publication")
    { accepted_remote_publication(fs::path(argv[2])); return 0; }
    require(argc == 1, "usage: grow_pool_owner_test [--codec-publication EMPTY_DIRECTORY]");
    repartition_and_global_admission();
    owner_broker_restart();
    accepted_remote_publication();
    std::cout << "owner scratch recovery, global admission, packed prefetch, broker restart and remote publication passed\n";
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
