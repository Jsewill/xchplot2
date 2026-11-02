#include "host/GroupRemoteCodec.hpp"
#include "compute_pool_fixture.hpp"
#include <iostream>

using namespace group_grow;
using group_tools::require;

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

// An authenticated replacement worker sends a valid but wrong canonical
// result, or a real application error, after lease acceptance.
struct RejectingWorker
{
    pool::Listener listener;
    pool::BrokerClient broker;
    std::atomic<bool> stop{false}, served{false};
    std::vector<std::vector<uint8_t>> wrong, initial;
    std::atomic<unsigned> calls{0};
    std::exception_ptr error;
    std::thread thread;
    RejectingWorker(pool::Endpoint endpoint, pool::TlsOptions tls, std::vector<std::vector<uint8_t>> payload = {},
                    std::vector<std::vector<uint8_t>> correct = {})
        : listener({"127.0.0.1", 0}, tls), broker(endpoint, tls), wrong(std::move(payload)), initial(std::move(correct))
    {
        pool::WorkerOffer offer;
        offer.instance = pool::random_token(); offer.compatibility = worker_compatibility();
        offer.endpoint = listener.endpoint(); offer.capacity = {1, 1, uint64_t(128) << 20, 0};
        offer.capabilities = pool::capability(pool::Kind::Pack); offer.min_k = 18; offer.max_k = 28;
        broker.register_worker(offer);
        thread = std::thread([this]
        {
            try
            {
                while (!stop.load())
                {
                    auto channel = listener.accept();
                    if (!channel) continue;
                    auto work = channel.receive();
                    require(work.type == pool::Message::Work, "test replacement worker received no task");
                    pool::Reader input(work.payload); auto lease = pool::read_lease(input);
                    require(broker.begin(lease, channel.peer_identity()), "test replacement lease was not accepted");
                    (void)input.u32(); // Task deadline.
                    require(input.u8() == 3, "test replacement expected a codec operation");
                    (void)input.u8(); (void)input.u8();
                    uint64_t remaining = input.u64(); (void)input.u64(); input.finish();
                    while (remaining)
                    {
                        auto frame = channel.receive();
                        require(frame.type == pool::Message::Data && !frame.payload.empty() &&
                            frame.payload.size() <= remaining, "test replacement received malformed input");
                        remaining -= frame.payload.size();
                    }
                    auto const& response = !calls && !initial.empty() ? initial : wrong;
                    ++calls;
                    if (response.empty())
                    {
                        pool::Writer error; error.u8(0); error.string("deliberate bounded codec failure");
                        channel.send(pool::Message::Error, error.data());
                    }
                    else
                    {
                        // Encoded data frames carry their chunk position, byte
                        // offset and a bounded byte string, as in the worker.
                        for (size_t i = 0; i < response.size(); ++i)
                        {
                            pool::Writer header; header.u32(uint32_t(i)); header.u64(response[i].size());
                            channel.send(pool::Message::Result, header.data());
                            pool::Writer output; output.u8(5); output.u32(uint32_t(i)); output.u64(0); output.bytes(response[i]);
                            channel.send(pool::Message::Data, output.data());
                        }
                        channel.send(pool::Message::End, {});
                    }
                    broker.finish(lease);
                    if (initial.empty() || calls > 1) { served.store(true); return; }
                }
            }
            catch (...) { error = std::current_exception(); }
        });
    }
    ~RejectingWorker() { stop.store(true); if (thread.joinable()) thread.join(); }
    void finish()
    {
        stop.store(true); thread.join();
        if (error) std::rethrow_exception(error);
    }
};

void bounded_coalescing()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "coalescing-ca");
    auto identity = pool_test::issue(directory.path, "coalescing-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 2000; tls.max_frame_bytes = 4096;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    options.tls.max_frame_bytes = 65536;
    RunningBroker broker(options);
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0};
    offered.tls = tls; offered.capacity = {1, 1, uint64_t(256) << 20, 0};
    offered.max_task_bytes = uint64_t(1) << 20;
    WorkerService service(offered); service.start();
    constexpr uint8_t k = 18;
    uint64_t const range = uint64_t(1) << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<std::vector<uint64_t>> raw(129);
    std::vector<FragmentRegion> regions; regions.reserve(raw.size());
    std::vector<ChunkInput> tasks; tasks.reserve(raw.size());
    std::vector<uint8_t> expected;
    std::vector<uint64_t> sizes(raw.size());
    ChunkCodec codec(k, uint64_t(4) << 20);
    for (size_t i = 0; i < raw.size(); ++i)
    {
        for (uint64_t n = 0; n < 64; ++n) raw[i].push_back(i * range + n * (range / 64) + 17);
        regions.emplace_back(std::span<uint64_t const>(raw[i]));
        auto encoded = codec.encode(i, {}, 0, {&regions.back(), 1});
        expected.insert(expected.end(), encoded.begin(), encoded.end());
        tasks.push_back({uint64_t(i), {}, 0, {&regions.back(), 1}, {}});
    }
    for (uint64_t limit : {uint64_t(1) << 20, uint64_t(16) << 10})
    {
        auto scheduler = std::make_shared<CodecScheduler>(1);
        WorkerClientConfig client_options; client_options.broker = offered.broker; client_options.tls = tls;
        client_options.owner_id = "coalescing-" + std::to_string(limit);
        client_options.cpu_scheduler = scheduler; client_options.max_task_bytes = limit;
        auto client = std::make_shared<WorkerClient>(client_options);
        PoolRemoteEncoder remote(uint64_t(64) << 20, tls.max_frame_bytes,
            offered.max_task_bytes, [client] { return client; }, scheduler);
        ParallelEncoder local(k, 1, uint64_t(8) << 20, scheduler);
        std::promise<void> submitted;
        std::future<void> calibration;
        {
            CodecScheduler::Lease pressure(scheduler.get());
            calibration = std::async(std::launch::async, [&]
            {
                auto const begin = std::chrono::steady_clock::now();
                submitted.set_value();
                auto blocks = local.encode(tasks, sizes);
                remote.observe_local(k, tasks, sizes, blocks, uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - begin).count()));
            });
            submitted.get_future().wait();
            require(calibration.wait_for(std::chrono::milliseconds(200)) != std::future_status::ready,
                "coalescing fixture escaped shared CPU pressure");
        }
        calibration.get();
        // Each raw chunk has 18 fixed bytes, a 4-byte raw count and
        // 64 uint64 values; each request starts with one chunk count.
        uint64_t const per_request = std::min<uint64_t>(client->max_window_chunks(), (limit - 4) / (22 + 64 * 8));
        uint64_t const requests = (tasks.size() + per_request - 1) / per_request;
        require(remote.counters.calibration_windows == 1 && client->stats().completed_codec == requests,
            "remote calibration did not coalesce within count and serialized byte limits");
        auto output = remote.encode(k, tasks, true);
        require(output && output->bytes == expected && output->sizes == sizes &&
            client->stats().completed_codec == requests * 2 && remote.counters.accepted_windows == 1,
            "bounded coalesced remote window changed canonical content or request count");
    }
}

void integration(uint8_t k)
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "codec-ca");
    auto identity = pool_test::issue(directory.path, "codec-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 2000; tls.max_frame_bytes = 4096;
    pool::BrokerOptions broker_options; broker_options.listen = {"127.0.0.1", 0}; broker_options.tls = tls;
    broker_options.tls.max_frame_bytes = 65536;
    broker_options.worker_ms = 2000;
    RunningBroker broker(broker_options);
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0};
    offered.tls = tls; offered.capacity = {2, 2, uint64_t(256) << 20, 0}; offered.max_task_bytes = uint64_t(1) << 20;
    WorkerService service(offered); service.start();
    auto scheduler = std::make_shared<CodecScheduler>(1);
    WorkerClientConfig client_options; client_options.broker = offered.broker; client_options.tls = tls;
    client_options.owner_id = "remote-codec-owner"; client_options.task_timeout_ms = 3000;
    client_options.max_task_bytes = offered.max_task_bytes; client_options.cpu_scheduler = scheduler;
    std::mutex selection_mutex; std::condition_variable selection_ready;
    bool parallel = false; unsigned waiting = 0;
    client_options.selected = [&](WorkerTaskObservation const&)
    {
        std::unique_lock lock(selection_mutex);
        if (!parallel) return;
        ++waiting; selection_ready.notify_all();
        require(selection_ready.wait_for(lock, std::chrono::seconds(1), [&] { return waiting == 2; }),
            "codec policy serialized independent admitted windows");
    };
    auto client = std::make_shared<WorkerClient>(client_options);
    auto remote = std::make_shared<PoolRemoteEncoder>(uint64_t(64) << 20, tls.max_frame_bytes,
        client_options.max_task_bytes, [client] { return client; }, scheduler);

    uint64_t const range = uint64_t(1) << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<std::vector<uint64_t>> raw(4);
    std::vector<std::vector<uint8_t>> old(4), canonical(4), wrong(4);
    std::vector<std::array<EncodedChunk, 2>> contributions(4);
    std::vector<FragmentRegion> regions; regions.reserve(raw.size());
    std::vector<ChunkInput> tasks; tasks.reserve(raw.size());
    ChunkCodec codec(k, uint64_t(4) << 20);
    for (size_t i = 0; i < raw.size(); ++i)
    {
        for (uint64_t value : {1, 5, 13, 17}) raw[i].push_back(uint64_t(i) * range + value);
        for (uint64_t n = 1; n < 64; ++n) raw[i].push_back(uint64_t(i) * range + n * (range / 64) + 17);
        regions.emplace_back(std::span<uint64_t const>(raw[i]));
        auto encoded = codec.encode(i, {}, 0, {&regions.back(), 1}); old[i].assign(encoded.begin(), encoded.end());
        contributions[i] = {{{old[i], 1}, {old[i], 1}}};
        auto joined = codec.join(i, contributions[i], {&regions.back(), 1}); canonical[i].assign(joined.begin(), joined.end());
        auto altered = raw[i]; altered[1] += 4;
        FragmentRegion const different{std::span<uint64_t const>(altered)};
        auto changed = codec.join(i, contributions[i], {&different, 1}); wrong[i].assign(changed.begin(), changed.end());
        auto const correct = codec.inspect(i, canonical[i], 3), collision = codec.inspect(i, wrong[i], 3);
        require(canonical[i] != wrong[i] && canonical[i].size() == wrong[i].size() &&
            correct.first == collision.first && correct.summary == collision.summary,
            "wrong-content fixture does not collide with the old summary-only validator");
        tasks.push_back({uint64_t(i), {}, 0, {&regions.back(), 1}, contributions[i]});
    }
    std::vector<uint8_t> expected;
    for (auto const& chunk : canonical) expected.insert(expected.end(), chunk.begin(), chunk.end());
    ParallelEncoder encoder(k, 1, uint64_t(8) << 20, scheduler, remote);
    std::array<uint64_t, 4> sizes{};
    // Actual local queue pressure supplies the calibration observation. There
    // is no production speed override or synthetic remote timing result.
    auto calibrate = [&](PoolRemoteEncoder& candidate)
    {
        ParallelEncoder local_encoder(k, 1, uint64_t(8) << 20, scheduler);
        std::future<void> calibration;
        std::promise<void> submitted;
        {
            CodecScheduler::Lease pressure(scheduler.get());
            calibration = std::async(std::launch::async, [&]
            {
                // Start the real local sample before notifying the holder.
                // Thread descheduling after this signal must not move the
                // measured interval past the shared CPU queue pressure.
                auto const begin = std::chrono::steady_clock::now();
                submitted.set_value();
                auto const blocks = local_encoder.encode(tasks, sizes);
                candidate.observe_local(k, tasks, sizes, blocks, uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - begin).count()));
            });
            submitted.get_future().wait();
            require(calibration.wait_for(std::chrono::milliseconds(200)) != std::future_status::ready,
                "local codec escaped actual shared CPU pressure");
        }
        calibration.get();
    };
    calibrate(*remote);
    require(remote->counters.calibration_windows.load() == 1, "real remote codec did not calibrate canonical bytes");
    auto const first = encoder.encode(tasks, sizes, true);
    require(first.size() == 1 && std::ranges::equal(first.front().bytes, expected) &&
        remote->counters.accepted_windows.load() > 0 && !encoder.cpu_time().measured,
        "real remote codec was not accepted after measured queue savings");
    auto held = first.front().bytes;
    encoder.encode(tasks, sizes, true);
    require(std::ranges::equal(held, expected), "real network output reused a retained previous window");
    for (size_t i = 0; i < sizes.size(); ++i) require(sizes[i] == canonical[i].size(), "network canonical chunk length changed");

    {
        { std::lock_guard lock(selection_mutex); parallel = true; }
        auto first_window = std::async(std::launch::async, [&] { return remote->encode(k, tasks, false); });
        auto second_window = std::async(std::launch::async, [&] { return remote->encode(k, tasks, false); });
        auto a = first_window.get(), b = second_window.get();
        { std::lock_guard lock(selection_mutex); parallel = false; }
        require(a && b && a->bytes == expected && b->bytes == expected && waiting == 2,
            "independent remote codec windows did not run concurrently");
    }

    unsigned initialized = 0;
    auto bounded = std::make_shared<PoolRemoteEncoder>(1, tls.max_frame_bytes, client_options.max_task_bytes,
        [&] { ++initialized; return client; }, scheduler);
    ParallelEncoder local(k, 1, uint64_t(8) << 20, scheduler, bounded);
    local.encode(tasks, sizes);
    require(initialized == 0 && bounded->counters.accepted_windows.load() == 0,
        "remote codec dispatched before receiver memory admission");

    {
        auto changing = std::make_shared<PoolRemoteEncoder>(uint64_t(64) << 20, tls.max_frame_bytes,
            client_options.max_task_bytes, [client] { return client; }, scheduler);
        ParallelEncoder adaptive(k, 1, uint64_t(8) << 20, scheduler, changing);
        calibrate(*changing);
        for (unsigned window = 0; window < 66; ++window)
        {
            auto blocks = adaptive.encode(tasks, sizes);
            std::vector<uint8_t> bytes;
            for (auto const& block : blocks) bytes.insert(bytes.end(), block.bytes.begin(), block.bytes.end());
            require(bytes == expected, "periodic local comparison changed canonical bytes");
        }
        require(changing->counters.accepted_windows.load() > 0 && changing->counters.calibration_attempts.load() >= 2 && changing->counters.fallback_windows.load() > 0,
            "remote winner did not remeasure local cost after CPU pressure ended");
    }

    service.stop();
    for (bool malformed : {true, false})
    {
        // A process replacement using the same certificate must wait for the
        // previous registration to expire, just as a restarted worker does.
        std::this_thread::sleep_for(std::chrono::milliseconds(broker_options.worker_ms + 100));
        RejectingWorker rejected(offered.broker, tls, malformed ? wrong : std::vector<std::vector<uint8_t>>{}, canonical);
        require(!remote->encode(k, tasks, false) && !rejected.calls.load(),
            "replacement worker inherited another process's calibration");
        auto checked = std::make_shared<PoolRemoteEncoder>(uint64_t(64) << 20, tls.max_frame_bytes,
            client_options.max_task_bytes, [client] { return client; }, scheduler);
        ParallelEncoder candidate(k, 1, uint64_t(8) << 20, scheduler, checked);
        calibrate(*checked);
        if (checked->counters.calibration_windows.load() != 1) rejected.finish();
        require(checked->counters.calibration_windows.load() == 1,
            "replacement worker did not calibrate first (k=" + std::to_string(k) + ", calls=" + std::to_string(rejected.calls.load()) + ")");
        auto const fallback = candidate.encode(tasks, sizes, true);
        rejected.finish();
        require(!fallback.empty() && rejected.served.load() && rejected.calls == 2 && checked->counters.fallback_windows.load() > 0 &&
            checked->counters.accepted_windows.load() == 0,
            "authenticated codec failure did not use the local path");
        std::vector<uint8_t> fallback_bytes;
        for (auto const& block : fallback) fallback_bytes.insert(fallback_bytes.end(), block.bytes.begin(), block.bytes.end());
        require(fallback_bytes == expected, "remote failure changed local fallback bytes");
    }
    {
        pool::BrokerOptions empty_options = broker_options;
        RunningBroker empty(empty_options);
        auto unavailable_options = client_options; unavailable_options.broker = empty.server.endpoint();
        unavailable_options.task_timeout_ms = 3600000;
        auto unavailable = std::make_shared<WorkerClient>(unavailable_options);
        auto optional = std::make_shared<PoolRemoteEncoder>(uint64_t(64) << 20, tls.max_frame_bytes,
            unavailable_options.max_task_bytes, [unavailable] { return unavailable; }, scheduler);
        auto const begin = std::chrono::steady_clock::now();
        calibrate(*optional);
        require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1) &&
            optional->counters.calibration_attempts.load() == 1 && optional->counters.calibration_windows.load() == 0,
            "optional codec probe waited for the generation timeout after local bytes were ready");
    }
    auto const stats = client->stats();
    require(stats.completed_codec > 0 && stats.input_bytes > 0 && stats.output_bytes > 0 && stats.failed_tasks > 0,
        "real codec completion/error wire accounting is incomplete");
}
} // namespace

int main() try
{
    bounded_coalescing();
    for (uint8_t k : {18, 22, 28}) integration(k);
    std::cout << "network codec calibration, bounded admission, retained output and fallback passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
