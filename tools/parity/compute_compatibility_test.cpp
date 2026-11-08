// Independent executable contracts, following the construction cross-build
// fixture's fork/exec boundary. No source identity participates in scheduling.
#include "host/ComputeWorker.hpp"
#ifndef GROW_COMPUTE_FIXTURE_WORKER
#include "host/GroupCodec.hpp"
#endif
#include "host/ComputeIdentity.hpp"
#include "compute_pool_fixture.hpp"
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

using namespace group_grow;
using pool_test::require;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

#ifdef GROW_COMPUTE_FIXTURE_WORKER
namespace
{
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
}
int main(int argc, char** argv)
{
    try
    {
        require(argc == 8, "cross-build worker expects ready file, broker and TLS enrollment");
        WorkerConfig config; config.broker = pool::parse_endpoint(argv[2]); config.broker.server_name = argv[7];
        config.listen = {"127.0.0.1", 0, argv[7]}; config.server_name = argv[7];
        config.tls = {argv[3], argv[4], argv[5], 1000, 65536}; config.tls.membership_directory = argv[6];
        config.capacity = {1, 1, uint64_t(512) << 20, 0}; config.generation_threads = 1;
        config.max_task_bytes = uint64_t(1) << 20; config.task_timeout_ms = 10000;
        std::signal(SIGTERM, stop); std::signal(SIGINT, stop);
        WorkerService service(config); service.start();
        fs::path const ready(argv[1]); auto partial = ready; partial += ".partial";
        { std::ofstream output(partial); output << service.endpoint().port << '\n'; require(bool(output), "cannot publish worker route"); }
        fs::rename(partial, ready);
        while (!stopped) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        service.stop(); return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
#else
std::vector<uint64_t> worker_original_reference(uint8_t, uint8_t, std::array<uint8_t, 32> const&);
namespace
{
template <typename Predicate> void eventually(Predicate&& predicate, char const* message)
{
    auto const until = Clock::now() + std::chrono::seconds(8);
    do { if (predicate()) return; std::this_thread::sleep_for(std::chrono::milliseconds(10)); } while (Clock::now() < until);
    throw std::runtime_error(message);
}
struct RunningBroker
{
    pool::BrokerServer server;
    std::atomic<bool> stop{false};
    std::thread thread;
    explicit RunningBroker(pool::BrokerOptions options) : server(std::move(options)), thread([this] { server.serve(stop); }) {}
    ~RunningBroker() { stop.store(true); thread.join(); }
};
struct Process
{
    pid_t pid = -1;
    pool::Endpoint endpoint;
    Process(fs::path const& binary, fs::path const& ready, pool::Endpoint broker, pool::TlsOptions const& tls)
    {
        fs::remove(ready);
        std::vector<std::string> arguments{binary.string(), ready.string(), broker.host + ":" + std::to_string(broker.port),
            tls.ca_file, tls.cert_file, tls.key_file, tls.membership_directory, broker.server_name};
        std::vector<char*> command;
        for (auto& argument : arguments) command.push_back(argument.data()); command.push_back(nullptr);
        pid = ::fork(); require(pid >= 0, "cannot fork compute compatibility worker");
        if (!pid) { ::execv(binary.c_str(), command.data()); std::_Exit(127); }
        try
        {
            eventually([&]
            {
                int status = 0;
                if (::waitpid(pid, &status, WNOHANG) == pid) { pid = -1; throw std::runtime_error("cross-build worker exited during startup"); }
                return fs::is_regular_file(ready);
            }, "cross-build worker did not publish its route");
            unsigned port = 0; std::ifstream input(ready); input >> port;
            require(bool(input) && port && port <= UINT16_MAX, "invalid cross-build worker route");
            endpoint = {"127.0.0.1", uint16_t(port), broker.server_name};
        }
        catch (...) { terminate(true); throw; }
    }
    ~Process() { terminate(true); }
    void terminate(bool crash = false)
    {
        if (pid < 0) return;
        ::kill(pid, crash ? SIGKILL : SIGTERM); int status = 0;
        if (crash) { (void)::waitpid(pid, &status, 0); pid = -1; return; }
        eventually([&] { return ::waitpid(pid, &status, WNOHANG) == pid; }, "cross-build worker did not stop");
        pid = -1; require(WIFEXITED(status) && !WEXITSTATUS(status), "cross-build worker failed while stopping");
    }
};
void incompatible(pool::BrokerClient& broker, pool::Kind kind)
{
    auto session = broker.open_owner("incompatible-" + pool::random_token());
    pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility(kind); task.kind = kind;
    task.k = 18; task.strength = 2; task.resources = {1, 1, uint64_t(64) << 20, 0};
    if (kind == pool::Kind::Pack)
    { task.work_units = pool::codec_window_header_bytes + pool::codec_chunk_header_bytes; task.chunk_count = 1; task.max_output_bytes = 4096; }
    require(!broker.acquire(session, task), "incompatible compute contract acquired a worker lease");
    pool::StatusRequest query; query.section = pool::StatusSection::Jobs;
    auto page = broker.status(query);
    require(std::any_of(page.job_records.begin(), page.job_records.end(), [](auto const& job)
        { return job.state == pool::JobState::Queued && job.waiting == pool::WaitReason::Incompatible; }),
        "incompatible task did not retain a clear closed admission reason");
    broker.cancel(session, task.task_id, task.attempt); broker.close_owner(session);
}
void codec(WorkerClient& client)
{
    std::array<uint64_t, 4> raw{1, 7, 19, 31}; std::array<std::span<uint64_t const>, 1> members{raw};
    ChunkCodec canonical(18, uint64_t(4) << 20); auto bytes = canonical.encode(0, {}, 0, members);
    std::vector<uint8_t> expected(bytes.begin(), bytes.end());
    WorkerChunkInput input; input.new_members = members;
    auto result = client.encode(18, {&input, 1}, 4096);
    require(result.size() == 1 && result.front() == expected, "independently built compatible codec changed canonical bytes");
}
void generation(WorkerClient& client)
{
    std::array<uint8_t, 32> id{};
    auto const expected = worker_original_reference(18, 2, id);
    std::unique_ptr<CompactMemberReceiver> receiver;
    client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
    { if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count)); receiver->append(tile); });
    require(bool(receiver), "independently built generation worker emitted no member");
    auto member = receiver->finish(); auto view = member.view(); std::vector<uint64_t> output;
    for (uint64_t ordinal = 0; ordinal < PlotGroupFile::getChunkCountForK(18); ++ordinal)
        for (auto fragment : view.region(ordinal)) output.push_back(fragment);
    require(output == expected, "independently built compatible generation changed pinned reference bytes");
}
void cross_build(fs::path const& binaries)
{
    pool_test::Directory directory; auto ca = pool_test::issue(directory.path, "compatibility-ca");
    auto identity = pool_test::issue(directory.path, "compatibility-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 2000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls; options.worker_ms = 1000;
    RunningBroker server(options); pool::BrokerClient broker(server.server.endpoint(), tls);
    for (std::string const variant : {"compatible", "generation", "task", "codec", "representation"})
    {
        Process worker(binaries / ("compute_compatibility_" + variant), directory.path / "ready", server.server.endpoint(), tls);
        eventually([&] { return broker.status().worker_records.size() == 1; }, "cross-build worker did not enroll");
        auto const status = WorkerService::remote_status(worker.endpoint, tls);
        require(status.producer_build != worker_producer_build() && status.producer_build == "compute-fixture-" + variant,
            "cross-build fixture did not preserve different source provenance");
        auto const observed = broker.status().worker_records.front();
        require(observed.producer_build == status.producer_build && observed.compatibility == status.compatibility &&
            observed.codec_compatibility == status.codec_compatibility &&
            observed.capabilities == (pool::capability(pool::Kind::GenerateCpu) | pool::capability(pool::Kind::Pack)),
            "broker diagnostics lost compute contracts, capabilities or provenance");
        WorkerClientConfig client_options; client_options.broker = server.server.endpoint(); client_options.tls = tls;
        client_options.owner_id = "cross-build-" + variant; client_options.task_timeout_ms = 10000;
        WorkerClient client(client_options);
        if (variant == "compatible") { generation(client); codec(client); }
        else
        {
            incompatible(broker, pool::Kind::GenerateCpu);
            if (variant == "generation") codec(client);
            else incompatible(broker, pool::Kind::Pack);
        }
        worker.terminate();
        eventually([&] { return broker.status().worker_records.empty(); }, "cross-build worker did not unregister");
    }
}
void process_controls(fs::path const& binaries)
{
    pool_test::Directory directory; auto identity = pool::create_pool(directory.path / "pool", "restart-worker");
    auto tls = identity.tls; tls.timeout_ms = 2000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0, identity.server_name}; options.tls = tls; options.worker_ms = 1000;
    RunningBroker server(options);
    auto const binary = binaries / "compute_compatibility_compatible", ready = directory.path / "ready";
    for (WorkerControl const control : {WorkerControl::Pause, WorkerControl::Drain, WorkerControl::Resume})
    {
        Process worker(binary, ready, server.server.endpoint(), tls);
        auto const requested = WorkerService::remote_control(worker.endpoint, tls, control);
        worker.terminate(true);
        Process restarted(binary, ready, server.server.endpoint(), tls);
        auto const restored = WorkerService::remote_status(restarted.endpoint, tls);
        require(restored.durable_control && restored.admission == requested.admission && !restored.accepted_tasks && !restored.active.slots,
            "process restart lost enrolled admission intent or restored stale task accounting");
        restarted.terminate();
    }
}
}
int main(int argc, char** argv)
{
    try
    {
        require(argc == 1, "compute compatibility test takes no arguments");
        auto const binaries = fs::absolute(argv[0]).parent_path(); cross_build(binaries); process_controls(binaries);
        std::cout << "cross-build compute contracts, canonical output and process-restart controls passed\n"; return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
#endif
