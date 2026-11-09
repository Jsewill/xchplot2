#include "group_commands.hpp"
#include "compute_output.hpp"
#include "host/ComputeDiscovery.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/ComputeWorker.hpp"
#include <algorithm>
#include <charconv>
#include <iostream>
#include <set>
#include <thread>

namespace
{
namespace pool = group_grow::pool;
using Clock = std::chrono::steady_clock;
void require(bool value, std::string const& message)
{ if (!value) throw std::invalid_argument(message); }
unsigned number(std::string_view value, std::string const& option)
{
    unsigned result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && result,
        option + " requires a positive integer");
    return result;
}
std::string address(pool::Endpoint const& endpoint)
{ return (endpoint.host.find(':') == endpoint.host.npos ? endpoint.host : '[' + endpoint.host + ']') + ':' + std::to_string(endpoint.port); }
void string(std::string_view value)
{
    std::cout << '"';
    for (unsigned char c : value)
    {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 32) std::cout << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
        else std::cout << c;
    }
    std::cout << '"';
}
void resources(pool::Resources const& value)
{
    std::cout << "{\"cpus\":" << value.cpu_threads << ",\"slots\":" << value.slots
        << ",\"host_bytes\":" << value.host_bytes << ",\"gpu_bytes\":" << value.gpu_bytes << '}';
}
char const* kind(pool::Kind value)
{
    if (value == pool::Kind::GenerateCpu) return "generate-cpu";
    if (value == pool::Kind::GenerateGpu) return "generate-gpu";
    if (value == pool::Kind::GenerateAny) return "generate";
    return "encode";
}
struct BrokerSnapshot
{
    pool::Endpoint endpoint;
    pool::BrokerStatusPage summary;
    std::vector<pool::BrokerWorkerStatus> workers;
    std::vector<pool::BrokerOwnerStatus> owners;
    std::vector<pool::BrokerJobStatus> jobs;
    bool truncated = false;
};
BrokerSnapshot snapshot(pool::Endpoint const& endpoint, pool::TlsOptions const& tls,
    unsigned limit, Clock::time_point deadline)
{
    pool::BrokerClient base(endpoint, tls, 1); auto client = base.with_deadline(deadline);
    BrokerSnapshot result; result.endpoint = endpoint;
    for (auto section : {pool::StatusSection::Workers, pool::StatusSection::Owners, pool::StatusSection::Jobs})
    {
        uint64_t cursor = 0; unsigned remaining = limit;
        do
        {
            auto page = client.status({section, cursor, std::min(remaining, pool::max_status_page_size)});
            result.summary.workers = page.workers; result.summary.owners = page.owners; result.summary.jobs = page.jobs;
            result.summary.queued = page.queued; result.summary.granted = page.granted;
            result.summary.running = page.running; result.summary.cancelled = page.cancelled;
            auto count = page.worker_records.size() + page.owner_records.size() + page.job_records.size();
            require(count <= remaining, "broker status exceeded requested page bound");
            result.workers.insert(result.workers.end(), page.worker_records.begin(), page.worker_records.end());
            result.owners.insert(result.owners.end(), page.owner_records.begin(), page.owner_records.end());
            result.jobs.insert(result.jobs.end(), page.job_records.begin(), page.job_records.end());
            require(!page.next_cursor || page.next_cursor > cursor, "broker status cursor did not advance");
            cursor = page.next_cursor; remaining -= unsigned(count);
            if (cursor && !remaining) result.truncated = true;
        } while (cursor && remaining);
    }
    return result;
}
void broker_json(BrokerSnapshot const& value)
{
    std::cout << "{\"endpoint\":"; string(address(value.endpoint));
    std::cout << ",\"authenticated\":true,\"summary\":{\"workers\":" << value.summary.workers
        << ",\"owners\":" << value.summary.owners << ",\"jobs\":" << value.summary.jobs
        << ",\"queued\":" << value.summary.queued << ",\"granted\":" << value.summary.granted
        << ",\"running\":" << value.summary.running << ",\"cancelled\":" << value.summary.cancelled << "},\"workers\":[";
    bool comma = false;
    for (auto const& worker : value.workers)
    {
        if (comma) std::cout << ','; comma = true;
        std::cout << "{\"identity\":"; string(worker.identity);
        std::cout << ",\"instance\":"; string(worker.instance);
        std::cout << ",\"generation_compatibility\":"; string(worker.compatibility);
        std::cout << ",\"codec_compatibility\":"; string(worker.codec_compatibility);
        std::cout << ",\"producer_build\":"; string(worker.producer_build);
        std::cout << ",\"endpoint\":"; string(address(worker.endpoint));
        std::cout << ",\"admission\":"; string(pool::admission_name(worker.admission));
        std::cout << ",\"capacity\":"; resources(worker.capacity);
        std::cout << ",\"reserved\":"; resources(worker.reserved);
        std::cout << ",\"seen_ms\":" << worker.seen_ms << '}';
    }
    std::cout << "],\"owners\":["; comma = false;
    for (auto const& owner : value.owners)
    {
        if (comma) std::cout << ','; comma = true;
        std::cout << "{\"identity\":"; string(owner.identity);
        std::cout << ",\"owner_id\":"; string(owner.owner_id);
        std::cout << ",\"queued\":" << owner.queued << ",\"granted\":" << owner.granted
            << ",\"running\":" << owner.running << ",\"cancelled\":" << owner.cancelled << '}';
    }
    std::cout << "],\"jobs\":["; comma = false;
    for (auto const& job : value.jobs)
    {
        if (comma) std::cout << ','; comma = true;
        std::cout << "{\"id\":" << job.id << ",\"owner_id\":"; string(job.owner_id);
        std::cout << ",\"worker_identity\":"; string(job.worker_identity);
        std::cout << ",\"kind\":"; string(kind(job.kind));
        std::cout << ",\"state\":"; string(pool::job_state_name(job.state));
        std::cout << ",\"waiting\":"; string(pool::wait_reason_name(job.waiting));
        std::cout << ",\"resources\":"; resources(job.resources);
        std::cout << ",\"k\":" << unsigned(job.k) << ",\"attempt\":" << job.attempt
            << ",\"work_units\":" << job.work_units << ",\"age_ms\":" << job.age_ms << '}';
    }
    std::cout << "],\"truncated\":" << (value.truncated ? "true" : "false") << '}';
}
void worker_json(pool::Endpoint const& endpoint, group_grow::WorkerStatus const& value)
{
    std::cout << "{\"endpoint\":"; string(address(endpoint));
    std::cout << ",\"authenticated\":true,\"identity\":"; string(value.identity);
    std::cout << ",\"node_id\":"; string(value.node_id);
    std::cout << ",\"instance\":"; string(value.instance);
    std::cout << ",\"generation_compatibility\":"; string(value.compatibility);
    std::cout << ",\"codec_compatibility\":"; string(value.codec_compatibility);
    std::cout << ",\"producer_build\":"; string(value.producer_build);
    std::cout << ",\"durable_control\":" << (value.durable_control ? "true" : "false");
    std::cout << ",\"admission\":"; string(pool::admission_name(value.admission));
    std::cout << ",\"capacity\":"; resources(value.capacity);
    std::cout << ",\"active\":"; resources(value.active);
    std::cout << ",\"accepted_tasks\":" << value.accepted_tasks << ",\"waiting_cpu\":" << value.waiting_cpu
        << ",\"baseline_host_bytes\":" << value.baseline_host_bytes
        << ",\"broker_limit\":" << value.broker_limit << ",\"known_brokers\":" << value.known_brokers
        << ",\"healthy_brokers\":" << value.healthy_brokers << ",\"deferred_brokers\":" << value.deferred_brokers
        << ",\"broker_transport_bytes\":" << value.broker_transport_bytes
        << ",\"broker_saturated\":" << (value.broker_saturated ? "true" : "false") << '}';
}
void broker_text(BrokerSnapshot const& value, bool verbose)
{
    std::cout << "Broker " << address(value.endpoint) << ": " << value.summary.workers << " workers, "
        << value.summary.owners << " plotting job owners; " << value.summary.queued << " queued, " << value.summary.granted
        << " assigned, " << value.summary.running << " running\n";
    if (value.truncated) std::cerr << "Listing truncated; increase --limit to inspect more rows.\n";
    if (!verbose) return;
    for (auto const& worker : value.workers)
        std::cout << "  Worker " << worker.identity.substr(0, 12) << ' ' << address(worker.endpoint) << ' '
            << pool::admission_name(worker.admission) << ": reserved " << worker.reserved.slots << '/' << worker.capacity.slots
            << " slots, " << worker.reserved.cpu_threads << '/' << worker.capacity.cpu_threads << " CPUs, "
            << worker.reserved.host_bytes << '/' << worker.capacity.host_bytes << " host bytes, "
            << worker.reserved.gpu_bytes << '/' << worker.capacity.gpu_bytes << " GPU bytes\n"
            << "    Generation: " << worker.compatibility << "; codec: " << worker.codec_compatibility
            << "; producer: " << worker.producer_build << '\n';
    for (auto const& owner : value.owners)
        std::cout << "  Plotting job owner " << owner.owner_id << ": " << owner.queued << " queued, " << owner.granted
            << " assigned, " << owner.running << " running\n";
    for (auto const& job : value.jobs)
        std::cout << "  Task " << job.id << " owner=" << job.owner_id << ' ' << kind(job.kind) << " k=" << unsigned(job.k)
            << ' ' << pool::job_state_name(job.state) << " waiting=" << pool::wait_reason_name(job.waiting) << '\n';
}
void worker_text(pool::Endpoint const& endpoint, group_grow::WorkerStatus const& value, bool verbose)
{
    std::cout << "Worker " << address(endpoint) << ' ' << pool::admission_name(value.admission)
        << ": " << value.accepted_tasks << " accepted tasks, " << value.waiting_cpu << " waiting for CPU; active "
        << value.active.slots << '/' << value.capacity.slots << " slots, " << value.active.cpu_threads << '/' << value.capacity.cpu_threads
        << " CPUs";
    if (verbose)
        std::cout << ", " << value.active.host_bytes << '/' << value.capacity.host_bytes << " host bytes, "
            << value.active.gpu_bytes << '/' << value.capacity.gpu_bytes << " GPU bytes\n"
            << "  Brokers: " << value.healthy_brokers << " healthy, " << value.known_brokers << '/' << value.broker_limit
            << " registered, " << value.deferred_brokers << " deferred; " << value.broker_transport_bytes << " reserved transport bytes";
    std::cout << '\n';
    std::cout << "  Worker control setting " << (value.durable_control ? "survives service restarts." : "lasts only for this process (explicit TLS credentials without enrollment).") << '\n';
    if (verbose) std::cout << "  Generation: " << value.compatibility << "; codec: " << value.codec_compatibility
        << "; producer: " << value.producer_build << '\n';
    if (value.broker_saturated) std::cout << "  Broker limit reached; raise compute worker --max-brokers to register with more brokers.\n";
}
} // namespace

int pos2gpu::run_compute_status(int argc, char** argv) try
{
    std::string const command = argv[0]; bool const control = command != "status";
    require(command == "status" || command == "pause" || command == "drain" || command == "resume", "unknown compute control");
    auto directory = pool::pool_directory(); bool json = false;
    ComputeOutputOptions output;
    unsigned timeout = 5000, limit = 128;
    std::vector<pool::Endpoint> brokers, workers;
    for (int i = 1; i < argc; ++i)
    {
        std::string const option = argv[i];
        if (option == "--help" || option == "-h" || option == "--help-all")
        {
            std::cout << "Usage: xchplot2 compute " << command << " [OPTIONS]\n";
            if (command == "pause") std::cout << "Stop this machine's running worker from accepting new tasks. Accepted tasks finish normally.\n";
            else if (command == "drain") std::cout << "Stop this machine's running worker from accepting new tasks. Its status becomes drained when accepted tasks finish.\n";
            else if (command == "resume") std::cout << "Allow this machine's running worker to accept new tasks after a pause or drain.\n";
            else std::cout << "Show saved setup and authenticated live worker, plotting job owner and task status.\n";
            std::cout << "  --directory DIR     saved pool membership\n"
                   "  --broker HOST:PORT  query a broker; repeat for additional brokers\n"
                   "  --worker HOST:PORT  query or control a worker directly\n"
                   "  --timeout-ms N      network query time in milliseconds, 100..60000; default 5000\n"
                   "  --limit N           rows per broker section, default 128 (maximum 4096)\n"
                   "  --json              machine-readable status\n"
                << compute_output_help
                << "Without addresses, use saved broker settings and LAN discovery. Control commands require the worker's own enrollment.\n";
            if (control) std::cout << "Enrolled workers keep their control setting across service restarts.\n";
            if (command == "pause" || command == "drain") std::cout << "Use compute resume to accept new tasks again.\n";
            return 0;
        }
        if (option == "--json") { json = true; continue; }
        if (option == "--no-json") { json = false; continue; }
        if (output.parse(option)) continue;
        require(i + 1 < argc, "missing value for " + option); std::string value = argv[++i];
        if (option == "--directory") directory = std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str()));
        else if (option == "--broker") brokers.push_back(pool::parse_endpoint(value));
        else if (option == "--worker") workers.push_back(pool::parse_endpoint(value));
        else if (option == "--timeout-ms") timeout = number(value, option);
        else if (option == "--limit") limit = number(value, option);
        else throw std::invalid_argument("unknown compute status option: " + option);
    }
    output.validate();
    require(timeout >= 100 && timeout <= 60000, "--timeout-ms must be between 100 and 60000 milliseconds");
    require(limit <= 4096, "--limit must be between 1 and 4096");
    require(brokers.size() <= 128 && workers.size() <= 128, "too many explicit status endpoints");
    auto identity = pool::load_pool(directory);
    if (!identity)
    {
        require(!control && brokers.empty() && workers.empty(), "shared compute is not configured; use compute init or compute join");
        std::cout << (json ? "{\"configured\":false}\n" : "Shared compute is not configured. Use compute init or compute join.\n"); return 0;
    }
    bool const explicit_workers = !workers.empty();
    bool const explicit_endpoints = explicit_workers || !brokers.empty();
    if (!explicit_endpoints && !identity->defaults.broker.empty()) brokers.push_back(pool::parse_endpoint(identity->defaults.broker));
    auto const deadline = Clock::now() + std::chrono::milliseconds(timeout);
    std::string discovery_status = explicit_endpoints ? "explicit endpoints selected" : "discovery disabled";
    if (!explicit_endpoints && identity->defaults.discovery)
    {
        pool::DiscoveryOptions options; options.pool_id = identity->pool_id; options.node_id = identity->node_id;
        options.name = identity->name; options.server_name = identity->server_name; options.interfaces = identity->defaults.interfaces;
        pool::ComputeDiscovery discovery(std::move(options));
        std::this_thread::sleep_until(std::min(deadline, Clock::now() + std::chrono::seconds(1)));
        for (auto const& service : discovery.services())
        {
            auto& destination = service.role == "broker" ? brokers : workers;
            if (destination.size() < 128 && (!control || !explicit_workers || service.role == "broker")) destination.push_back(service.endpoint);
        }
        discovery_status = discovery.status();
    }
    auto unique = [&](std::vector<pool::Endpoint>& values)
    {
        std::set<std::string> seen;
        std::erase_if(values, [&](auto& endpoint)
        { endpoint.server_name = identity->server_name; return !seen.insert(address(endpoint)).second; });
    };
    unique(brokers); unique(workers);
    struct Error { std::string endpoint, role, message; };
    std::vector<Error> errors;
    auto tls = identity->tls; tls.max_frame_bytes = 65536; tls.timeout_ms = std::min(timeout, 1000u);
    if (json)
    {
        std::cout << "{\"configured\":true,\"node_id\":"; string(identity->node_id);
        std::cout << ",\"name\":"; string(identity->name);
        std::cout << ",\"pool_id\":"; string(identity->pool_id);
        std::cout << ",\"policy_generation\":" << identity->policy_generation
            << ",\"certificate_expired\":" << (identity->certificate_expired ? "true" : "false")
            << ",\"authority_expired\":" << (identity->authority_expired ? "true" : "false") << ",\"certificate_expires\":";
        string(identity->certificate_expires); std::cout << ",\"authority_expires\":"; string(identity->authority_expires);
        std::cout << ",\"can_invite\":" << (identity->can_invite ? "true" : "false") << ",\"defaults\":{\"broker\":";
        string(identity->defaults.broker.empty() ? "auto" : identity->defaults.broker);
        std::cout << ",\"backend\":"; string(identity->defaults.backend);
        std::cout << ",\"cpus\":" << identity->defaults.cpus << ",\"host_bytes\":" << identity->defaults.host_bytes
            << ",\"gpu_bytes\":" << identity->defaults.gpu_bytes << ",\"max_brokers\":" << identity->defaults.max_brokers
            << ",\"discovery\":" << (identity->defaults.discovery ? "true" : "false") << ",\"interfaces\":[";
        bool comma = false;
        for (auto const& interface : identity->defaults.interfaces)
        { if (comma) std::cout << ','; comma = true; string(interface); }
        std::cout << "]},\"brokers\":[";
    }
    else if (!output.quiet)
    {
        std::cout << "Machine: " << identity->name
            << "\nBroker: " << (identity->defaults.broker.empty() ? "automatic for each plotting job" : identity->defaults.broker) << '\n';
        if (output.details())
        {
            std::cout << "Node: " << identity->node_id << "\nPool: " << identity->pool_id.substr(0, 12)
                << "\nMembership policy: " << identity->policy_generation
                << "\nCertificate expires: " << identity->certificate_expires << (identity->certificate_expired ? " (renewal required)" : "")
                << "\nPool authority expires: " << identity->authority_expires << (identity->authority_expired ? " (re-enrollment required)" : "") << '\n';
            auto automatic = [](uint64_t value) { return value ? std::to_string(value) : std::string("automatic"); };
            std::cout << "LAN discovery: " << (identity->defaults.discovery ? "on" : "off")
                << "\nOffered defaults: " << identity->defaults.backend << ", CPUs " << automatic(identity->defaults.cpus)
                << ", host bytes " << automatic(identity->defaults.host_bytes) << ", GPU bytes " << automatic(identity->defaults.gpu_bytes)
                << ", broker limit " << identity->defaults.max_brokers << "\nInterfaces:";
            if (identity->defaults.interfaces.empty()) std::cout << " active LAN interfaces";
            else for (auto const& interface : identity->defaults.interfaces) std::cout << ' ' << interface;
            std::cout << '\n';
        }
    }
    if (!json && identity->authority_expired)
        std::cerr << "The pool authority has expired; create a new pool and re-enroll its members.\n";
    else if (!json && identity->certificate_expired)
        std::cerr << "This machine's certificate needs renewal before it can connect.\n";
    bool comma = false; unsigned authenticated = 0, changed = 0;
    for (auto const& endpoint : brokers)
    {
        if (Clock::now() >= deadline) { errors.push_back({address(endpoint), "broker", "status time budget exhausted"}); continue; }
        try
        {
            auto value = snapshot(endpoint, tls, limit, deadline); ++authenticated;
            if (!explicit_workers)
                for (auto const& worker : value.workers) if (workers.size() < 256) workers.push_back(worker.endpoint);
            if (json) { if (comma) std::cout << ','; comma = true; broker_json(value); }
            else broker_text(value, output.details());
        }
        catch (std::exception const& error) { errors.push_back({address(endpoint), "broker", error.what()}); }
    }
    unique(workers); comma = false;
    if (json) std::cout << "],\"workers\":[";
    for (auto const& endpoint : workers)
    {
        if (Clock::now() >= deadline) { errors.push_back({address(endpoint), "worker", "status time budget exhausted"}); continue; }
        try
        {
            auto bounded = tls;
            bounded.deadline = deadline;
            auto value = group_grow::WorkerService::remote_status(endpoint, bounded);
            if (control)
            {
                if (value.node_id != identity->node_id)
                { if (explicit_workers) throw std::invalid_argument("this worker belongs to another machine"); continue; }
                auto action = command == "pause" ? group_grow::WorkerControl::Pause : command == "drain" ?
                    group_grow::WorkerControl::Drain : group_grow::WorkerControl::Resume;
                value = group_grow::WorkerService::remote_control(endpoint, bounded, action); ++changed;
            }
            ++authenticated;
            if (json) { if (comma) std::cout << ','; comma = true; worker_json(endpoint, value); }
            else worker_text(endpoint, value, output.details());
        }
        catch (std::exception const& error) { errors.push_back({address(endpoint), "worker", error.what()}); }
    }
    if (control && !changed) errors.push_back({"", "worker", "no reachable worker for this machine; supply --worker HOST:PORT when discovery is unavailable"});
    if (json)
    {
        std::cout << "],\"discovery\":"; string(discovery_status);
        std::cout << ",\"changed\":" << changed << ",\"errors\":["; comma = false;
        for (auto const& error : errors)
        {
            if (comma) std::cout << ','; comma = true;
            std::cout << "{\"endpoint\":"; string(error.endpoint);
            std::cout << ",\"role\":"; string(error.role);
            std::cout << ",\"message\":"; string(error.message); std::cout << '}';
        }
        std::cout << "]}\n";
    }
    else
    {
        if (!authenticated && !control) std::cout << "No authenticated live services found. " << discovery_status << '\n';
        for (auto const& error : errors) std::cerr << error.role << ' ' << error.endpoint << ": " << error.message << '\n';
    }
    return control ? (changed ? 0 : 1) : (explicit_endpoints && !authenticated ? 1 : 0);
}
catch (std::exception const& error) { std::cerr << "compute: " << error.what() << '\n'; return 1; }
