// Same-file group growth and bounded rollover. Ordinary upstream group bytes.
#include "host/GroupEngine.hpp"
#include "host/GroupStaging.hpp"
#include "host/GroupStorage.hpp"
#include "host/GroupMerge.hpp"
#include "host/GroupExternal.hpp"
#include "host/GroupScratch.hpp"
#include "host/GroupPolicy.hpp"
#include "host/GroupFarming.hpp"
#include "host/GroupFarmState.hpp"
#include "host/GroupWriter.hpp"
#include "host/Random.hpp"
#include "plot/Plotter.hpp"
#include "pos2_keygen.h"
#include "host/GroupPlotter.hpp"
#ifdef XCHPLOT2_GROUP_COMMANDS
#include "host/ComputeSession.hpp"
#endif
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
#include "host/ComputeWorker.hpp"
#include "host/GroupRemoteCodec.hpp"
#endif
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
#include "host/GroupGpu.hpp"
#endif

#include <chrono>
#include <iostream>
#include <map>
#include <optional>
#include <syncstream>
#include <thread>
#ifdef _WIN32
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

std::array<uint8_t, 32> pos2gpu::compute_owner_seed(std::array<uint8_t, 32> const& seed, std::string_view owner)
{
    // All storage owners sharing explicit farm keys still need disjoint identities.
    constexpr std::string_view domain = "xchplot2-pool-storage-owner-v1";
    std::array<pos2_sha256_segment, 3> const segments{{
        {reinterpret_cast<uint8_t const*>(domain.data()), domain.size()},
        {seed.data(), seed.size()},
        {reinterpret_cast<uint8_t const*>(owner.data()), owner.size()}}};
    std::array<uint8_t, 32> scoped{};
    if (pos2_sha256_segments(segments.data(), segments.size(), scoped.data()) != POS2_OK)
        throw std::runtime_error("storage-owner key namespace derivation failed");
    return scoped;
}

#ifdef GROW_TEST_CONSTRUCTION
namespace grow_controller_test
#else
namespace
#endif
{
using namespace group_tools;
namespace file_io = group_grow::file_io;
namespace fs = std::filesystem;
constexpr uint64_t mib = 1024 * 1024;

uint64_t sum(uint64_t left, uint64_t right)
{
    require(right <= UINT64_MAX - left, "byte count overflow");
    return left + right;
}
uint64_t product(uint64_t left, uint64_t right)
{
    require(!right || left <= UINT64_MAX / right, "byte count overflow");
    return left * right;
}

struct Identity
{
    PlotGroupParams params;
    std::vector<uint8_t> memo;
};
void same_identity(Identity const& left, Identity const& right)
{
    require(left.params.get_plot_group_id() == right.params.get_plot_group_id() &&
                left.params.get_k() == right.params.get_k() &&
                left.params.get_strength() == right.params.get_strength() &&
                left.params.get_meta_group() == right.params.get_meta_group() &&
                left.memo == right.memo,
            "requested identity/memo differs from destination");
}
void validate(Identity const& identity)
{
    auto const k = identity.params.get_k();
    require(k >= 18 && k <= 28 && !(k & 1), "even k in 18..28 required");
    require(identity.params.get_strength() >= 2 && identity.params.get_strength() <= k - 3,
            "strength must be in 2..k-3");
    require(identity.memo.size() <= 255, "memo exceeds upstream byte limit");
}

using Arguments = pos2gpu::GroupOptions;
struct Interrupted : std::runtime_error
{
    Interrupted() : std::runtime_error("interrupted; repeat the command with --resume") {}
};
void check_cancel(Arguments const& args)
{
    if (args.cancelled && args.cancelled()) throw Interrupted{};
}
void report_output(Arguments const& args, fs::path const& path, uint16_t members, uint64_t bytes, bool existing = false)
{
    if (args.on_output) args.on_output(path, members, bytes, existing);
    else if (!args.command.empty() && !args.json && members)
        std::cout << file_io::path_text(path) << '\n' << std::flush;
}

void add_destination_temporary(std::vector<fs::path>& roots, std::string_view policy,
                               fs::path const& directory, group_grow::StorageDevice const& device)
{
    bool const enabled = policy == "on" || (!device.drives.empty() &&
        ((policy == "ssd" && device.nonrotating) || (policy == "hdd" && device.rotating)));
    if (!enabled) return;
    if (std::find(roots.begin(), roots.end(), directory) == roots.end()) roots.push_back(directory);
}

group_grow::Identity initial_identity(Identity const& requested)
{
    validate(requested);
    group_grow::Identity value;
    value.header.magic = PlotGroupFile::MAGIC;
    value.header.version = PlotGroupFile::FORMAT_VERSION;
    value.header.group_id = requested.params.get_plot_group_id();
    value.header.k = uint8_t(requested.params.get_k());
    value.header.strength = requested.params.get_strength();
    value.header.meta_group = requested.params.get_meta_group();
    value.header.memo_length = uint8_t(requested.memo.size());
    value.header.chunk_index_offset = sizeof(value.header) + requested.memo.size();
    value.memo = requested.memo;
    return value;
}
Identity identity_from(group_grow::Identity const& saved)
{
    return {PlotGroupParams(saved.header.group_id, saved.header.k,
                           saved.header.strength, saved.header.meta_group), saved.memo};
}
void requested_identity(Arguments const& args, Identity const& identity)
{
    validate(identity);
    if (args.group)
        require(PlotGroupId(*args.group) == identity.params.get_plot_group_id(),
                "--group-id differs from saved identity");
    if (args.memo) require(hex(*args.memo) == identity.memo, "--memo differs from saved memo");
    if (args.k) require(*args.k == identity.params.get_k(), "--k differs from saved identity");
    if (args.strength)
        require(*args.strength == identity.params.get_strength(), "--strength differs from saved identity");
    if (args.meta)
        require(*args.meta == identity.params.get_meta_group(), "--meta-group differs from saved identity");
}

// Use exactly the production CLI's base-seed/subseed and BLS group derivation.
// Named files retain their identity in the output. Directory jobs also save
// their base seed and original owner namespace before creating any output.
struct Keys
{
    std::array<uint8_t, 32> seed{};
    std::vector<uint8_t> farmer, pool;
    int pool_kind = POS2_POOL_PH;
    uint8_t k = 28, strength = 2, meta = 0;

    explicit Keys(Arguments const& args)
    {
        require(args.seed && args.farmer,
                "key-derived output requires --seed and --farmer-pk");
        unsigned const sources = unsigned(args.pool_pk.has_value()) +
            unsigned(args.pool_ph.has_value()) + unsigned(args.pool_address.has_value());
        require(sources == 1,
                "choose exactly one --pool-pk, --pool-ph or --pool-contract-address");
        auto const raw_seed = hex(*args.seed);
        require(raw_seed.size() == seed.size(), "--seed must be 64 hex chars");
        std::copy(raw_seed.begin(), raw_seed.end(), seed.begin());
        auto const owner = args.seed_owner.value_or(args.pool_broker.empty() ? "" : args.pool_owner);
        if (!owner.empty()) seed = pos2gpu::compute_owner_seed(seed, owner);
        farmer = hex(*args.farmer);
        require(farmer.size() == 48, "--farmer-pk must be 96 hex chars");
        if (args.pool_pk)
        {
            pool = hex(*args.pool_pk);
            pool_kind = POS2_POOL_PK;
            require(pool.size() == 48, "--pool-pk must be 96 hex chars");
        }
        else if (args.pool_ph)
        {
            pool = hex(*args.pool_ph);
            require(pool.size() == 32, "--pool-ph must be 64 hex chars");
        }
        else
        {
            pool.resize(32);
            require(pos2_keygen_decode_address(args.pool_address->c_str(), pool.data()) == POS2_OK,
                    "invalid --pool-contract-address; expected xch1... or txch1...");
        }
        k = uint8_t(args.k.value_or(28));
        strength = uint8_t(args.strength.value_or(2));
        meta = uint8_t(args.meta.value_or(0));
        // This also validates the public keys before opening any output.
        validate(derive(0));
    }
    Identity derive(uint64_t ordinal) const
    {
        std::array<uint8_t, 32> subseed{}, group{};
        require(pos2_keygen_derive_subseed(seed.data(), ordinal, subseed.data()) == POS2_OK,
                "subseed derivation failed");
        std::vector<uint8_t> memo(128);
        size_t length = memo.size();
        int const rc = pos2_keygen_derive_group(subseed.data(), subseed.size(), farmer.data(),
            pool.data(), pool_kind, strength, group.data(), memo.data(), &length);
        require(rc == POS2_OK, "group key derivation failed (rc=" + std::to_string(rc) + ")");
        require(length <= memo.size(), "group key derivation returned an invalid memo size");
        memo.resize(length);
        Identity result{PlotGroupParams(PlotGroupId(group), k, strength, meta), std::move(memo)};
        validate(result);
        return result;
    }
};

struct Fixtures
{
    Identity identity;
    std::vector<fs::path> paths;
    uint64_t bytes = 0;
};
Fixtures fixtures(fs::path const& path, uint64_t budget)
{
    uint64_t const size = fs::file_size(file_io::native_path(path));
    require(size < budget / 16, "fixture manifest exceeds RAM budget");
    auto in = input(file_io::path_text(path));
    unsigned k, strength, meta;
    std::string group, memo, tail, base, extra;
    in >> k >> strength >> meta >> group >> memo;
    require(k <= 255 && strength <= 255 && meta <= 255, "invalid fixture parameters");
    std::getline(in, tail);
    std::istringstream range(tail);
    if (range >> base)
        require(number(base.c_str()) == 0 && !(range >> extra),
                "ordinary groups require fixture index base zero");
    Fixtures result{{PlotGroupParams(PlotGroupId(group), uint8_t(k), uint8_t(strength),
                                    uint8_t(meta)), hex(memo)}, {}, size * 4};
    validate(result.identity);
    in.exceptions(std::ios::badbit);
    std::string name;
    while (in >> std::quoted(name))
    {
        require(!name.empty() && name.size() <= 4096 && result.paths.size() < 65535,
                "invalid fixture path or membership");
        result.paths.emplace_back(fs::absolute(file_io::native_path(name)).lexically_normal());
        result.bytes = sum(result.bytes, name.size() * 4 + 1024);
        require(result.bytes < budget / 8, "fixture paths exceed RAM budget");
    }
    require(in.eof() && !result.paths.empty(), "invalid or empty fixture manifest");
    return result;
}

// Match the pinned Plotter's layout and final copy, before constructing it.
// Plotter retains its layout through its final fragment transfer. The build's
// producer-only move patch removes the upstream redundant vector copy; retain
// the conservative model when building without that patch. Reserving the maximum
// pairing count also bounds unusual identities.
uint64_t cpu_peak(PlotGroupParams const& group)
{
    auto const params = group.get_plot_params_for_index(0);
    uint64_t const section_pairs = max_pairs_per_section_possible(params);
    uint64_t const pairs = product(section_pairs, params.get_num_sections());
    uint64_t const element = std::max(
        {sizeof(Xs_Candidate), sizeof(T1Pairing), sizeof(T2Pairing), sizeof(T3Pairing)});
    uint64_t const block = PlotLayout::align_up(product(section_pairs, element) / 4, 64);
    uint64_t const layout = sum(product(block, 32), 2 * mib);
#if defined(GROW_MOVE_PLOTTER_OUTPUT) && GROW_MOVE_PLOTTER_OUTPUT
    constexpr uint64_t copies = 1;
#else
    constexpr uint64_t copies = 2;
#endif
    return sum(sum(layout, product(pairs, sizeof(uint64_t) * copies)), 128 * mib);
}
uint64_t fragment_bound(PlotGroupParams const& group)
{
    auto const params = group.get_plot_params_for_index(0);
    return product(product(max_pairs_per_section_possible(params), params.get_num_sections()),
                   sizeof(uint64_t));
}

struct RawPlan
{
    struct Chunk { uint64_t offset, bytes, count; };
    uint64_t count = 0, scratch = 0;
    std::vector<Chunk> chunks;
};
RawPlan inspect_raw(fs::path const& path, Identity const& identity, uint16_t member)
{
    auto in = input(file_io::path_text(path));
    uint64_t const size = fs::file_size(file_io::native_path(path));
    require(read<std::array<char, 4>>(in) == std::array<char, 4>{'p', 'o', 's', '2'} &&
                read<uint8_t>(in) == PlotFile::FORMAT_VERSION, "expected raw-v2 fixture");
    require(PlotGroupId(read<std::array<uint8_t, 32>>(in)) ==
                identity.params.get_plot_group_id(), "fixture group ID mismatch");
    require(read<uint8_t>(in) == identity.params.get_k() &&
                read<uint8_t>(in) == identity.params.get_strength() &&
                read<uint16_t>(in) == member &&
                read<uint8_t>(in) == identity.params.get_meta_group(),
            "fixture parameters or member index mismatch");
    require(read<uint8_t>(in) == identity.memo.size(), "fixture memo length mismatch");
    std::vector<uint8_t> memo(identity.memo.size());
    in.read(reinterpret_cast<char*>(memo.data()), memo.size());
    require(memo == identity.memo, "fixture memo mismatch");
    uint64_t const chunks = read<uint64_t>(in);
    require(chunks > 0 && chunks <= (1ull << (identity.params.get_k() -
                                           PlotFile::CHUNK_SPAN_RANGE_BITS)),
            "fixture raw chunk count outside bounds");
    uint64_t end = sum(51 + memo.size(), product(chunks, 8));
    require(end <= size, "truncated fixture index");
    auto payload = input(file_io::path_text(path));
    RawPlan result;
    for (uint64_t c = 0; c < chunks; ++c)
    {
        uint64_t const offset = read<uint64_t>(in);
        require(offset == end && offset <= size && size - offset >= 8,
                "invalid fixture raw offset");
        payload.seekg(offset);
        uint64_t const bytes = read<uint64_t>(payload);
        require(bytes >= 12 && bytes <= size - offset - 8, "invalid fixture raw length");
        uint64_t const count = read<uint32_t>(payload);
        uint64_t const fse = read<uint32_t>(payload), stubs = read<uint32_t>(payload);
        require(fse + stubs + 12 == bytes &&
                    stubs == (count * (identity.params.get_k() - 2) + 7) / 8,
                "invalid fixture compressed lengths");
        result.count = sum(result.count, count);
        result.scratch = std::max(result.scratch, sum(bytes, product(count, 32)));
        result.chunks.push_back({offset, bytes, count});
        end = sum(offset, sum(bytes, 8));
    }
    require(end == size && result.count > 0 &&
                product(result.count, 8) <= fragment_bound(identity.params),
            "fixture fragment count or trailing data invalid");
    result.scratch = sum(result.scratch, product(result.chunks.capacity(), sizeof(RawPlan::Chunk)));
    return result;
}
std::vector<uint64_t> load_raw(fs::path const& path, RawPlan const& plan, uint8_t k)
{
    std::vector<uint64_t> result(plan.count);
    auto in = input(file_io::path_text(path));
    size_t position = 0;
    uint64_t const range = 1ull << (k + PlotFile::CHUNK_SPAN_RANGE_BITS);
    for (size_t c = 0; c < plan.chunks.size(); ++c)
    {
        auto const& chunk = plan.chunks[c];
        in.seekg(chunk.offset);
        require(read<uint64_t>(in) == chunk.bytes, "fixture chunk changed after admission");
        std::vector<uint8_t> compressed(chunk.bytes);
        in.read(reinterpret_cast<char*>(compressed.data()), compressed.size());
        uint32_t count;
        std::memcpy(&count, compressed.data(), sizeof(count));
        require(count == chunk.count, "fixture fragment count changed after admission");
        // Decode only the admitted immutable bytes. Reopening through the
        // upstream reader would trust a concurrently replaced file's sizes.
        uint32_t fse, stubs;
        std::memcpy(&fse, compressed.data() + 4, sizeof(fse));
        std::memcpy(&stubs, compressed.data() + 8, sizeof(stubs));
        require(uint64_t(fse) + stubs + 12 == compressed.size() &&
                    stubs == (chunk.count * (k - 2) + 7) / 8,
                "fixture compressed lengths changed after admission");
        auto values = ChunkCompressor::decompressProofFragments(compressed, c * range, k - 2);
        require(values.size() == chunk.count && values.size() <= result.size() - position,
                "fixture fragment count changed");
        require(values.empty() || (values.front() >= c * range && values.back() < (c + 1) * range),
                "fixture fragments outside admitted raw chunk");
        std::copy(values.begin(), values.end(), result.begin() + position);
        position += values.size();
    }
    require(position == result.size(), "fixture fragment count changed");
    return result;
}
uint64_t resident(std::span<group_grow::Member const> values)
{
    uint64_t bytes = 0;
    for (auto const& value : values) bytes = sum(bytes, value.storage_bytes());
    return bytes;
}
uint64_t batch_hash_bytes(Arguments const& args, size_t count)
{
    // A generation-overlap batch can coexist with the batch being consumed.
    return args.member_storage == "compact" && args.pool_broker.empty() ?
        product(2, group_grow::MemberBatchHash::memory_bytes(count)) : 0;
}
uint64_t filesystem_capacity(fs::path const& target, uint64_t headroom, uint64_t metadata_allocated = 0)
{
    file_io::Space space{};
    auto const directory = target.parent_path();
    require(file_io::space_path(directory.c_str(), &space) == 0, "cannot inspect destination filesystem space");
    uint64_t const available = product(space.f_bavail, space.f_frsize);
    file_io::Stat info{};
    uint64_t allocated = 0;
    if (file_io::inspect_path(target.c_str(), &info) == 0) allocated = product(uint64_t(info.st_blocks), 512);
    uint64_t const total = sum(sum(available, allocated), metadata_allocated);
    require(total > headroom, "destination filesystem has no capacity above headroom");
    return total - headroom;
}

uint64_t available_host_bytes()
{
#ifdef _WIN32
    MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
    require(::GlobalMemoryStatusEx(&memory), "cannot inspect available host memory"); return memory.ullAvailPhys;
#endif
#ifdef __linux__
    std::ifstream in("/proc/meminfo");
    std::string key, unit;
    uint64_t value;
    while (in >> key >> value >> unit)
        if (key == "MemAvailable:") return product(value, 1024);
#endif
    return 0;
}

struct MemberLease
{
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    std::optional<group_grow::GpuProducer::BorrowedFragments> fragments;
#endif
    bool active() const
    {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        return fragments.has_value();
#else
        return false;
#endif
    }
    void reset()
    {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        fragments.reset();
#endif
    }
};
bool controller_profile();

struct Backend
{
    std::shared_ptr<group_grow::CodecScheduler> cpu_scheduler;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
    std::unique_ptr<group_grow::WorkerService> pool_service;
    std::shared_ptr<group_grow::WorkerClient> pool_client;
    group_grow::WorkerClientConfig pool_client_config;
    std::shared_ptr<group_grow::PoolRemoteEncoder> pool_remote;
#ifdef GROW_TEST_CONSTRUCTION
    std::shared_ptr<group_grow::RemoteEncoder> test_remote;
#endif
    std::mutex pool_mutex;
    uint64_t pool_retained = 0;
    std::atomic<uint64_t> pool_members{0}, pool_contribution_members{0}, pool_local_fallbacks{0}, pool_retries{0};

    static group_grow::pool::Kind pool_kind(Arguments const& args)
    {
        return args.backend == "cpu" ? group_grow::pool::Kind::GenerateCpu : args.backend == "gpu" ?
            group_grow::pool::Kind::GenerateGpu : group_grow::pool::Kind::GenerateAny;
    }
    static void observe_pool_task(group_grow::WorkerTaskObservation const& task)
    {
        if (!controller_profile()) return;
        std::osyncstream(std::cerr) << "[grow-pool-task] worker=" << task.worker_identity
            << " instance=" << task.worker_instance << " work_units=" << task.work_units
            << " elapsed_ns=" << task.elapsed_ns << " execution_ns=" << task.execution_ns
            << " transfer_ns=" << task.transfer_ns << " local_direct=" << unsigned(task.local_direct) << '\n';
    }
    static void wait_pool_retry(group_grow::WorkerCancellation const& cancel, std::chrono::milliseconds& delay)
    {
        auto const until = std::min(cancel.deadline, std::chrono::steady_clock::now() + delay);
        while (std::chrono::steady_clock::now() < until)
        {
            cancel.check();
            std::this_thread::sleep_until(std::min(until,
                std::chrono::steady_clock::now() + std::chrono::milliseconds(50)));
        }
        cancel.check(); delay = std::min(delay * 2, std::chrono::milliseconds(250));
    }

    std::shared_ptr<group_grow::WorkerClient> acquire_pool_client()
    {
        std::lock_guard lock(pool_mutex);
        if (!pool_client) pool_client = std::make_shared<group_grow::WorkerClient>(pool_client_config);
        return pool_client;
    }

    void initialize_pool(Arguments const& args)
    {
        group_grow::pool::TlsOptions tls;
        tls.ca_file = args.pool_ca; tls.cert_file = args.pool_cert; tls.key_file = args.pool_key;
        if (args.automatic_compute) tls.membership_directory = args.pool_directory;
        tls.timeout_ms = unsigned(args.pool_socket_timeout_ms); tls.max_frame_bytes = uint32_t(args.pool_frame_bytes);
        auto broker = group_grow::pool::parse_endpoint(args.pool_broker); broker.server_name = args.pool_server_name;
        if (args.pool_offer_ram)
        {
            group_grow::WorkerConfig offered;
            offered.broker = broker; offered.tls = tls;
            offered.listen = args.pool_listen.empty() ? group_grow::pool::Endpoint{"::", 0} : group_grow::pool::parse_endpoint(args.pool_listen);
            if (!args.pool_advertise.empty()) offered.advertise = group_grow::pool::parse_endpoint(args.pool_advertise);
            offered.capacity = {uint32_t(args.pool_offer_cpus), uint32_t(args.pool_offer_slots), args.pool_offer_ram, args.pool_offer_vram};
            offered.backend = args.pool_offer_backend; offered.tier = args.tier; offered.device = int(args.device);
            offered.task_timeout_ms = unsigned(args.pool_timeout_ms);
            offered.cpu_scheduler = cpu_scheduler;
            offered.max_task_bytes = sum(args.window, args.pool_frame_bytes);
#ifdef XCHPLOT2_GROUP_COMMANDS
            if (args.compute_session) args.compute_session->offer_local(std::move(offered));
            else
#endif
            {
                pool_service = std::make_unique<group_grow::WorkerService>(std::move(offered));
                pool_service->start();
            }
        }
        group_grow::WorkerClientConfig client;
        client.broker = broker; client.tls = tls;
        client.owner_id = args.pool_session_owner.empty() ? args.pool_owner : args.pool_session_owner;
        client.ephemeral_owner = args.automatic_compute && !args.pool_owner_explicit;
        client.compatibility = group_grow::worker_compatibility();
        client.generation_kind = pool_kind(args);
        client.task_timeout_ms = unsigned(args.pool_timeout_ms);
        client.cpu_scheduler = cpu_scheduler;
        client.local_service = pool_service.get();
#ifdef XCHPLOT2_GROUP_COMMANDS
        if (args.compute_session) client.local_service_provider = [session = args.compute_session] { return session->local_worker(); };
#endif
        client.max_active_generation = unsigned(args.pool_inflight);
        client.observe = observe_pool_task;
        client.max_task_bytes = sum(args.window, args.pool_frame_bytes);
        { std::lock_guard lock(pool_mutex); pool_client_config = std::move(client); }
        try { (void)acquire_pool_client(); }
        catch (std::exception const&) { if (!pool_service) throw; }
        // The controller's fixed reserve covers the client metadata. Keep its
        // control TLS connection allowance live in every engine admission.
        pool_retained = sum(group_grow::WorkerClient::memory_bytes(),
            group_grow::pool::Channel::memory_bytes(uint32_t(args.pool_frame_bytes)));
        mode = "pool"; generation_overlap = args.generation_overlap;
    }

    void configure_pool_codec(Arguments const& args)
    {
        if (args.pool_broker.empty() || !args.pool_codec_ram || pool_remote) return;
        group_grow::WorkerClientConfig client;
        client.broker = group_grow::pool::parse_endpoint(args.pool_broker);
        client.broker.server_name = args.pool_server_name;
        client.tls = {args.pool_ca, args.pool_cert, args.pool_key,
            unsigned(args.pool_socket_timeout_ms), uint32_t(args.pool_frame_bytes)};
        if (args.automatic_compute) client.tls.membership_directory = args.pool_directory;
        client.owner_id = args.pool_session_owner.empty() ? args.pool_owner : args.pool_session_owner;
        client.ephemeral_owner = args.automatic_compute && !args.pool_owner_explicit;
        client.compatibility = group_grow::worker_compatibility();
        client.generation_kind = pool_kind(args);
        client.task_timeout_ms = unsigned(args.pool_timeout_ms);
        client.cpu_scheduler = cpu_scheduler;
        client.local_service = pool_service.get();
#ifdef XCHPLOT2_GROUP_COMMANDS
        if (args.compute_session) client.local_service_provider = [session = args.compute_session] { return session->local_worker(); };
#endif
        client.max_active_generation = unsigned(args.pool_inflight);
        client.observe = observe_pool_task;
        client.max_task_bytes = sum(args.window, args.pool_frame_bytes);
        { std::lock_guard lock(pool_mutex); pool_client_config = client; }
        pool_remote = std::make_shared<group_grow::PoolRemoteEncoder>(args.pool_codec_ram,
            uint32_t(args.pool_frame_bytes), client.max_task_bytes, [this]
            {
                return acquire_pool_client();
            }, cpu_scheduler);
    }

    group_grow::Member generate_pool(Identity const& identity, uint16_t index, Arguments const& args,
                                    std::atomic<bool> const* stop = nullptr)
    {
        auto const k = uint8_t(identity.params.get_k());
        uint64_t const bound = fragment_bound(identity.params) / sizeof(uint64_t);
        std::optional<group_grow::CompactMemberReceiver> receiver;
        uint64_t received_count = 0;
        auto const plot_id = identity.params.get_plot_params_for_index(index).get_plot_id().data();
        group_grow::WorkerCancellation cancel;
        cancel.stop = stop;
        cancel.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(args.pool_timeout_ms);
        auto receive = [&](group_grow::PackedFragmentTile const& tile, uint64_t count)
            {
                group_grow::CodecScheduler::Lease compute(cpu_scheduler.get(), 1, [cancel] { cancel.check(); });
                require(count && count <= bound && (!receiver || count == received_count),
                    "remote member count exceeds admitted bound or changed during transfer");
                if (!receiver) { receiver.emplace(k, size_t(count), false); received_count = count; }
                receiver->append(tile);
            };
        std::string const task_id = group_grow::pool::random_token();
        group_grow::WorkerTaskIdentity task(task_id);
        uint64_t attempt = 0;
        unsigned permanent_failures = 0;
        auto delay = std::chrono::milliseconds(50);
        auto retry = [&]
        {
            receiver.reset(); received_count = 0; pool_retries.fetch_add(1);
            if (acquire_pool_client()->local_available())
            {
                cancel.check();
                acquire_pool_client()->generate_local(k, uint8_t(identity.params.get_strength()), plot_id,
                    receive, cancel);
                pool_local_fallbacks.fetch_add(1); return false;
            }
            wait_pool_retry(cancel, delay); return true;
        };
        for (;;)
        {
            cancel.check();
            require(attempt < UINT64_MAX, "pool task attempt exhausted"); task.attempt = ++attempt;
            try
            {
                acquire_pool_client()->generate(k, uint8_t(identity.params.get_strength()), plot_id,
                    receive, cancel, task);
                break;
            }
            catch (group_grow::pool::TransientTransport const&)
            { if (!retry()) break; }
            catch (group_grow::pool::StaleLease const&)
            { if (!retry()) break; }
            catch (std::exception const&)
            {
                receiver.reset(); received_count = 0;
                if (++permanent_failures < 2) { pool_retries.fetch_add(1); continue; }
                if (!acquire_pool_client()->local_available()) throw;
                cancel.check();
                acquire_pool_client()->generate_local(k, uint8_t(identity.params.get_strength()), plot_id,
                    receive, cancel);
                pool_local_fallbacks.fetch_add(1); break;
            }
        }
        cancel.check();
        require(receiver.has_value(), "remote producer did not deliver a member");
        group_grow::CodecScheduler::Lease compute(cpu_scheduler.get(), 1, [cancel] { cancel.check(); });
        auto result = receiver->finish();
        pool_members.fetch_add(1);
        if (args.member_storage == "raw")
        {
            // The compact destination and this exact raw allocation were both
            // charged before dispatch. No borrowed remote output survives.
            std::vector<uint64_t> raw(result.size()); size_t position = 0;
            auto const view = result.view();
            for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(k); ++chunk)
                for (auto value : view.region(chunk)) raw[position++] = value;
            require(position == raw.size(), "remote member boundary coverage differs");
            return group_grow::Member(std::move(raw));
        }
        return result;
    }
#endif
    void configure_network(Arguments const& args)
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (!args.pool_broker.empty())
            pool_retained = sum(group_grow::WorkerClient::memory_bytes(),
                group_grow::pool::Channel::memory_bytes(uint32_t(args.pool_frame_bytes)));
        configure_pool_codec(args);
#else
        (void)args;
#endif
    }
    std::shared_ptr<group_grow::RemoteEncoder> remote_encoder() const
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
#ifdef GROW_TEST_CONSTRUCTION
        if (test_remote) return test_remote;
#endif
        return pool_remote;
#else
        return {};
#endif
    }
    std::string mode = "idle", tier, fallback;
    bool generation_overlap = false;
    uint64_t fallback_retained = 0;
    uint64_t generation_ns = 0, generated_members = 0;
    group_grow::GenerationHistory generation_history;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    std::unique_ptr<group_grow::GpuProducer> gpu;
#endif

    static void report_compact_cpu(uint16_t index, group_grow::CpuTime const& cpu, bool profile)
    {
        if (!profile) return;
        auto const end = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(stderr, "[grow-producer-timing] event=compact member_index=%u end_ns=%llu "
            "cpu_ns=%llu cpu_measured=%u\n", unsigned(index), static_cast<unsigned long long>(end),
            static_cast<unsigned long long>(cpu.nanoseconds), unsigned(cpu.measured));
    }

    bool admit_gpu_generation(Arguments const& args, uint64_t fixed, uint64_t owned, uint64_t borrowed,
                              uint64_t available = 0, uint64_t retained = 0)
    {
        // Automatic raw output may use the existing final-member lease when
        // its owning copy cannot fit. Never re-enable overlap on a reused
        // backend: a destination task may still own that borrowed output.
        auto fits_headroom = [&](uint64_t peak)
        { return !available || peak <= retained || peak - retained <= available; };
        // Raw generation has no asynchronous GPU output consumer, so this
        // decision precedes any generation that could acquire a new lease.
        if (args.member_storage == "raw" && args.generation_overlap_auto && generation_overlap &&
            (sum(fixed, owned) > args.max_ram || !fits_headroom(owned)))
            generation_overlap = false;
        auto const peak = args.member_storage == "raw" && !generation_overlap ? borrowed : owned;
        return sum(fixed, peak) <= args.max_ram && fits_headroom(peak);
    }

    void initialize(Arguments const& args, Identity const& identity, uint64_t fixed, bool fixture)
    {
        if (mode != "idle")
        {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (gpu && args.member_storage == "raw" && args.generation_overlap_auto && generation_overlap)
                (void)admit_gpu_generation(args, fixed,
                    generation_peak(identity, args), generation_peak(identity, args, true), available_host_bytes(), retained());
#endif
            return;
        }
        generation_overlap = args.generation_overlap;
        mode = "cpu";
        if (fixture) { mode = "fixture"; return; }
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (!args.pool_broker.empty()) { initialize_pool(args); return; }
#endif
        if (args.backend == "cpu") { generation_overlap = false; return; }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        // A failed attempt can leave the production thread-local GPU context
        // alive even after its pool is released. Keep its host reserve on the
        // CPU fallback path as well.
        fallback_retained = group_grow::GpuProducer::runtime_host_allowance_bytes();
        try
        {
            group_grow::GpuProducerOptions options;
            options.device_id = int(args.device);
            options.tier = args.tier;
            options.member_overlap = args.generation_overlap && args.member_storage == "compact";
            options.preserve_match_scratch = args.generation_overlap_auto;
            options.compact_transfer = options.member_overlap;
            gpu = std::make_unique<group_grow::GpuProducer>(identity.params.get_k(),
                identity.params.get_strength(), options);
            require(admit_gpu_generation(args, fixed, generation_peak(identity, args),
                        generation_peak(identity, args, true), available_host_bytes(), retained()),
                    "GPU producer model exceeds --max-ram or available host memory");
            mode = "gpu";
            tier = gpu->tier();
            fallback_retained = 0;
        }
        catch (std::exception const& error)
        {
            gpu.reset();
            if (args.backend == "gpu") throw;
            mode = "cpu";
            generation_overlap = false;
            tier.clear();
            fallback = "gpu-unavailable-or-over-budget";
            if (!args.quiet) std::cerr << "auto backend selected CPU: " << error.what() << '\n';
        }
#else
        generation_overlap = false;
        require(args.backend != "gpu", "GPU producer is unavailable in this build");
        fallback = "gpu-not-built";
        if (!args.quiet) std::cerr << "auto backend selected CPU: GPU producer is unavailable in this build\n";
#endif
    }
    uint64_t generation_peak(Identity const& identity, Arguments const& args, bool borrow = false) const
    {
        if (mode == "idle") return 0;
        auto const k = uint8_t(identity.params.get_k());
        uint64_t const raw = fragment_bound(identity.params);
        uint64_t const compact = args.member_storage == "compact" ?
            group_grow::Member::compact_bytes(k, raw / sizeof(uint64_t)) : 0;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (mode == "pool")
        {
            uint64_t const packed = group_grow::Member::compact_bytes(k, raw / sizeof(uint64_t));
            uint64_t const returned = args.member_storage == "raw" ? sum(raw, packed) : packed;
            uint64_t const boundaries = product(PlotGroupFile::getChunkCountForK(k) + 1, sizeof(uint32_t));
            return sum(sum(sum(pool_retained, returned), boundaries),
                product(group_grow::pool::Channel::memory_bytes(uint32_t(args.pool_frame_bytes)), 2));
        }
#endif
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (gpu)
        {
            if (args.member_storage == "compact")
                return sum(gpu->host_tiled_generation_peak_bytes(), compact);
            return borrow ? gpu->host_borrowed_generation_peak_bytes() : gpu->host_generation_peak_bytes();
        }
#else
        (void)borrow;
#endif
        // CPU generation releases its layout before compact conversion. The
        // owned raw vector and complete compact destination coexist in conversion.
        return sum(std::max(cpu_peak(identity.params), sum(raw, compact)), fallback_retained);
    }
    uint64_t retained() const
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (pool_retained) return pool_retained;
#endif
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (gpu) return gpu->retained_host_bytes();
#endif
        return fallback_retained;
    }
    uint64_t device_bytes() const
    {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (gpu) return gpu->device_resident_bytes();
#endif
        return 0;
    }
    group_grow::Member generate(Identity const& identity, uint16_t index,
                                Arguments const& args, MemberLease* lease,
                                group_grow::MemberBatchHash* hash)
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (mode == "pool") return generate_pool(identity, index, args);
#endif
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (gpu)
        {
            auto const plot_id = identity.params.get_plot_params_for_index(index).get_plot_id().data();
            if (args.member_storage == "compact")
            {
                group_grow::CpuTime cpu;
                std::optional<group_grow::CompactMemberBuilder> builder;
                auto const bound = fragment_bound(identity.params) / sizeof(uint64_t);
                gpu->generate_into(plot_id, [&](std::span<uint64_t const> tile, uint64_t total)
                {
                    group_grow::CodecScheduler::Lease compute(cpu_scheduler.get());
                    std::optional<group_grow::ThreadCpuTimer> timing;
                    if (gpu->profile_enabled()) timing.emplace(cpu);
                    require(total <= bound, "GPU member count exceeds admitted compact bound");
                    if (!builder) builder.emplace(uint8_t(identity.params.get_k()), size_t(total), hash);
                    builder->append(tile);
                });
                require(builder.has_value(), "GPU producer did not deliver its member");
                group_grow::Member member;
                {
                    group_grow::CodecScheduler::Lease compute(cpu_scheduler.get());
                    std::optional<group_grow::ThreadCpuTimer> timing;
                    if (gpu->profile_enabled()) timing.emplace(cpu);
                    member = builder->finish();
                }
                report_compact_cpu(index, cpu, gpu->profile_enabled());
                return member;
            }
            if (lease)
            {
                lease->fragments.emplace(gpu->generate_borrowed(plot_id));
                return group_grow::Member::borrowed(lease->fragments->fragments());
            }
            return group_grow::Member(gpu->generate(plot_id));
        }
#else
        (void)args;
        (void)lease;
        (void)hash;
#endif
        require(mode == "cpu", "member producer was not initialized");
        Plotter::Options options{};
        options.validate = false;
        options.verbose = false;
        return group_grow::Member(
            Plotter(identity.params.get_plot_params_for_index(index)).run(options).t3_proof_fragments);
    }
};

struct Producer
{
    enum class FillStop { full, requested, memory };
    Identity const& identity;
    Fixtures const* fixture;
    Backend& backend;
    Arguments const& args;
    uint64_t budget, fixed, peak = 0, generation_peak = 0;
    uint64_t overlap_members = 0;
    FillStop fill_stop = FillStop::full;

#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
    uint64_t pool_receive_bytes(uint16_t first, size_t count) const
    {
        uint64_t const raw = fragment_bound(identity.params);
        uint64_t const member = args.member_storage == "compact" ?
            group_grow::Member::compact_bytes(uint8_t(identity.params.get_k()), raw / sizeof(uint64_t)) : raw;
        uint64_t const pending = next_peak(first) - backend.retained() - member;
        uint64_t const ordered = sizeof(std::pair<size_t const, group_grow::Member>) + 4 * sizeof(void*);
        uint64_t const tasks = std::min<uint64_t>(args.pool_inflight, count);
        return sum(product(count, sum(member, ordered)), product(tasks,
            sum(pending, sizeof(std::future<group_grow::Member>) + 2 * sizeof(uint64_t) + 256)));
    }
    uint64_t contribution_bytes(group_grow::Config const& config, uint16_t count) const
    {
        return sum(group_grow::ExternalEngine::contribution_input_memory_bytes(config, count),
            product(count, sum(group_grow::pool::Channel::memory_bytes(uint32_t(args.pool_frame_bytes)),
                sizeof(std::future<void>) + 256)));
    }
    // A fill may run beside other destinations. The owner records its timing
    // only after joining, keeping the construction history on its controller.
    bool defer_pool_history = false;
    uint64_t pool_elapsed_ns = 0;
    void record_pool_generation(size_t count, std::chrono::steady_clock::time_point begin)
    {
        pool_elapsed_ns = group_grow::elapsed_ns(begin);
        if (!count || defer_pool_history) return;
        backend.generated_members = sum(backend.generated_members, count);
        backend.generation_ns = sum(backend.generation_ns, pool_elapsed_ns);
        backend.generation_history.add(count, pool_elapsed_ns);
    }
    size_t fill_contributions(group_grow::ExternalEngine::ContributionInput& input, uint16_t first,
        uint64_t other_retained, uint64_t admitted, std::string const& progress,
        std::atomic<bool> const* stopped = nullptr,
        std::function<std::optional<group_grow::ScratchPool::IoLease>()> const& acquire_io = {})
    {
        auto const begin = std::chrono::steady_clock::now();
        uint64_t const planned = sum(sum(sum(fixed, other_retained), backend.retained()), admitted);
        require(planned <= budget, "contribution receive exceeds owner RAM admission");
        peak = std::max(peak, planned);
        generation_peak = std::max(generation_peak, admitted);
        auto client = backend.acquire_pool_client();
        auto const k = uint8_t(identity.params.get_k()), strength = uint8_t(identity.params.get_strength());
        std::atomic<bool> cancelled{false};
        std::timed_mutex receive_mutex;
        auto with_io = [&](group_grow::WorkerCancellation const& cancel, auto&& operation)
        {
            std::unique_lock lock(receive_mutex, std::defer_lock);
            do { cancel.check(); }
            while (!lock.try_lock_until(std::min(cancel.deadline,
                std::chrono::steady_clock::now() + std::chrono::milliseconds(50))));
            std::optional<group_grow::ScratchPool::IoLease> io;
            if (acquire_io)
                while (!(io = acquire_io()))
                {
                    cancel.check();
                    std::this_thread::sleep_until(std::min(cancel.deadline,
                        std::chrono::steady_clock::now() + std::chrono::milliseconds(10)));
                }
            cancel.check(); operation();
        };
        std::vector<std::future<void>> pending;
        pending.reserve(size_t(std::min<uint64_t>(args.pool_inflight, input.size())));
        struct Drain
        {
            std::atomic<bool>& cancelled;
            std::vector<std::future<void>>& pending;
            ~Drain() { cancelled.store(true); for (auto& task : pending) if (task.valid()) task.wait(); }
        } drain{cancelled, pending};
        uint16_t scheduled = 0;
        while (scheduled < input.size() || !pending.empty())
        {
            check_cancel(args);
            if (stopped && stopped->load()) cancelled.store(true);
            require(!cancelled.load(), "pool generation cancelled");
            // Retire every ready slot, then replenish it immediately. A slow
            // worker retains only its admitted group and never a whole wave.
            bool made_progress = false;
            for (size_t slot = 0; slot < pending.size();)
            {
                if (pending[slot].wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                { ++slot; continue; }
                pending[slot].get(); pending.erase(pending.begin() + slot); made_progress = true;
            }
            size_t const concurrency = client->suggested_generation_concurrency(k, strength,
                unsigned(std::min<uint64_t>(args.pool_inflight, input.size())));
            while (pending.size() < concurrency && scheduled < input.size())
            {
                uint16_t const member = scheduled;
                uint16_t const count = uint16_t(client->suggested_batch_members(k, strength,
                    unsigned(input.size() - scheduled)));
                require(count && count <= input.size() - scheduled, "invalid pool batch recommendation");
                std::vector<std::array<uint8_t, 32>> plot_ids; plot_ids.reserve(count);
                for (uint16_t offset = 0; offset < count; ++offset)
                {
                    auto const index = uint16_t(first + member + offset);
                    if (args.command.empty() || args.verbose) std::osyncstream(std::cerr) << progress << index << '\n';
                    plot_ids.push_back(identity.params.get_plot_params_for_index(index).get_plot_id().data());
                }
                pending.push_back(std::async(std::launch::async, [&, member, count, plot_ids = std::move(plot_ids)]
                {
                    group_grow::WorkerCancellation cancel;
                    cancel.stop = &cancelled;
                    cancel.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(args.pool_timeout_ms);
                    auto generate = [&](uint16_t slot, std::span<std::array<uint8_t, 32> const> ids)
                    {
                        uint16_t const members = uint16_t(ids.size());
                        bool touched = false;
                        auto receive = [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t received)
                        {
                            require(received == members, "pool contribution membership changed during transfer");
                            with_io(cancel, [&]
                            {
                                touched = true; // append may claim its group before validation or I/O throws.
                                input.append_chunk(slot, ordinal, bytes, received);
                            });
                        };
                        auto reset = [&]
                        {
                            if (touched)
                            {
                                with_io(cancel, [&] { input.reset_member(slot, members); });
                                touched = false;
                            }
                        };
                        // This identity owns exactly this ordered plot-ID body.
                        // A repartition below creates independent identities.
                        group_grow::WorkerTaskIdentity task(group_grow::pool::random_token());
                        uint64_t attempt = 0;
                        unsigned permanent_failures = 0;
                        auto delay = std::chrono::milliseconds(50);
                        auto retry = [&]
                        {
                            reset(); backend.pool_retries.fetch_add(1);
                            if (client->local_available())
                            {
                                cancel.check(); client->contributions_local(k, strength, ids, receive, cancel);
                                backend.pool_local_fallbacks.fetch_add(1); return false;
                            }
                            Backend::wait_pool_retry(cancel, delay); return true;
                        };
                        for (;;)
                        {
                            cancel.check();
                            require(attempt < UINT64_MAX, "pool task attempt exhausted"); task.attempt = ++attempt;
                            try
                            {
                                client->contributions(k, strength, ids, receive, cancel, task);
                                break;
                            }
                            catch (group_grow::pool::TransientTransport const&)
                            { if (!retry()) break; }
                            catch (group_grow::pool::StaleLease const&)
                            { if (!retry()) break; }
                            catch (group_grow::pool::UnsupportedResources const&)
                            {
                                reset();
                                // Admission rejection occurs before any output.
                                // Repeating this exact batch cannot reduce its
                                // codec scratch demand on the same executor.
                                if (members > 1 || !client->local_available()) throw;
                                cancel.check(); client->contributions_local(k, strength, ids, receive, cancel);
                                backend.pool_local_fallbacks.fetch_add(1); break;
                            }
                            catch (std::exception const&)
                            {
                                reset();
                                if (++permanent_failures < 2) { backend.pool_retries.fetch_add(1); continue; }
                                if (!client->local_available()) throw;
                                cancel.check(); client->contributions_local(k, strength, ids, receive, cancel);
                                backend.pool_local_fallbacks.fetch_add(1); break;
                            }
                        }
                        with_io(cancel, [&] { input.finish_member(slot, members); });
                    };
                    try { generate(member, plot_ids); }
                    catch (group_grow::pool::UnsupportedResources const&)
                    {
                        if (count == 1) throw;
                        // The prior producer has stopped. Any touched group
                        // was reset to unclaimed slots. Preserve every slot while
                        // retrying its members through independently fenced
                        // singleton tasks, without extending the group deadline.
                        backend.pool_retries.fetch_add(1);
                        for (uint16_t offset = 0; offset < count; ++offset)
                            generate(uint16_t(member + offset), {&plot_ids[offset], 1});
                    }

                }));
                scheduled = uint16_t(scheduled + count); made_progress = true;
            }
            if (!made_progress && !pending.empty()) pending.front().wait_for(std::chrono::milliseconds(10));
        }
        input.seal();
        backend.pool_contribution_members.fetch_add(input.size());
        record_pool_generation(input.size(), begin);
        fill_stop = FillStop::full;
        return input.size();
    }

    size_t fill_pool(std::vector<group_grow::Member>& values, uint16_t first, size_t wanted,
                     uint64_t other_retained, std::string const& progress,
                     std::function<bool()> const& stop, std::atomic<bool> const* stopped = nullptr)
    {
        size_t const before = values.size();
        auto const begin = std::chrono::steady_clock::now();
        struct Pending
        {
            size_t member;
            std::future<group_grow::Member> ready;
            uint64_t bytes;
        };
        std::atomic<bool> cancelled{false};
        std::vector<Pending> pending;
        pending.reserve(size_t(std::min<uint64_t>(args.pool_inflight, wanted - before)));
        std::map<size_t, group_grow::Member> completed;
        struct Drain
        {
            std::atomic<bool>& cancelled;
            std::vector<Pending>& pending;
            ~Drain() { cancelled.store(true); for (auto& item : pending) if (item.ready.valid()) item.ready.wait(); }
        } drain{cancelled, pending};
        auto client = backend.acquire_pool_client();
        auto const k = uint8_t(identity.params.get_k()), strength = uint8_t(identity.params.get_strength());
        uint64_t in_flight = product(pending.capacity(), sizeof(Pending));
        constexpr uint64_t completed_metadata = sizeof(std::pair<size_t const, group_grow::Member>) + 4 * sizeof(void*);
        uint64_t completed_bytes = 0;
        size_t scheduled = before;
        bool accepting = true;
        fill_stop = FillStop::full;
        while (accepting || !pending.empty())
        {
            check_cancel(args);
            if (stopped && stopped->load()) cancelled.store(true);
            require(!cancelled.load(), "pool generation cancelled");
            bool made_progress = false;
            for (size_t slot = 0; slot < pending.size();)
            {
                auto& item = pending[slot];
                if (item.ready.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                { ++slot; continue; }
                auto result = item.ready.get(); in_flight -= item.bytes;
                completed_bytes = sum(completed_bytes, sum(result.storage_bytes(), completed_metadata));
                completed.emplace(item.member, std::move(result));
                pending.erase(pending.begin() + slot); made_progress = true;
            }
            while (!completed.empty() && completed.begin()->first == values.size())
            {
                auto found = completed.begin();
                completed_bytes -= sum(found->second.storage_bytes(), completed_metadata);
                values.push_back(std::move(found->second)); completed.erase(found);
            }
            if (accepting && stop && stop()) { accepting = false; fill_stop = FillStop::requested; }
            size_t const concurrency = client->suggested_generation_concurrency(k, strength,
                unsigned(std::min<uint64_t>(args.pool_inflight, wanted - before)));
            while (accepting && pending.size() < concurrency && scheduled < wanted)
            {
                auto const index = uint16_t(first + scheduled);
                uint64_t const held = sum(sum(sum(other_retained, resident(values)), completed_bytes), in_flight);
                if (!fits(index, sum(held, completed_metadata + 256)))
                { accepting = false; fill_stop = FillStop::memory; break; }
                uint64_t const allocation = sum(next_peak(index) - backend.retained(), completed_metadata + 256);
                generation_peak = std::max(generation_peak, next_peak(index));
                if (args.command.empty() || args.verbose) std::osyncstream(std::cerr) << progress << index << '\n';
                pending.push_back({scheduled, std::async(std::launch::async, [&, index]
                    { return backend.generate_pool(identity, index, args, &cancelled); }), allocation});
                in_flight = sum(in_flight, allocation); ++scheduled; made_progress = true;
            }
            if (scheduled == wanted) accepting = false;
            uint64_t const actual = sum(sum(sum(sum(sum(fixed, other_retained), resident(values)),
                completed_bytes), in_flight), backend.retained());
            require(actual <= budget, "remote members exceed the owner RAM admission");
            peak = std::max(peak, actual);
            if (!made_progress && !pending.empty()) pending.front().ready.wait_for(std::chrono::milliseconds(10));
        }
        require(completed.empty(), "remote generation lost its ordered member prefix");
        size_t const added = values.size() - before;
        record_pool_generation(added, begin);
        return added;
    }

#endif

#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    struct PendingMember
    {
        uint8_t k;
        uint16_t index;
        bool profile;
        uint64_t count_bound, storage_bound, received = 0;
        group_grow::MemberBatchHash* hash;
        std::shared_ptr<group_grow::CodecScheduler> cpu_scheduler;
        std::optional<group_grow::CompactMemberBuilder> builder;
        std::optional<group_grow::CompactMemberReceiver> receiver;
        uint64_t packed_received = 0, packed_count = 0;
        std::optional<group_grow::Member> member;
        std::shared_future<uint64_t> ready;
        group_grow::CpuTime compact_cpu;

        ~PendingMember() { if (ready.valid()) ready.wait(); }
        void consume(std::span<uint64_t const> tile, uint64_t total)
        {
            group_grow::CodecScheduler::Lease compute(cpu_scheduler.get());
            std::optional<group_grow::ThreadCpuTimer> timing;
            if (profile) timing.emplace(compact_cpu);
            require(!member && !receiver && total <= count_bound, "GPU member count exceeds admitted compact bound");
            if (!builder) builder.emplace(k, size_t(total), hash);
            builder->append(tile);
            received = sum(received, tile.size());
            if (received == total)
            {
                // Commit the hash on the consumer thread before the device
                // slot is released to the next member's consumer.
                member.emplace(builder->finish());
                builder.reset();
            }
        }
        void consume_packed(group_grow::PackedFragmentTile const& tile, uint64_t total)
        {
            group_grow::CodecScheduler::Lease compute(cpu_scheduler.get());
            std::optional<group_grow::ThreadCpuTimer> timing;
            if (profile) timing.emplace(compact_cpu);
            require(!hash && !builder && !member && total <= count_bound,
                    "GPU packed member requires a hash-free compact consumer");
            if (!receiver)
            {
                receiver.emplace(k, size_t(total));
                packed_count = total;
            }
            require(total == packed_count, "GPU packed member count changed during transfer");
            receiver->append(tile);
            packed_received = sum(packed_received, tile.words.size());
            uint64_t const words = (total * (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS) + 63) / 64;
            if (packed_received == words)
            {
                member.emplace(receiver->finish());
                receiver.reset();
                received = total;
            }
        }
        group_grow::Member take()
        {
            require(ready.get() == received && member.has_value(), "GPU producer did not deliver its member");
            require(member->storage_bytes() <= storage_bound, "GPU compact allocation exceeds admitted bound");
            return std::move(*member);
        }
    };
    static_assert(2 * sizeof(PendingMember) < mib); // Existing controller reserve.
    std::unique_ptr<PendingMember> generate_pending(uint16_t index, uint64_t retained,
                                                   group_grow::MemberBatchHash* hash)
    {
        require(fits(index, retained), "member producer exceeds --max-ram or available host memory");
        uint64_t const count = fragment_bound(identity.params) / sizeof(uint64_t);
        auto const k = uint8_t(identity.params.get_k());
        auto pending = std::make_unique<PendingMember>();
        pending->k = k;
        pending->index = index;
        pending->profile = backend.gpu->profile_enabled();
        pending->count_bound = count;
        pending->storage_bound = group_grow::Member::compact_bytes(k, size_t(count));
        pending->hash = hash;
        pending->cpu_scheduler = backend.cpu_scheduler;
        auto const plot_id = identity.params.get_plot_params_for_index(index).get_plot_id().data();
        auto* const state = pending.get();
        if (!hash && backend.gpu->compact_transfer_available())
            pending->ready = backend.gpu->generate_packed_into_async(plot_id,
                [state](group_grow::PackedFragmentTile const& tile, uint64_t total) { state->consume_packed(tile, total); });
        else
            pending->ready = backend.gpu->generate_into_async(plot_id,
                [state](std::span<uint64_t const> tile, uint64_t total) { state->consume(tile, total); });
        generation_peak = std::max(generation_peak, next_peak(index));
        return pending;
    }
#endif

    // Fill normal, recovered and prefetched batches through the same bounded
    // producer. A pending compact member owns its destination until completion;
    // count its maximum storage before admitting the next device generation.
    size_t fill(std::vector<group_grow::Member>& values, uint16_t first, size_t wanted,
                uint64_t other_retained, group_grow::MemberBatchHash* hash, MemberLease* lease,
                std::string const& progress, std::function<bool()> const& stop = {})
    {
        fill_stop = FillStop::full;
        size_t const before = values.size();
        // Publication-only recovery needs no producer, even when the requested
        // backend is unavailable. Keep an already filled batch equally idle.
        if (before >= wanted) return 0;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (!args.pool_broker.empty())
        {
            // Packed network members use the engine's existing digest path.
            // Only CompactMemberBuilder can commit a producer prefix cache.
            require(!hash, "pool producer cannot populate a compact-builder hash cache");
            backend.initialize(args, identity, sum(fixed, other_retained), false);
            return fill_pool(values, first, wanted, other_retained, progress, stop);
        }
#endif
        auto const begin = std::chrono::steady_clock::now();
        bool borrowable = lease && can_borrow(sum(other_retained, resident(values)));
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        std::unique_ptr<PendingMember> pending;
        auto finish_pending = [&]
        {
            if (pending)
            {
                values.push_back(pending->take());
                Backend::report_compact_cpu(pending->index, pending->compact_cpu, pending->profile);
                pending.reset();
            }
        };
#endif
        while (values.size() < wanted)
        {
            check_cancel(args);
            size_t count = values.size();
            uint64_t retained = sum(other_retained, resident(values));
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (pending)
            {
                if (++count == wanted) break;
                retained = sum(retained, pending->storage_bound);
            }
#endif
            if (stop && stop()) { fill_stop = FillStop::requested; break; }
            auto const index = uint16_t(first + count);
            bool borrow = borrowable && count + 1 == wanted;
            uint64_t pending_allocation = 0;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (pending) pending_allocation = pending->storage_bound;
#endif
            bool admitted = fits(index, retained, borrow, pending_allocation);
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (!admitted && pending)
            {
                // Retry with the completed member's actual allocation before
                // reducing the batch for a conservative pending-member bound.
                finish_pending();
                continue;
            }
#endif
            // Live headroom can disable automatic raw overlap during fits().
            // Reconsider its lease before rejecting an otherwise fitting GPU.
            if (!admitted && lease && !borrowable) borrowable = can_borrow(retained);
            if (!admitted && borrowable && !borrow)
            {
                borrow = fits(index, retained, true);
                admitted = borrow;
            }
            if (!admitted)
            {
                require(!values.empty() || other_retained,
                        "member producer exceeds --max-ram or available host memory");
                fill_stop = FillStop::memory;
                break;
            }
            if (args.command.empty() || args.verbose) std::cerr << progress << index << '\n';
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (backend.gpu && backend.gpu->member_overlap_available())
            {
                bool const overlaps = pending &&
                    pending->ready.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
                auto next = generate_pending(index, retained, hash);
                finish_pending();
                pending = std::move(next);
                if (overlaps) ++overlap_members;
            }
            else
#endif
                values.push_back(generate(index, retained, borrow ? lease : nullptr, hash));
            if (borrow) break;
        }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        finish_pending();
#endif
        size_t const added = values.size() - before;
        if (added)
        {
            uint64_t const elapsed = group_grow::elapsed_ns(begin);
            backend.generation_ns = sum(backend.generation_ns, elapsed);
            backend.generated_members = sum(backend.generated_members, added);
            backend.generation_history.add(added, elapsed);
        }
        return added;
    }

    uint64_t input_peak(uint16_t index) const
    {
        if (fixture) return next_peak(index);
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (backend.mode == "pool") return next_peak(index);
#endif
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (backend.gpu) return backend.gpu->host_tiled_generation_peak_bytes();
#endif
        return sum(cpu_peak(identity.params), backend.fallback_retained);
    }
    bool fits_input(uint16_t index, uint64_t retained)
    {
        backend.initialize(args, identity, sum(fixed, retained), fixture != nullptr);
        uint64_t const generation = input_peak(index), held = backend.retained();
        uint64_t const planned = sum(sum(fixed, retained), std::max(generation, held));
        if (planned > budget) return false;
        uint64_t const available = fixture ? 0 : available_host_bytes();
        if (available && generation > held && generation - held > available) return false;
        peak = std::max(peak, planned); return true;
    }
    size_t fill_input(group_grow::ExternalEngine::PreparedInput& input, uint16_t first,
                    uint64_t other_retained, std::string const& progress, std::function<bool()> const& stop = {})
    {
        require(!input.sealed() && input.size(), "invalid prepared producer input");
        auto const begin = std::chrono::steady_clock::now();
        size_t const wanted = input.size(); size_t generated = 0; fill_stop = FillStop::full;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        // Queue/device calls remain on this constructing thread. Output
        // callbacks own only the scratch sink; they never call GPU methods.
        struct Sink
        {
            group_grow::ExternalEngine::PreparedInput& input;
            uint16_t member;
            uint8_t k;
            uint64_t count_bound, total = 0, received = 0;
            bool started = false;
            void start(uint64_t count)
            {
                require(count && count <= count_bound && (!started || total == count),
                    "GPU prepared member count exceeds admission");
                if (!started) { input.begin_member(member, count); total = count; started = true; }
            }
            void raw(std::span<uint64_t const> values, uint64_t count)
            {
                start(count); input.append_raw(values); received = sum(received, values.size());
                if (received == total) input.finish_member();
            }
            void packed(group_grow::PackedFragmentTile const& tile, uint64_t count)
            {
                start(count); input.append_packed(tile); received = sum(received, tile.words.size());
                uint64_t const bits = product(total, k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
                if (received == bits / 64 + (bits % 64 != 0)) input.finish_member();
            }
        };
        struct Drain
        {
            std::shared_future<uint64_t> ready;
            ~Drain() { if (ready.valid()) ready.wait(); }
        } pending;
#endif
        for (uint16_t member = 0; member < input.size(); ++member)
        {
            check_cancel(args);
            if (member && stop && stop()) { fill_stop = FillStop::requested; break; }
            uint16_t const index = uint16_t(first + member);
            require(fits_input(index, other_retained), "prepared member producer exceeds --max-ram or host headroom");
            generation_peak = std::max(generation_peak, input_peak(index));
            if (args.command.empty() || args.verbose) std::cerr << progress << index << '\n';
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            if (backend.mode == "pool")
            {
                auto options = args; options.member_storage = "compact";
                auto values = backend.generate_pool(identity, index, options);
                input.begin_member(member, values.size());
                std::vector<uint64_t> tile;
                tile.reserve(size_t(args.pool_frame_bytes / sizeof(uint64_t)));
                auto const view = values.view();
                for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(uint8_t(identity.params.get_k())); ++chunk)
                    for (auto value : view.region(chunk))
                    {
                        tile.push_back(value);
                        if (tile.size() == tile.capacity()) { input.append_raw(tile); tile.clear(); }
                    }
                if (!tile.empty()) input.append_raw(tile);
                input.finish_member(); ++generated; continue;
            }
#endif
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (backend.gpu)
            {
                auto const plot_id = identity.params.get_plot_params_for_index(index).get_plot_id().data();
                auto sink = std::make_shared<Sink>(Sink{input, member, uint8_t(identity.params.get_k()),
                    fragment_bound(identity.params) / sizeof(uint64_t)});
                if (backend.gpu->member_overlap_available())
                {
                    // output.acquire drains the previous OutputHandoff before
                    // launch reuses its device slot, so callbacks across members
                    // are serialized even while the next match stage overlaps.
                    bool const overlaps = pending.ready.valid() &&
                        pending.ready.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
                    auto ready = backend.gpu->compact_transfer_available() ?
                        backend.gpu->generate_packed_into_async(plot_id,
                            [sink](group_grow::PackedFragmentTile const& tile, uint64_t total) { sink->packed(tile, total); }) :
                        backend.gpu->generate_into_async(plot_id,
                            [sink](std::span<uint64_t const> tile, uint64_t total) { sink->raw(tile, total); });
                    auto previous = std::move(pending.ready); pending.ready = std::move(ready);
                    if (previous.valid()) previous.get();
                    if (overlaps) ++overlap_members;
                }
                else backend.gpu->generate_into(plot_id,
                    [sink](std::span<uint64_t const> tile, uint64_t total) { sink->raw(tile, total); });
                ++generated; continue;
            }
#endif
            // CPU generation inherently returns an owned raw member. Retire it
            // immediately after its bounded scratch consumer, before generating
            // another member. Fixture loading keeps its established validation.
            group_grow::Member values;
            if (fixture)
            {
                auto const plan = inspect_raw(fixture->paths[index], identity, index);
                values = group_grow::Member(load_raw(fixture->paths[index], plan, uint8_t(identity.params.get_k())));
            }
            else
            {
                auto options = args; options.member_storage = "raw";
                values = backend.generate(identity, index, options, nullptr, nullptr);
            }
            input.begin_member(member, values.size()); input.append_raw(values.view().raw()); input.finish_member(); ++generated;
        }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (pending.ready.valid()) pending.ready.get();
#endif
        if (generated < wanted) input.seal_prefix(); else input.seal();
        backend.generation_ns = sum(backend.generation_ns, group_grow::elapsed_ns(begin));
        backend.generated_members = sum(backend.generated_members, generated); return generated;
    }

    bool can_borrow(uint64_t retained = 0)
    {
        backend.initialize(args, identity, sum(fixed, retained), fixture != nullptr);
        return backend.mode == "gpu" && args.member_storage == "raw" && !overlap_enabled();
    }
    bool overlap_enabled() const { return backend.generation_overlap; }
    uint64_t fixture_peak(RawPlan const& plan) const
    {
        uint64_t const raw = product(plan.count, sizeof(uint64_t));
        uint64_t const conversion = args.member_storage == "compact" ?
            group_grow::Member::compact_bytes(uint8_t(identity.params.get_k()), size_t(plan.count)) : 0;
        return sum(raw, std::max(plan.scratch, conversion));
    }
    uint64_t next_peak(uint16_t index, bool borrow = false) const
    {
        if (!fixture) return backend.generation_peak(identity, args, borrow);
        require(index < fixture->paths.size(), "fixture member unavailable");
        return fixture_peak(inspect_raw(fixture->paths[index], identity, index));
    }
    bool fits(uint16_t index, uint64_t retained, bool borrow = false, uint64_t pending_allocation = 0)
    {
        backend.initialize(args, identity, sum(fixed, retained), fixture != nullptr);
        uint64_t const generation = next_peak(index, borrow), held = backend.retained();
        // Switching GPU handoff modes releases the old pinned output before
        // allocating the new one. Admit both the current and the next state.
        uint64_t const planned = sum(sum(fixed, retained), std::max(generation, held));
        if (planned > budget) return false;
        if (!fixture)
        {
            uint64_t const available = available_host_bytes();
            // An asynchronous consumer may not have allocated its compact
            // destination yet. Conservatively reserve that full allocation in
            // the live-headroom check as well as in the app-budget calculation.
            uint64_t const growth = sum(generation > held ? generation - held : 0, pending_allocation);
            if (available && growth > available) return false;
        }
        peak = std::max(peak, planned);
        return true;
    }
    group_grow::Member generate(uint16_t index, uint64_t retained, MemberLease* lease = nullptr,
                                group_grow::MemberBatchHash* hash = nullptr)
    {
        backend.initialize(args, identity, fixed, fixture != nullptr);
        group_grow::Member values;
        uint64_t predicted = 0;
        if (fixture)
        {
            auto const plan = inspect_raw(fixture->paths[index], identity, index);
            predicted = fixture_peak(plan);
            uint64_t const planned = sum(sum(fixed, retained), std::max(predicted, backend.retained()));
            require(planned <= budget, "fixture producer exceeds --max-ram");
            peak = std::max(peak, planned);
            values = group_grow::Member(load_raw(fixture->paths[index], plan,
                                                 uint8_t(identity.params.get_k())));
        }
        else
        {
            predicted = next_peak(index, lease != nullptr);
            require(fits(index, retained, lease != nullptr),
                    "member producer exceeds --max-ram or available host memory");
            uint64_t const available = available_host_bytes();
            uint64_t const already_resident = backend.retained();
            require(!available || predicted <= already_resident || predicted - already_resident <= available,
                    "available host memory is below the producer generation model");
            values = backend.generate(identity, index, args, lease, hash);
        }
        generation_peak = std::max(generation_peak, predicted);
        if (args.member_storage == "compact" && !values.compact())
        {
            uint64_t const compact = group_grow::Member::compact_bytes(
                uint8_t(identity.params.get_k()), values.size());
            uint64_t const conversion = sum(sum(sum(sum(fixed, retained), backend.retained()),
                                                   values.storage_bytes()), compact);
            require(conversion <= budget, "member compact conversion exceeds --max-ram");
            uint64_t const available = available_host_bytes();
            require(!available || compact <= available,
                    "available host memory is below the member compact conversion model");
            peak = std::max(peak, conversion);
            group_grow::CodecScheduler::Lease compute(backend.cpu_scheduler.get());
            values.compact(uint8_t(identity.params.get_k()), hash);
        }
        // Compact construction validates the fragments and builds boundaries.
        // Engine preparation performs those checks for raw members.
        uint64_t const actual = sum(sum(sum(fixed, retained), backend.retained()),
                                    values.storage_bytes());
        require(actual <= budget,
                "returned producer allocation exceeds --max-ram");
        peak = std::max(peak, actual);
        return values;
    }
    void admit_engine(std::span<group_grow::Member const> values)
    {
        uint64_t const planned = sum(sum(fixed, resident(values)), backend.retained());
        require(planned <= budget, "engine and retained producer buffers exceed --max-ram");
        peak = std::max(peak, planned);
    }
};
// One owning batch for the next group. No engine, output path or scratch I/O
// lease is acquired until the current group has finished publishing.
struct GroupPrefetch
{
    size_t owner = 0;
    std::optional<Identity> identity;
    uint16_t capacity = 0;
    std::vector<group_grow::Member> members;
    std::unique_ptr<group_grow::MemberBatchHash> hash;
    uint64_t metadata_bytes = 0;

    uint64_t resident_bytes() const { return resident(members); }
    void clear() { *this = GroupPrefetch{}; }
    void take(std::vector<group_grow::Member>& incoming,
              std::unique_ptr<group_grow::MemberBatchHash>& incoming_hash, size_t wanted, uint16_t cap)
    {
        size_t const count = std::min({members.size(), wanted, size_t(cap)});
        if (capacity <= cap)
        {
            incoming = std::move(members); incoming.resize(count);
            // A smaller cache cannot grow with the receiver's admitted batch.
            // Hash uncached members through the engine's existing fallback.
            if (capacity == cap) incoming_hash = std::move(hash);
            if (incoming_hash) incoming_hash->truncate(count);
        }
        else
        {
            incoming.reserve(count);
            for (size_t i = 0; i < count; ++i) incoming.push_back(std::move(members[i]));
            // The engine can hash compact members when a differently admitted
            // batch cannot retain the producer's original prefix cache.
        }
        clear();
    }
};

static_assert(sizeof(GroupPrefetch) + 255 < mib); // Existing controller reserve.

bool group_prefetch_enabled(Arguments const& args, Backend const& backend)
{
    return args.member_storage == "compact" && args.generation_overlap &&
        backend.generation_overlap && backend.mode != "cpu" && backend.mode != "idle";
}

size_t prefetch_group(GroupPrefetch& batch, Producer& producer, Identity const& identity,
    size_t owner, uint16_t capacity, uint64_t retained, std::string const& progress,
    std::function<bool()> const& stop)
{
    producer.fill_stop = Producer::FillStop::requested;
    if (!capacity || !batch.members.empty() || !group_prefetch_enabled(producer.args, producer.backend)) return 0;
    if (!producer.fits(0, retained)) { producer.fill_stop = Producer::FillStop::memory; return 0; }
    if (!group_prefetch_enabled(producer.args, producer.backend)) return 0;
    batch.owner = owner; batch.identity = identity; batch.capacity = capacity;
    if (producer.args.pool_broker.empty()) batch.hash = std::make_unique<group_grow::MemberBatchHash>(capacity);
    batch.members.reserve(capacity);
    require(batch.members.capacity() <= capacity, "next group batch exceeds admitted metadata capacity");
    auto const added = producer.fill(batch.members, 0, capacity, retained, batch.hash.get(), nullptr,
        progress, stop);
    if (!added) batch.clear();
    return added;
}

enum class ControllerWait : unsigned
{ consumer, cpu_workers, storage_lease, writer_queue, engine_memory, producer_memory, member_lease, generation_order,
    batch_full, group_full, scratch_capacity, count };
constexpr std::array controller_wait_names{"consumer", "cpu_workers", "storage_lease", "writer_queue",
    "engine_memory", "producer_memory", "member_lease", "generation_order", "batch_full", "group_full", "scratch_capacity"};
constexpr uint32_t wait_bit(ControllerWait reason) { return uint32_t(1) << unsigned(reason); }
uint32_t fill_wait_reason(Producer::FillStop reason)
{
    if (reason == Producer::FillStop::requested) return 0;
    return wait_bit(reason == Producer::FillStop::full ? ControllerWait::batch_full : ControllerWait::producer_memory);
}
bool controller_profile()
{
    static bool const enabled = [] { auto value = std::getenv("POS2GPU_PHASE_TIMING"); return value && value[0] == '1'; }();
    return enabled;
}
class ControllerTiming
{
public:
    ControllerTiming(char const* phase, uint64_t owner, uint32_t reasons = 0)
        : phase_(phase), owner_(owner), reasons_(reasons)
    { if (controller_profile()) begin_ = std::chrono::steady_clock::now(); }
    ~ControllerTiming()
    {
        if (!controller_profile()) return;
        auto stamp = [](auto time)
        { return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count(); };
        std::osyncstream(std::cerr) << "[grow-controller-timing] phase=" << phase_ << " owner=" << owner_
            << " begin_ns=" << stamp(begin_) << " end_ns=" << stamp(std::chrono::steady_clock::now())
            << " reasons=" << reasons_ << '\n';
    }
private:
    char const* phase_;
    uint64_t owner_;
    uint32_t reasons_;
    std::chrono::steady_clock::time_point begin_{};
};
struct ControllerWaits
{
    std::array<uint64_t, size_t(ControllerWait::count)> nanoseconds{};
    // Reasons may overlap, including why a still-retained batch stopped growing.
    // Charge only actual idle waits; their sum is not total elapsed time.
    void add(uint32_t reasons, uint64_t elapsed)
    {
        for (size_t i = 0; i < nanoseconds.size(); ++i)
            if (reasons & (uint32_t(1) << i)) nanoseconds[i] = sum(nanoseconds[i], elapsed);
    }
    void add(ControllerWaits const& other)
    { for (size_t i = 0; i < nanoseconds.size(); ++i) nanoseconds[i] = sum(nanoseconds[i], other.nanoseconds[i]); }
};
struct Metrics : group_grow::Stats
{
    ControllerWaits controller_waits;
    uint64_t members = 0, final_bytes = 0, capacity_bytes = 0, peak_file_bytes = 0;
    uint64_t workspace_bytes = 0, attempted_final_bytes = 0, attempted_peak_bytes = 0;
    uint64_t generated_members = 0, batches = 0, peak_modeled_bytes = 0, engine_reserved_bytes = 0;
    uint64_t cpu_generation_peak_bytes = 0, gpu_generation_peak_bytes = 0, pool_generation_peak_bytes = 0;
    uint64_t fresh_single_pass_batches = 0, exact_plan_batches = 0;
    uint64_t staged_batches = 0, staging_fallbacks = 0;
    uint64_t scratch_peak_bytes = 0;
    uint64_t automatic_merges = 0, automatic_fallbacks = 0;
    uint64_t tuning_changes = 0, selected_window_bytes = 0, selected_checkpoint_bytes = 0, max_batch_members = 0;
    uint64_t generation_overlap_batches = 0;
    uint64_t generation_overlap_members = 0;
    uint64_t stream_input_batches = 0, stream_input_written_bytes = 0, stream_input_read_bytes = 0, stream_input_preflight_rejections = 0;
    uint64_t packing_engine_bytes = 0, packing_released_bytes = 0;
    uint64_t destination_count = 0, writer_count = 0, queued_peak_bytes = 0, drive_overlap_batches = 0, queue_wait_ns = 0;
    uint64_t execution_active = 0, execution_batch = 0;
    group_grow::ExecutionEstimate execution_estimate;
    bool recovered = false, scratch_released = true;
    long double farming_response_bound_seconds = 0, farming_dynamic_work_seconds_per_sp = 0;
    uint64_t farming_certified_max_eligible_groups = 0;
    std::string stop_reason, construction_selected;
    std::string stream_input_trial = "not-tried";
    void record_generation(std::string_view mode, uint64_t bytes)
    {
        auto& target = mode == "pool" ? pool_generation_peak_bytes : mode == "cpu" ?
            cpu_generation_peak_bytes : gpu_generation_peak_bytes;
        target = std::max(target, bytes);
    }
};

uint64_t external_metadata(group_grow::ExternalConfig const& config)
{
    uint64_t bytes = sum(sizeof(config), product(config.scratch.capacity(), sizeof(group_grow::ScratchGrant)));
    for (auto const& root : config.scratch) bytes = sum(bytes, root.directory.capacity() + 1);
    return bytes;
}
uint64_t external_metadata(group_grow::ExternalEngine::Inspection const& saved)
{
    uint64_t bytes = sum(sizeof(saved), external_metadata(saved.external));
    bytes = sum(bytes, sum(saved.identity.memo.capacity(), product(saved.scratch.capacity(), sizeof(group_grow::ScratchGrant))));
    for (auto const& root : saved.scratch) bytes = sum(bytes, root.directory.capacity() + 1);
    for (auto const* values : {&saved.reserved_bytes, &saved.peak_reserved_bytes, &saved.allocated_bytes, &saved.occupied_bytes})
        bytes = sum(bytes, product(values->capacity(), sizeof(uint64_t)));
    return bytes;
}
uint64_t path_metadata(std::vector<fs::path> const& paths)
{
    uint64_t bytes = product(paths.capacity(), sizeof(fs::path) * 2);
    for (auto const& path : paths) bytes = sum(bytes, product(path.native().capacity() + 1, 2));
    return bytes;
}
struct Output
{
    uint64_t ordinal = 0, device = 0, inode = 0, bytes = 0, allocated = 0;
    int64_t modified_seconds = 0, modified_nanoseconds = 0;
    int64_t changed_seconds = 0, changed_nanoseconds = 0;
    uint16_t members = 0, pending_members = 0;
    bool pending = false;
    uint64_t auxiliary = 0, auxiliary_allocated = 0, retired = 0, retired_allocated = 0, reservation = 0;
    uint64_t charge() const { return sum(std::max({bytes, allocated, reservation}), auxiliary); }
};
Output snapshot(fs::path const& path, uint64_t ordinal)
{
    file_io::Stat info{};
    require(file_io::inspect_link(path.c_str(), &info) == 0 && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
                info.st_size >= 0 && info.st_blocks >= 0,
            "rollover output must be an unchanged regular file with one link");
    return {ordinal, uint64_t(info.st_dev), uint64_t(info.st_ino), uint64_t(info.st_size),
            product(uint64_t(info.st_blocks), 512), info.st_mtim.tv_sec, info.st_mtim.tv_nsec,
            info.st_ctim.tv_sec, info.st_ctim.tv_nsec};
}
std::optional<group_grow::ExternalEngine::Inspection> output_auxiliary(
    fs::path const& path, Output& output, uint64_t memory_limit)
{
    output.auxiliary = output.auxiliary_allocated = output.retired = output.retired_allocated = output.reservation = 0;
    std::optional<group_grow::ExternalEngine::Inspection> active;
    for (auto const& entry : fs::directory_iterator(file_io::native_path(path.parent_path())))
    {
        auto const name = file_io::path_text(entry.path().filename()), base = file_io::path_text(path.filename());
        std::optional<group_grow::ExternalEngine::Inspection> saved;
        bool const retired = name.starts_with(base + ".external.retired-");
        if (name != base + ".external" && !retired) continue;
        uint64_t const held = active ? external_metadata(*active) : 0;
        require(held < memory_limit, "saved output metadata exceeds --max-ram");
        if (!retired) saved = group_grow::ExternalEngine::inspect(file_io::path_text(path), memory_limit - held);
        else if (retired)
            saved = group_grow::ExternalEngine::inspect_retired(file_io::path_text(path), file_io::path_text(entry.path()), memory_limit - held);
        if (!saved) continue;
        uint64_t const charge = std::max(saved->metadata_reserved_bytes, saved->metadata_occupied_bytes);
        output.auxiliary = sum(output.auxiliary, charge);
        output.auxiliary_allocated = sum(output.auxiliary_allocated, saved->metadata_occupied_bytes);
        if (retired)
        {
            output.retired = sum(output.retired, charge);
            output.retired_allocated = sum(output.retired_allocated, saved->metadata_occupied_bytes);
        }
        else
        {
            output.reservation = saved->peak_bytes - saved->metadata_reserved_bytes;
            active = std::move(saved);
        }
    }
    return active;
}

void preserve_retired(group_grow::ScratchPool& pool, fs::path const& target, uint64_t memory_limit,
                      bool rollover = false)
{
    for (auto const& entry : fs::directory_iterator(file_io::native_path(target.parent_path())))
    {
        auto const name = file_io::path_text(entry.path().filename());
        auto const position = name.rfind(".external.retired-");
        if (position == std::string::npos) continue;
        auto const base = name.substr(0, position);
        if (base != target.filename() && !(rollover && base.starts_with("grow-group-"))) continue;
        auto const saved = group_grow::ExternalEngine::inspect_retired(
            file_io::path_text((target.parent_path() / file_io::native_path(base))), file_io::path_text(entry.path()), memory_limit);
        for (size_t i = 0; i < saved.external.scratch.size(); ++i) if (saved.occupied_bytes[i])
            pool.preserve(group_grow::ScratchRoot::inspect(file_io::native_path(saved.external.scratch[i].directory)), saved.occupied_bytes[i]);
    }
}

group_grow::Config configuration(Arguments const& args, uint64_t capacity, uint64_t caller_fixed,
    std::shared_ptr<group_grow::CodecScheduler> const& scheduler = {},
    std::shared_ptr<group_grow::RemoteEncoder> const& remote = {})
{
    group_grow::Config config;
    config.capacity_bytes = capacity;
    config.window_bytes = args.window;
    config.codec_bytes = std::min(args.max_ram / 16, 256 * mib);
    config.workers = args.workers ? unsigned(args.workers) :
        std::max(1u, std::thread::hardware_concurrency());
    config.cpu_scheduler = scheduler;
    config.remote_encoder = remote;
    config.checkpoint_bytes = args.checkpoint_bytes;
    config.io_overlap = args.io_overlap;
    config.release_cache = args.cache_release;
    require(caller_fixed < args.max_ram, "caller metadata exceeds --max-ram");
    config.memory_limit_bytes = args.max_ram - caller_fixed;
    return config;
}

std::shared_ptr<group_grow::SharedConstructionHistory> construction_history(
    group_grow::ConstructionHistories* histories, Arguments const& args,
    fs::path const& path, group_grow::Identity const& identity, unsigned workers, bool create = true,
    group_grow::ExternalConfig const* scratch = nullptr,
    std::shared_ptr<group_grow::SharedConstructionHistory> const& demand = {}, bool execution = false)
{
    if (!histories) return {};
    auto const device = group_grow::storage_device(path.parent_path());
    group_grow::ConstructionContext context{device.drives, identity.header.k, identity.header.strength, identity.header.meta_group,
        identity.memo.size(), args.max_ram, args.window, args.checkpoint_bytes,
        std::min(args.max_ram / 16, 256 * mib), workers, args.member_storage == "compact",
        args.io_overlap, args.cache_release};
    if (scratch) for (auto const& grant : scratch->scratch)
    {
        auto const source = group_grow::storage_device(file_io::native_path(grant.directory));
        if (source.drives.empty()) return {};
        context.scratch_drives.insert(context.scratch_drives.end(), source.drives.begin(), source.drives.end());
    }
    std::sort(context.scratch_drives.begin(), context.scratch_drives.end());
    context.scratch_drives.erase(std::unique(context.scratch_drives.begin(), context.scratch_drives.end()), context.scratch_drives.end());
    if (auto reused = histories->reuse_external(demand, context)) return reused;
    if (execution) return histories->execution(context);
    return histories->find(std::move(context), create);
}

using group_grow::ExternalBatchHistory;
static_assert(sizeof(ExternalBatchHistory) < mib / 16); // Existing controller reserve.

// Keep protocol selection and the one-way staging fallback in the controller.
// Each engine still owns its file lock, admission, checkpoints and publication.
struct Construction
{
    std::string path;
    group_grow::Config config;
    std::unique_ptr<group_grow::Engine> incremental;
    std::unique_ptr<group_grow::StagingEngine> staged;
    std::unique_ptr<group_grow::MergeEngine> merged;
    std::unique_ptr<group_grow::ExternalEngine> external;
    bool external_released = false, scratch_cleanup_complete = true;
    uint64_t peak_scratch_bytes = 0, remaining_metadata_bytes = 0;
    group_grow::Stats completed{};
    group_grow::ConstructionHistory history;
    ExternalBatchHistory external_batches;
    std::shared_ptr<group_grow::SharedConstructionHistory> shared_history;
    std::shared_ptr<group_grow::SharedConstructionHistory> demand_history;
    bool shape_recorded = false;
    uint64_t window_limit = 0, tuning_changes = 0, selected_batch = 0, external_checkpoint_bytes = 0;
    bool merge_reserve_failed = false, window_failed = false, nonrotating = false, known_storage = false;
    uint64_t staged_batches = 0, fallbacks = 0, peak_file_bytes = 0, workspace_bytes = 0;
    uint64_t automatic_merges = 0, automatic_fallbacks = 0;
    bool automatic = false, finish_merged = false;
    bool staged_recovery = false, terminal_packing = false, fresh_external_pack = false;

    static file_io::Stat snapshot(std::string const& path)
    {
        file_io::Stat value{};
        require(file_io::inspect_link(path.c_str(), &value) == 0 && file_io::is_regular(value.st_mode) && value.st_nlink == 1,
                "construction target changed during engine handoff");
        return value;
    }
    static void unchanged(file_io::Stat const& before, std::string const& path)
    {
        auto const after = snapshot(path);
        require(before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
            before.st_size == after.st_size && before.st_blocks == after.st_blocks &&
            before.st_mtim.tv_sec == after.st_mtim.tv_sec && before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
            before.st_ctim.tv_sec == after.st_ctim.tv_sec && before.st_ctim.tv_nsec == after.st_ctim.tv_nsec,
            "construction target changed during engine handoff");
    }
    static void same_saved(group_grow::Identity const& expected, group_grow::Identity const& actual)
    {
        same_identity(identity_from(expected), identity_from(actual));
        require(expected.header.group_size == actual.header.group_size && expected.bytes == actual.bytes &&
            expected.header.chunk_index_offset == actual.header.chunk_index_offset && expected.pending == actual.pending,
            "construction membership changed during engine handoff");
    }

    Construction(std::string path_, group_grow::Config config_, bool resume,
                 std::optional<group_grow::Identity> initial = {}, std::string_view mode = "incremental",
                 std::optional<group_grow::ExternalConfig> scratch = {}, uint64_t retained_metadata = 0)
        : path(std::move(path_)), config(config_), window_limit(config.window_bytes), automatic(mode == "auto"), finish_merged(mode == "merged")
    {
        require(retained_metadata <= config.capacity_bytes, "retired metadata exceeds the destination reservation");
        remaining_metadata_bytes = retained_metadata;
        config.capacity_bytes -= retained_metadata;
        auto const device = group_grow::storage_device(file_io::native_path(path).parent_path());
        nonrotating = device.nonrotating; known_storage = !device.drives.empty();
        if (nonrotating && config.window_bytes >= 256 * 1024) config.merge_inputs = 4;
        bool const external_recovery = group_grow::ExternalEngine::pending(path);
        bool const merged_recovery = group_grow::MergeEngine::pending(path);
        staged_recovery = external_recovery || merged_recovery || group_grow::StagingEngine::pending(path);
        if (external_recovery || (!resume && (mode == "external" || (automatic && scratch))))
        {
            if (!scratch && external_recovery)
                scratch = group_grow::ExternalEngine::inspect(path, config.memory_limit_bytes).external;
            require(scratch.has_value(), "external construction requires admitted temporary space");
            external = std::make_unique<group_grow::ExternalEngine>(path, config, std::move(*scratch), resume, std::move(initial));
        }
        else if (merged_recovery || (mode == "merged" && !resume))
            merged = std::make_unique<group_grow::MergeEngine>(path, config, resume, std::move(initial));
        else if (staged_recovery || (mode == "staged" && !resume))
            staged = std::make_unique<group_grow::StagingEngine>(path, config, resume, std::move(initial));
        else
        {
            incremental = std::make_unique<group_grow::Engine>(path, config, resume, std::move(initial));
            // An ordinary pending transaction must recover through its own
            // protocol. A complete group may start new compressed contributions.
            if ((mode == "staged" || mode == "merged") && !incremental->identity().pending)
            {
                auto saved = incremental->identity();
                auto const before = snapshot(path); incremental->confirm_identity();
                completed = add_stats(completed, incremental->stats());
                incremental.reset();
                if (mode == "merged") merged = std::make_unique<group_grow::MergeEngine>(path, config, true, saved);
                else staged = std::make_unique<group_grow::StagingEngine>(path, config, true, saved);
                unchanged(before, path);
                same_saved(saved, identity());
            }
        }
    }
    void attach_history(std::shared_ptr<group_grow::SharedConstructionHistory> source)
    {
        shared_history = std::move(source);
        if (external && !demand_history) demand_history = shared_history;
        refresh_history();
    }
    unsigned staging_workers(unsigned workers)
    {
        if (!demand_history) return workers / 2;
        std::lock_guard lock(demand_history->mutex);
        return group_grow::external_staging_workers(workers, demand_history->history);
    }
    void admit_staging_workers(unsigned workers, unsigned staging)
    {
        if (!demand_history) return;
        std::lock_guard lock(demand_history->mutex);
        demand_history->history.overlap_workers = workers;
        demand_history->history.overlap_staging_workers = staging;
    }
    void observe_demand(group_grow::CpuTime const& before, group_grow::CpuTime const& after,
        uint64_t elapsed, unsigned workers, bool packing, bool history_eligible = true)
    {
        if (!elapsed) return;
        group_grow::ConstructionSample sample;
        sample.elapsed_ns = elapsed; sample.workers = workers;
        sample.cpu_measured = before.measured && after.measured && after.nanoseconds >= before.nanoseconds;
        if (sample.cpu_measured) sample.cpu_ns = after.nanoseconds - before.nanoseconds;
        if (controller_profile())
            std::osyncstream(std::cerr) << "[grow-controller-cpu] phase=" << (packing ? "external_pack_body" : "external_append")
                << " history_eligible=" << history_eligible
                << " target=" << std::quoted(path) << " observed_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()
                << " cpu_ns=" << sample.cpu_ns << " measured=" << sample.cpu_measured
                << " elapsed_ns=" << elapsed << " grant=" << workers << '\n';
        if (!history_eligible || !demand_history) return;
        std::lock_guard lock(demand_history->mutex);
        auto& history = demand_history->history;
        (packing ? history.external_packing : history.external_staging).add(sample);
    }
    void refresh_history()
    { if (shared_history) history = shared_history->snapshot(); }
    void observe(group_grow::ConstructionSample sample,
                 group_grow::ConstructionSamples group_grow::ConstructionHistory::* phase = nullptr)
    {
        if (phase == &group_grow::ConstructionHistory::shapes ? !sample.member_bytes :
            (!sample.elapsed_ns || !sample.written_bytes)) return;
        if (phase) (history.*phase).add(sample); else history.add(sample);
        if (shared_history)
        {
            std::lock_guard lock(shared_history->mutex);
            if (phase) (shared_history->history.*phase).add(sample); else shared_history->history.add(sample);
        }
    }
    static group_grow::ConstructionSample phase_sample(group_grow::PhaseStats const& before,
        group_grow::PhaseStats const& after, long double codec_bytes, unsigned workers)
    {
        return {after.elapsed_ns - before.elapsed_ns, after.read_ns - before.read_ns,
            after.write_ns - before.write_ns, after.sync_ns - before.sync_ns,
            after.read_bytes - before.read_bytes, after.written_bytes - before.written_bytes,
            after.syncs - before.syncs, codec_bytes, workers, after.codec_ns - before.codec_ns,
            after.decode_ns - before.decode_ns, after.read_calls - before.read_calls,
            after.write_calls - before.write_calls};
    }
    static void execution_cpu_sample(group_grow::ConstructionSample& sample,
        group_grow::CpuTime const& before, group_grow::CpuTime const& after,
        group_grow::CpuTime const& serial_before, group_grow::CpuTime const& serial_after)
    {
        sample.cpu_measured = before.measured && after.measured && after.nanoseconds >= before.nanoseconds;
        if (sample.cpu_measured) sample.cpu_ns = after.nanoseconds - before.nanoseconds;
        sample.serial_cpu_measured = serial_before.measured && serial_after.measured &&
            serial_after.nanoseconds >= serial_before.nanoseconds;
        if (sample.serial_cpu_measured) sample.serial_cpu_ns = serial_after.nanoseconds - serial_before.nanoseconds;
    }
    void observe_external_contribution(group_grow::Stats const& before, uint64_t elapsed, uint16_t members,
        group_grow::CpuTime const& cpu, group_grow::CpuTime const& serial,
        group_grow::ExecutionIO const& scratch, group_grow::ExecutionIO const& destination)
    {
        auto const& after = external->stats();
        group_grow::ConstructionSample sample{elapsed, after.read_ns - before.read_ns,
            after.write_ns - before.write_ns, after.sync_ns - before.sync_ns,
            after.read_bytes - before.read_bytes, after.written_bytes - before.written_bytes,
            after.syncs - before.syncs, 2 * static_cast<long double>(external->contribution_bytes()),
            unsigned(std::max<uint64_t>(1, after.contribution_workers)), after.codec_ns - before.codec_ns,
            after.decode_ns - before.decode_ns, after.read_calls - before.read_calls, after.write_calls - before.write_calls};
        execution_cpu_sample(sample, cpu, external->contribution_cpu_time(), serial, external->contribution_serial_cpu_time());
        sample.member_bytes = members ? static_cast<long double>(external->contribution_bytes()) / members : 0;
        sample.scratch_io = external->scratch_service().since(scratch);
        sample.destination_io = external->destination_service().since(destination);
        sample.io_attributed = external->service_attributed();
        observe(sample, &group_grow::ConstructionHistory::execution_contributions);
    }
    void observe_external(uint16_t members, group_grow::Stats const& before)
    {
        require(bool(external), "external observation requires an external group");
        auto const& after = external->stats();
        auto const plan = size_t(group_grow::WorkPhase::plan), encode = size_t(group_grow::WorkPhase::encode);
        external_batches.add(members, phase_sample(before.phases[plan], after.phases[plan], 0, 1),
            phase_sample(before.phases[encode], after.phases[encode], external->contribution_bytes(),
                unsigned(std::max<uint64_t>(1, after.contribution_workers))));
        auto const resources = external->batch_resources();
        // Retain the accepted prefix that supplied this shape. A compatible
        // fresh successor has no local source chunks to forecast yet.
        external_batches.source_members = resources.members;
        external_batches.source_bytes = resources.source_bytes;
    }
    void observe_shape(group_grow::Result const& result)
    {
        if (shape_recorded || !result.members || !result.final_bytes) return;
        group_grow::ConstructionSample shape;
        shape.member_bytes = static_cast<long double>(result.final_bytes) / result.members;
        shape.chunk_bytes = incremental ? static_cast<long double>(incremental->largest_chunk()) / result.members :
            2 * shape.member_bytes / PlotGroupFile::getChunkCountForK(identity().header.k);
        observe(shape, &group_grow::ConstructionHistory::shapes); shape_recorded = true;
    }
    group_grow::Identity const& identity() const
    { return external ? external->identity() : merged ? merged->identity() : staged ? staged->identity() : incremental->identity(); }
    bool is_staging() const { return bool(external) || bool(staged) || bool(merged); }
    std::string_view mode() const { return external ? "external" : merged ? "merged" : staged ? "staged" : "incremental"; }
    uint64_t selected_checkpoint_bytes() const { return std::max(config.checkpoint_bytes, external_checkpoint_bytes); }
    bool can_select() const
    { return automatic && !merge_reserve_failed && incremental && !identity().pending && (identity().header.group_size || history.shapes.count); }
    bool can_append() const { return external ? external->can_append() : merged ? merged->can_append() : !staged || staged->can_append(); }
    uint16_t member_count() const
    { return external ? external->committed_member_count() : merged ? merged->committed_member_count() : staged ? staged->committed_member_count() : identity().header.group_size; }
    uint64_t memory_bytes(uint16_t batch_members, bool compact) const
    {
        // Only terminal packing gives up ordinary admission. Fallback and
        // recovery handoffs may still reopen an ordinary or merged engine.
        if (external && terminal_packing) return external->required_memory(batch_members);
        auto envelope = config; envelope.window_bytes = std::max(config.window_bytes, window_limit);
        auto const ordinary = group_grow::Engine::memory_bytes(identity().header.k, envelope,
            batch_members, 0, compact ? batch_members : 0);
        if (external) return std::max({ordinary, external->required_memory(batch_members),
            finish_merged ? group_grow::MergeEngine::memory_bytes(identity().header.k, config, batch_members) : 0});
        if (merged) return std::max(ordinary, merged->required_memory(batch_members));
        return staged ? std::max(ordinary,
            group_grow::StagingEngine::memory_bytes(identity().header.k, config, batch_members)) : ordinary;
    }
    uint64_t memory_for(group_grow::Config const& cfg, uint16_t batch_members, bool compact, bool merging) const
    {
        if (external && terminal_packing && !merging) return external->required_memory(batch_members, cfg.workers);
        auto envelope = cfg; envelope.window_bytes = std::max(cfg.window_bytes, window_limit);
        auto const ordinary = group_grow::Engine::memory_bytes(identity().header.k, envelope,
            batch_members, 0, compact ? batch_members : 0);
        uint64_t bytes = ordinary;
        if (external) bytes = std::max(bytes, external->required_memory(batch_members, cfg.workers));
        if (merged) bytes = std::max(bytes, merged->required_memory(batch_members, cfg.workers));
        if (merging || (external && finish_merged))
            bytes = std::max(bytes, group_grow::MergeEngine::memory_bytes(identity().header.k, cfg, batch_members));
        if (staged) bytes = std::max(bytes,
            group_grow::StagingEngine::memory_bytes(identity().header.k, cfg, batch_members));
        return bytes;
    }
    void set_workers(unsigned workers, uint16_t batch_members, bool compact, uint64_t admitted_bytes)
    {
        auto cfg = config; cfg.workers = workers;
        require(workers && memory_for(cfg, batch_members, compact, false) <= admitted_bytes,
            "worker grant exceeds the destination workspace lease");
        if (external) external->set_workers(workers);
        else if (merged) merged->set_workers(workers);
        else if (staged) staged->set_workers(workers);
        else incremental->set_workers(workers);
        if (config.workers != workers)
        {
            // Timings remain comparable only within their worker context.
            history = {}; shared_history.reset(); external_batches = {}; shape_recorded = false;
        }
        config.workers = workers;
    }
    void prepare_finish()
    {
        if (!external || terminal_packing) return;
        // This owner operation finishes before the controller releases RAM or
        // starts another producer. No controller reads asynchronous phase state.
        bool const fresh = external->can_append();
        external->start_packing(); fresh_external_pack = fresh; terminal_packing = true;
    }
    void retune(group_grow::Config const& cfg)
    {
        require(incremental && !identity().pending, "I/O tuning requires an idle ordinary group");
        if (cfg.window_bytes != config.window_bytes || cfg.checkpoint_bytes != config.checkpoint_bytes ||
            cfg.merge_inputs != config.merge_inputs) ++tuning_changes;
        incremental->retune(cfg.window_bytes, cfg.checkpoint_bytes); config = cfg;
    }
    void resize_capacity(uint64_t capacity)
    {
        require(capacity >= remaining_metadata_bytes, "retired metadata exceeds the destination reservation");
        capacity -= remaining_metadata_bytes;
        require(incremental && !identity().pending && capacity >= config.capacity_bytes,
            "capacity growth requires an idle ordinary group");
        if (capacity == config.capacity_bytes) return;
        auto const saved = identity(); auto const before = snapshot(path); confirm_identity();
        completed = add_stats(completed, incremental->stats()); incremental.reset();
        bool const reopen = saved.bytes || fs::exists(file_io::native_path(path));
        if (reopen) unchanged(before, path);
        config.capacity_bytes = capacity;
        incremental = std::make_unique<group_grow::Engine>(path, config, reopen, saved);
        if (reopen) unchanged(before, path);
        same_saved(saved, identity()); confirm_identity();
    }
    void select_merged(bool report = true)
    {
        require(can_select(), "automatic merge selection needs a complete ordinary group");
        auto saved = identity();
        auto const before = snapshot(path); confirm_identity();
        completed = add_stats(completed, incremental->stats()); incremental.reset();
        // Newly created empty owners remove their placeholder on close; saved
        // empty owners retain theirs. Only reopen an unchanged retained inode.
        bool const reopen = saved.bytes || fs::exists(file_io::native_path(path));
        if (reopen) unchanged(before, path);
        merged = std::make_unique<group_grow::MergeEngine>(path, config, reopen, saved);
        if (reopen) unchanged(before, path);
        same_saved(saved, identity()); confirm_identity(); ++automatic_merges;
        if (report) std::cerr << path << ": automatic construction selected merged\n";
    }
    void confirm_identity()
    {
        if (external) { require(!external_released, "external construction was already released"); external->confirm_identity(); }
        else if (merged) merged->confirm_identity();
        else if (staged) staged->confirm_identity();
        else incremental->confirm_identity();
    }
    static group_grow::Stats add_stats(group_grow::Stats a, group_grow::Stats const& b)
    {
        a.encode_chunks = sum(a.encode_chunks, b.encode_chunks);
        a.sized_chunks = sum(a.sized_chunks, b.sized_chunks);
        a.write_calls = sum(a.write_calls, b.write_calls);
        a.checkpoints = sum(a.checkpoints, b.checkpoints);
        a.syncs = sum(a.syncs, b.syncs);
        a.overlap_windows = sum(a.overlap_windows, b.overlap_windows);
        a.read_calls = sum(a.read_calls, b.read_calls);
        a.read_bytes = sum(a.read_bytes, b.read_bytes);
        a.written_bytes = sum(a.written_bytes, b.written_bytes);
        a.read_ns = sum(a.read_ns, b.read_ns);
        a.write_ns = sum(a.write_ns, b.write_ns);
        a.sync_ns = sum(a.sync_ns, b.sync_ns);
        a.merge_passes = sum(a.merge_passes, b.merge_passes);
        a.reclaimed_bytes = sum(a.reclaimed_bytes, b.reclaimed_bytes);
        a.max_active_runs = std::max(a.max_active_runs, b.max_active_runs);
        a.codec_ns = sum(a.codec_ns, b.codec_ns); a.decode_ns = sum(a.decode_ns, b.decode_ns);
        a.layout_moves = sum(a.layout_moves, b.layout_moves); a.layout_bytes = sum(a.layout_bytes, b.layout_bytes);
        a.layout_batches = sum(a.layout_batches, b.layout_batches); a.placed_blocks = sum(a.placed_blocks, b.placed_blocks);
        a.contribution_workers = std::max(a.contribution_workers, b.contribution_workers);
        a.packing_workers = std::max(a.packing_workers, b.packing_workers);
        a.packing_overlap_windows = sum(a.packing_overlap_windows, b.packing_overlap_windows);
        a.merge_inputs = std::max(a.merge_inputs, b.merge_inputs);
        for (size_t i = 0; i < a.phases.size(); ++i)
        {
            auto& x = a.phases[i]; auto const& y = b.phases[i];
            x.elapsed_ns = sum(x.elapsed_ns, y.elapsed_ns); x.read_ns = sum(x.read_ns, y.read_ns);
            x.write_ns = sum(x.write_ns, y.write_ns); x.sync_ns = sum(x.sync_ns, y.sync_ns);
            x.codec_ns = sum(x.codec_ns, y.codec_ns); x.decode_ns = sum(x.decode_ns, y.decode_ns);
            x.read_bytes = sum(x.read_bytes, y.read_bytes); x.written_bytes = sum(x.written_bytes, y.written_bytes);
            x.read_calls = sum(x.read_calls, y.read_calls); x.write_calls = sum(x.write_calls, y.write_calls);
            x.syncs = sum(x.syncs, y.syncs);
        }
        return a;
    }
    group_grow::Stats stats() const
    { return add_stats(completed, external ? external->stats() : merged ? merged->stats() : staged ? staged->stats() : incremental->stats()); }
    void release_external()
    {
        if (!external || external_released) return;
        uint64_t peak = 0;
        for (auto bytes : external->scratch_peak_bytes()) peak = sum(peak, bytes);
        peak_scratch_bytes = std::max(peak_scratch_bytes, peak);
        scratch_cleanup_complete = external->release_completed() && scratch_cleanup_complete;
        remaining_metadata_bytes = sum(remaining_metadata_bytes, external->remaining_metadata_bytes());
        if (!scratch_cleanup_complete)
            std::cerr << path << ": preserved changed temporary contents in retired construction metadata\n";
        external_released = true;
    }
    group_grow::Result finish_staging(bool keep_merged = false)
    {
        require(is_staging(), "staging finalization requires a staged target");
        require(!terminal_packing, "terminal packing cannot reopen an ordinary engine without its workspace reservation");
        auto const result = finish();
        auto saved = identity();
        peak_file_bytes = std::max(peak_file_bytes, result.peak_file_bytes);
        workspace_bytes = std::max(workspace_bytes, result.workspace_bytes);
        auto const before = snapshot(path); confirm_identity();
        uint64_t const metadata_before = remaining_metadata_bytes;
        release_external();
        uint64_t const retained = remaining_metadata_bytes - metadata_before;
        require(retained <= config.capacity_bytes, "retired metadata exceeds the destination reservation");
        config.capacity_bytes -= retained;
        completed = add_stats(completed, external ? external->stats() : merged ? merged->stats() : staged->stats());
        external.reset(); external_released = false; external_batches = {};
        staged.reset(); merged.reset();
        bool const reopen = saved.bytes || fs::exists(file_io::native_path(path));
        if (reopen) unchanged(before, path);
        if (keep_merged) merged = std::make_unique<group_grow::MergeEngine>(path, config, reopen, saved);
        else incremental = std::make_unique<group_grow::Engine>(path, config, reopen, saved);
        if (reopen) unchanged(before, path);
        same_saved(saved, identity());
        confirm_identity();
        return result;
    }
    group_grow::Result resume(group_grow::Members members, group_grow::MemberBatchHash const* hash = nullptr)
    {
        require(bool(incremental), "staged recovery must not regenerate committed members");
        return account_metadata(incremental->resume(members, hash));
    }
    group_grow::Result account_metadata(group_grow::Result result) const
    {
        result.peak_file_bytes = sum(result.peak_file_bytes, remaining_metadata_bytes);
        result.workspace_bytes = sum(result.workspace_bytes, remaining_metadata_bytes); return result;
    }
    group_grow::AppendResult account_metadata(group_grow::AppendResult result) const
    {
        result.attempted_peak_bytes = sum(result.attempted_peak_bytes, remaining_metadata_bytes);
        if (result.accepted) result.result = account_metadata(result.result); return result;
    }
    template<class Append>
    group_grow::AppendResult append_external(Append&& append, uint16_t members, bool retained_input = true)
    {
        auto const before = external->stats();
        auto const cpu = external->contribution_cpu_time(), serial = external->contribution_serial_cpu_time();
        auto const scratch = external->scratch_service(), destination = external->destination_service();
        auto const begin = std::chrono::steady_clock::now();
        auto result = append();
        if (result.accepted)
        {
            // Prepared input adds transport reads to this phase; its writes
            // occurred before append. Do not train retained-input service rates
            // from that different operation. Its trial uses durable completion.
            if (retained_input) observe_external_contribution(before, group_grow::elapsed_ns(begin),
                members, cpu, serial, scratch, destination);
            observe_demand(cpu, external->contribution_cpu_time(), group_grow::elapsed_ns(begin), config.workers, false);
            ++staged_batches; shape_recorded = false;
        }
        return result;
    }
    group_grow::AppendResult append(group_grow::ExternalEngine::PreparedInput& input, uint16_t prefix)
    {
        require(bool(external), "prepared input requires external construction");
        return account_metadata(append_external([&] { return external->append(input, prefix); }, prefix, false));
    }
    group_grow::AppendResult append(group_grow::ExternalEngine::ContributionInput& input, uint16_t prefix)
    {
        require(bool(external), "compressed input requires external construction");
        // The remote generation/compression phase is not a local CPU sample.
        auto result = external->append(input, prefix);
        if (result.accepted) { ++staged_batches; shape_recorded = false; }
        return account_metadata(std::move(result));
    }
    group_grow::Result fallback_input()
    {
        require(bool(external), "prepared input fallback requires external construction");
        std::cerr << path << ": bounded input admission rejected; finishing committed runs before incremental admission\n";
        auto result = finish_staging(); ++fallbacks;
        if (automatic) ++automatic_fallbacks;
        merge_reserve_failed = true; return result;
    }
    group_grow::AppendResult append(group_grow::Members members,
                                    group_grow::MemberBatchHash const* hash = nullptr)
    {
        if (external)
        {
            auto result = append_external([&] { return external->append(members); }, uint16_t(members.size()));
            if (result.accepted) return account_metadata(std::move(result));
            if (members.size() > 1) return account_metadata(std::move(result));
            std::cerr << path << ": temporary reserve exhausted; finishing committed runs before incremental admission\n";
            finish_staging(); ++fallbacks;
            if (automatic) ++automatic_fallbacks;
            merge_reserve_failed = true;
        }
        if (merged)
        {
            auto const before = merged->stats(); auto const begin = std::chrono::steady_clock::now();
            auto result = merged->append(members);
            if (result.accepted)
            {
                auto const& after = merged->stats();
                auto const table = PlotGroupFile::getChunkCountForK(identity().header.k) *
                    (sizeof(group_grow::ChunkSummary) + sizeof(uint64_t));
                if (!staged_recovery) observe({group_grow::elapsed_ns(begin), after.read_ns - before.read_ns,
                    after.write_ns - before.write_ns, after.sync_ns - before.sync_ns,
                    after.read_bytes - before.read_bytes, after.written_bytes - before.written_bytes,
                    after.syncs - before.syncs, 2 * static_cast<long double>(merged->contribution_bytes()) + 2 * table,
                    unsigned(std::max<uint64_t>(1, after.contribution_workers)), after.codec_ns - before.codec_ns,
                    after.decode_ns - before.decode_ns, after.read_calls - before.read_calls,
                    after.write_calls - before.write_calls}, &group_grow::ConstructionHistory::contributions);
                ++staged_batches; shape_recorded = false; staged_recovery = false;
            }
            if (result.accepted || !automatic || members.size() > 1) return account_metadata(std::move(result));
            std::cerr << path << ": merge reserve exhausted; finishing committed runs before incremental admission\n";
            finish_staging(); ++automatic_fallbacks; merge_reserve_failed = true;
        }
        if (staged)
        {
            auto result = staged->append(members);
            if (result.accepted) { ++staged_batches; return account_metadata(std::move(result)); }
            std::cerr << "staging admission rejected: " << result.reason << "; finishing committed runs\n";
            finish_staging();
            ++fallbacks;
        }
        // The rejected batch is still immutable and resident. Retry it through
        // ordinary exact admission, including its usual smaller-batch fallback.
        auto prior = incremental->stats(); auto const old_bytes = identity().bytes;
        auto begin = std::chrono::steady_clock::now();
        group_grow::AppendResult result;
        try { result = incremental->append(members, hash); }
        catch (group_grow::WindowBudgetError const&)
        {
            if (config.window_bytes >= window_limit) throw;
            auto cfg = config; cfg.window_bytes = window_limit; retune(cfg); window_failed = true;
            prior = incremental->stats(); begin = std::chrono::steady_clock::now();
            result = incremental->append(members, hash);
        }
        result.attempted_peak_bytes = sum(result.attempted_peak_bytes, remaining_metadata_bytes);
        if (result.accepted)
        {
            auto const& after = incremental->stats();
            observe({group_grow::elapsed_ns(begin), after.read_ns - prior.read_ns,
                after.write_ns - prior.write_ns, after.sync_ns - prior.sync_ns,
                after.read_bytes - prior.read_bytes, after.written_bytes - prior.written_bytes,
                after.syncs - prior.syncs,
                group_grow::construction_codec_work(old_bytes, result.result.final_bytes, result.fresh_single_pass),
                incremental->encoding_workers(), after.codec_ns - prior.codec_ns,
                after.decode_ns - prior.decode_ns, after.read_calls - prior.read_calls,
                after.write_calls - prior.write_calls});
            shape_recorded = false;
            result.result.peak_file_bytes = std::max(sum(result.result.peak_file_bytes, remaining_metadata_bytes), peak_file_bytes);
            result.result.workspace_bytes = std::max(sum(result.result.workspace_bytes, remaining_metadata_bytes), workspace_bytes);
        }
        return result;
    }
    group_grow::Result finish()
    {
        if (external)
        {
            auto const before = external->checkpoint_tuning_changes();
            auto const cpu = external->packing_cpu_time(), serial = external->packing_serial_cpu_time();
            auto const owner = external->packing_owner_cpu_time();
            auto const scratch = external->scratch_service(), destination = external->destination_service();
            uint64_t const elapsed = external->packing_body_ns();
            bool const whole_pack = external->can_append() || (terminal_packing && fresh_external_pack);
            group_grow::CpuTime finish_cpu;
            auto const begin = std::chrono::steady_clock::now();
            group_grow::Result result;
            try { group_grow::ThreadCpuTimer timing(finish_cpu); result = external->finish(); }
            catch (...) { staged_recovery = true; fresh_external_pack = false; throw; }
            tuning_changes = sum(tuning_changes, external->checkpoint_tuning_changes() - before);
            external_checkpoint_bytes = std::max(external_checkpoint_bytes, external->packing_checkpoint_bytes());
            if (whole_pack)
            {
                if (!staged_recovery) observe_shape(result);
                group_grow::ConstructionSample sample;
                sample.elapsed_ns = group_grow::elapsed_ns(begin); sample.written_bytes = result.final_bytes;
                sample.codec_bytes = 2 * static_cast<long double>(result.final_bytes);
                sample.workers = unsigned(std::max<uint64_t>(1, external->stats().packing_workers));
                execution_cpu_sample(sample, cpu, external->packing_cpu_time(), serial, external->packing_serial_cpu_time());
                auto const owner_after = external->packing_owner_cpu_time();
                bool const measured = finish_cpu.measured && owner.measured && owner_after.measured &&
                    owner_after.nanoseconds >= owner.nanoseconds;
                uint64_t const body_owner = measured ? owner_after.nanoseconds - owner.nanoseconds : 0;
                sample.cpu_measured = sample.cpu_measured && measured && sample.cpu_ns >= body_owner;
                sample.serial_cpu_measured = sample.serial_cpu_measured && measured && sample.serial_cpu_ns >= body_owner;
                if (sample.cpu_measured) sample.cpu_ns = sum(sample.cpu_ns - body_owner, finish_cpu.nanoseconds);
                if (sample.serial_cpu_measured) sample.serial_cpu_ns = sum(sample.serial_cpu_ns - body_owner, finish_cpu.nanoseconds);
                sample.scratch_io = external->scratch_service().since(scratch);
                sample.destination_io = external->destination_service().since(destination);
                sample.io_attributed = external->service_attributed();
                observe(sample, &group_grow::ConstructionHistory::execution_packing);
            }
            // A reopened staging prefix still performs its whole pack. An
            // interrupted pack performs a suffix: report it, but do not train
            // the whole-phase worker policy or relax recovery's shape gate.
            observe_demand(cpu, external->packing_cpu_time(), external->packing_body_ns() - elapsed,
                config.workers, true, terminal_packing && fresh_external_pack);
            staged_recovery = false; return account_metadata(result);
        }
        if (merged)
        {
            auto const before = merged->stats();
            group_grow::Result result;
            try { result = merged->finish(); }
            catch (...) { staged_recovery = true; throw; }
            auto const& after = merged->stats();
            // Recovery may have completed only a suffix of packing or layout.
            // It must publish successfully, but does not train a whole-phase rate.
            if (!staged_recovery)
            {
                auto const pack = size_t(group_grow::WorkPhase::pack), layout = size_t(group_grow::WorkPhase::layout);
                observe(phase_sample(before.phases[pack], after.phases[pack],
                    2 * static_cast<long double>(result.final_bytes), unsigned(std::max<uint64_t>(1, after.packing_workers))),
                    &group_grow::ConstructionHistory::packing);
                observe(phase_sample(before.phases[layout], after.phases[layout], 0, 1),
                    &group_grow::ConstructionHistory::layout);
                observe_shape(result);
            }
            staged_recovery = false;
            return account_metadata(result);
        }
        if (staged)
        {
            auto const result = staged->finish();
            if (!staged_recovery) observe_shape(result);
            staged_recovery = false; return account_metadata(result);
        }
        auto const& saved = incremental->identity();
        group_grow::Result result{saved.bytes, std::max(peak_file_bytes, sum(saved.bytes, remaining_metadata_bytes)),
            std::max(workspace_bytes, remaining_metadata_bytes), saved.header.group_size};
        observe_shape(result); return result;
    }
};
static_assert(sizeof(Construction) + sizeof(group_grow::ExternalEngine) < mib / 16); // Existing fixed controller reserve.

struct AutomaticSelection
{
    group_grow::Config config;
    uint64_t engine_bytes = 0, fixed = 0, batch = 0;
    bool merged = false;
};
std::optional<AutomaticSelection> automatic_selection(Construction const& engine, Producer& producer,
    uint16_t maximum, uint16_t batch_cap, uint64_t reserved_engine, uint64_t other_retained,
    uint64_t observed_member_storage, bool borrow_output, uint64_t incoming_retained = 0,
    uint64_t incoming_metadata = 0)
{
    if (!engine.incremental || engine.identity().pending || engine.member_count() >= maximum ||
        (!engine.member_count() && !engine.history.shapes.count)) return {};
    bool const compact = producer.args.member_storage == "compact";
    bool const borrow = producer.can_borrow(other_retained) && borrow_output;
    uint64_t const generation = producer.next_peak(engine.member_count(), borrow), held = producer.backend.retained();
    uint64_t storage = observed_member_storage;
    if (!storage)
    {
        auto const raw = fragment_bound(producer.identity.params);
        storage = compact ? group_grow::Member::compact_bytes(uint8_t(producer.identity.params.get_k()), raw / 8) : raw;
    }
    storage = std::max<uint64_t>(1, storage);
    uint64_t const available = producer.fixture ? 0 : available_host_bytes();
    auto const rates = group_grow::construction_rates(engine.history);
    long double const generation_cost = engine.history.count && producer.backend.generated_members ?
        static_cast<long double>(producer.backend.generation_ns) / producer.backend.generated_members *
            (maximum - engine.member_count()) : 0;
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(engine.identity().header.k);
    long double const estimated_member_bytes = group_grow::construction_shape(engine.history.shapes);
    uint64_t const chunk = std::max<uint64_t>(1, engine.member_count() ? engine.incremental->largest_chunk() :
        uint64_t(std::ceil(group_grow::construction_shape(engine.history.shapes, true))));
    unsigned const observed_workers = engine.history.count ?
        std::max(1u, engine.history.samples[(engine.history.next + engine.history.samples.size() - 1) %
            engine.history.samples.size()].workers) : engine.config.workers;
    // Leave room for several chunks per worker at the expected final size.
    // Exact sizing can restore the reserved maximum if a new member is skewed.
    long double const future_chunk = std::max(static_cast<long double>(chunk),
        (engine.member_count() ? static_cast<long double>(engine.identity().bytes) / engine.member_count() :
            estimated_member_bytes) / chunks * maximum * 2);
    uint64_t const minimum_window = std::min(engine.window_limit, uint64_t(std::ceil(future_chunk)));
    AutomaticSelection best{engine.config, reserved_engine, producer.fixed, 1, false};
    long double best_cost = std::numeric_limits<long double>::infinity(), original_cost = best_cost;
    auto consider = [&](group_grow::Config cfg, bool merging, bool original)
    {
        uint64_t const engine_bytes = engine.memory_for(cfg, batch_cap, compact, merging);
        uint64_t const fixed = sum(producer.fixed - reserved_engine, engine_bytes);
        uint64_t const admitted = sum(sum(fixed - incoming_metadata, other_retained), std::max(generation, held));
        if (engine_bytes > cfg.memory_limit_bytes || admitted > producer.budget) return;
        if (sum(sum(sum(fixed, other_retained), incoming_retained), held) > producer.budget) return;
        uint64_t const extra = engine_bytes > reserved_engine ? engine_bytes - reserved_engine : 0;
        if (available && sum(extra, generation > held ? generation - held : 0) > available) return;
        uint64_t const largest = std::min({uint64_t(batch_cap), uint64_t(maximum - engine.member_count()),
            uint64_t(1) + (producer.budget - admitted) / storage});
        for (bool overlap : {false, true})
        {
            if (overlap && (!producer.overlap_enabled() || producer.backend.mode == "cpu" ||
                !generation_cost || largest < 2)) continue;
            uint64_t const batch = overlap ? std::max<uint64_t>(1, largest / 2) : largest;
            auto model = cfg;
            // Small windows cannot keep all codec workers busy. Use the known
            // source chunk distribution to price that loss of concurrency.
            model.workers = unsigned(std::max<uint64_t>(1,
                std::min({uint64_t(cfg.workers), uint64_t(observed_workers),
                    cfg.window_bytes / std::max<uint64_t>(1, chunk * 2)})));
            auto const forecast = group_grow::construction_forecast(engine.identity().header.k, model,
                engine.member_count(), engine.identity().bytes, uint16_t(batch), maximum, estimated_member_bytes);
            if (!forecast.batches || (merging && !group_grow::prefer_merged(forecast, model, engine.history))) continue;
            if (merging && model.merge_inputs == 4 && !engine.nonrotating)
            {
                auto two = model; two.merge_inputs = 2;
                auto const alternative = group_grow::construction_forecast(engine.identity().header.k, two,
                    engine.member_count(), engine.identity().bytes, uint16_t(batch), maximum, estimated_member_bytes);
                if (!group_grow::prefer_four_inputs(forecast, alternative, model, engine.history)) continue;
            }
            long double const assembly = merging ? group_grow::construction_merged_cost(forecast, model, engine.history) :
                group_grow::construction_cost(forecast.incremental, model, rates);
            long double const finish = merging ? group_grow::construction_finalization_cost(forecast, model, engine.history) : 0;
            long double const streaming = std::max(0.0L, assembly - finish);
            long double const cost = overlap ? std::max(streaming, generation_cost) + finish +
                generation_cost * batch / (maximum - engine.member_count()) + streaming / forecast.batches :
                assembly + generation_cost;
            if (original && !overlap) original_cost = cost;
            if (cost < best_cost)
            {
                best_cost = cost; best = {cfg, engine_bytes, fixed, batch, merging};
            }
        }
    };
    consider(engine.config, false, true);
    std::array<uint64_t, 5> const windows{256 * 1024, mib, 4 * mib, 16 * mib, engine.config.window_bytes};
    std::array<uint64_t, 7> const checkpoints{
        4 * mib, 16 * mib, 64 * mib, 256 * mib, 512 * mib, 1024 * mib, engine.config.checkpoint_bytes};
    for (auto window : windows) for (auto checkpoint : checkpoints)
    {
        if (window > engine.window_limit || window < minimum_window ||
            ((producer.args.window_set || engine.window_failed) && window != engine.config.window_bytes) ||
            (producer.args.checkpoint_set && checkpoint != engine.config.checkpoint_bytes)) continue;
        auto cfg = engine.config; cfg.window_bytes = window; cfg.checkpoint_bytes = checkpoint;
        // Ordinary recovery keeps its existing automatic ceiling. Merged
        // construction can price a longer bounded transaction independently
        // of its block size, with the same memory and space admission above.
        if (checkpoint <= producer.args.checkpoint_bytes) consider(cfg, false, false);
        if (engine.can_select()) for (unsigned fan : {2u, 4u})
        {
            if (fan == 4 && !group_grow::four_input_candidate(cfg, engine.nonrotating,
                engine.known_storage, engine.window_limit, engine.history.count)) continue;
            cfg.merge_inputs = fan; consider(cfg, true, false);
        }
    }
    // Budget-driven batch changes remain useful even without a protocol or
    // buffer change. All other changes need a margin over the current plan.
    if (best_cost >= original_cost * (1 - group_grow::construction_margin(engine.history.count)))
    {
        best = {engine.config, reserved_engine, producer.fixed,
            std::min({uint64_t(batch_cap), uint64_t(maximum - engine.member_count()),
                uint64_t(1) + (producer.budget - std::min(producer.budget,
                    sum(sum(producer.fixed - incoming_metadata, other_retained), std::max(generation, held)))) / storage}), false};
    }
    return best;
}

uint64_t member_storage_bound(Arguments const& args, Identity const& identity)
{
    uint64_t const raw = fragment_bound(identity.params);
    return args.member_storage == "compact" ? group_grow::Member::compact_bytes(
        uint8_t(identity.params.get_k()), size_t(raw / sizeof(uint64_t))) : raw;
}

uint16_t external_batch_size(ExternalBatchHistory& history,
    group_grow::ExternalEngine::BatchResources const& resources, Producer const& producer,
    group_grow::Config const& config, uint16_t maximum, uint16_t batch_cap, uint64_t other_retained,
    uint64_t observed_member_storage)
{
    uint16_t const remaining = maximum > resources.members ? maximum - resources.members : 0;
    uint16_t const largest = uint16_t(std::min({producer.args.batch, uint64_t(batch_cap), uint64_t(remaining)}));
    // This schedule needs independent owning compact members. Raw borrowed
    // output, unavailable GPU overlap and mandatory recovery keep their usual
    // admission. A final single batch has no next append to overlap.
    if (!producer.args.generation_overlap || !producer.overlap_enabled() || producer.backend.mode != "gpu" ||
        producer.args.member_storage != "compact" || producer.fixture || largest < 2 || remaining <= largest) return largest;
    history.prepare(largest);
    uint16_t const half = history.sizes[1].members;
    uint64_t const storage = std::max(member_storage_bound(producer.args, producer.identity), observed_member_storage);
    uint64_t const generation = std::max(producer.next_peak(resources.members), producer.backend.retained());
    uint64_t const admitted = sum(sum(producer.fixed, other_retained), generation);
    if (admitted > producer.budget) return largest;
    uint64_t const resident_slots = (producer.budget - admitted) / std::max(uint64_t(1), storage);
    auto prefetch = [&](uint16_t batch)
    { return resident_slots < batch ? uint64_t(0) : std::min(uint64_t(batch), resident_slots - batch + 1); };
    // An explicit small upper bound may already leave two full batches. Do
    // not halve it merely because the RAM admission ceiling is larger.
    if (prefetch(largest) == largest || prefetch(half) != half) return largest;
    auto const& full = history.sizes[0]; auto const& half_samples = history.sizes[1];
    if (!full.planning.count || !full.encoding.count || !producer.backend.generated_members) return largest;
    auto median = [](group_grow::ConstructionSamples const& samples, auto value)
    {
        std::array<long double, 8> values{};
        for (size_t i = 0; i < samples.count; ++i) values[i] = value(samples.samples[i]);
        std::sort(values.begin(), values.begin() + samples.count);
        return (values[(samples.count - 1) / 2] + values[samples.count / 2]) / 2;
    };
    auto elapsed = [&](group_grow::ConstructionSamples const& samples)
    { return median(samples, [](auto const& sample) { return static_cast<long double>(sample.elapsed_ns); }); };
    auto batches = [&](uint16_t batch) { return (uint64_t(remaining) + batch - 1) / batch; };
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(uint8_t(producer.identity.params.get_k()));
    uint64_t const full_batches = batches(largest), half_batches = batches(half);
    // This batch-only estimate does not invent a new compaction forecast.
    // Keep the existing batch when the estimated pre-codec reader/descriptor
    // window shrinks. Equal estimates do not guarantee identical packing cost.
    if (!resources.packet_bytes || resources.runs + half_batches > resources.readers) return largest;
    uint64_t const source_members = resources.members ? resources.members : history.source_members;
    uint64_t const source_bytes = resources.members ? resources.source_bytes : history.source_bytes;
    uint64_t const source = source_members ?
        sum(product(source_bytes, maximum), source_members - 1) / source_members : 0;
    if (!source) return largest;
    auto packing_window = [&](uint64_t count)
    {
        uint64_t const runs = resources.runs + count, packets = resources.packets + count;
        uint64_t const tasks = sizeof(group_grow::ChunkInput) + sizeof(uint64_t) + runs * sizeof(uint64_t);
        return std::min({chunks, uint64_t(65536), config.window_bytes / std::max(uint64_t(1), source),
            uint64_t(UINT16_MAX) / std::max(uint64_t(1), packets),
            std::max(uint64_t(1), config.window_bytes / tasks)});
    };
    uint64_t const full_window = packing_window(full_batches);
    if (!full_window || packing_window(half_batches) < full_window) return largest;

    // Price final reads from each candidate's observed compressed payload,
    // packet headers and per-reader preread window. Before the half trial its
    // payload is unknown; added join CPU also has no measured packing rate.
    // With no scratch read yet, construction_rates supplies its byte-rate
    // fallback. Refill latency remains unmeasured: large buffered-write waits
    // do not measure small read calls. The guard and bounded trial retain that
    // uncertainty rather than substituting write-call latency.
    auto const rates = group_grow::construction_rates(full.encoding);
    long double const header = static_cast<long double>(chunks) * resources.packet_bytes;
    auto payload = [&](ExternalBatchHistory::Samples const& size)
    {
        long double const contribution = median(size.encoding, [](auto const& sample) { return sample.codec_bytes; });
        return std::max(0.0L, contribution - header) / size.members * remaining;
    };
    long double const full_bytes = resources.bytes + payload(full) + header * full_batches;
    long double const half_bytes = resources.bytes + (half_samples.encoding.count ? payload(half_samples) : payload(full)) + header * half_batches;
    auto reads = [&](uint64_t count, long double bytes)
    {
        uint64_t const runs = resources.runs + count;
        auto const window = std::max(resources.packet_bytes, config.window_bytes / std::max(uint64_t(1), runs));
        return std::ceil(bytes / window) + runs;
    };
    long double const extra_pack = std::max(0.0L, half_bytes - full_bytes) * rates.read +
        std::max(0.0L, reads(half_batches, half_bytes) - reads(full_batches, full_bytes)) * rates.read_call;
    long double const generate = static_cast<long double>(producer.backend.generation_ns) /
        producer.backend.generated_members * remaining;
    if (!generate) return largest;
    auto cost = [&](uint16_t batch, long double planning, long double encoding, long double packing)
    {
        long double const assembly = batches(batch) * (planning + encoding);
        uint64_t const ahead = prefetch(batch);
        if (!ahead) return assembly + generate + packing;
        // Same bounded startup/drain estimate as ordinary overlap selection.
        // A maximum batch can leave room for only part of its next batch.
        long double const hidden_generation = generate * ahead / batch;
        return generate - hidden_generation + std::max(assembly, hidden_generation) +
            generate * batch / remaining + assembly / batches(batch) + packing;
    };
    long double const full_plan = elapsed(full.planning), full_encode = elapsed(full.encoding);
    long double const full_cost = cost(largest, full_plan, full_encode, 0);
    if (!half_samples.planning.count || !half_samples.encoding.count)
    {
        if (history.half_attempted) return largest;
        long double const cpu = median(full.planning, [](auto const& sample)
        {
            return std::max(0.0L, static_cast<long double>(sample.elapsed_ns) -
                sample.read_ns - sample.write_ns - sample.sync_ns);
        });
        long double const ratio = static_cast<long double>(half) / largest;
        // One batch size cannot identify both fixed scan overhead and work
        // per member. Bound that serial residual rather than calling it codec
        // throughput. Even the optimistic estimate must justify one trial.
        long double const optimistic = cost(half, full_plan - cpu + cpu * ratio,
            full_encode * ratio, extra_pack);
        long double const conservative = cost(half, full_plan, full_encode * ratio, extra_pack);
        long double const margin = group_grow::construction_margin(full.planning.count);
        if (optimistic >= full_cost * (1 - margin) || conservative > full_cost * (1 + margin)) return largest;
        // Remember the trial at selection, so an unaccepted/short batch does
        // not repeatedly probe. Its result must still beat the margin below.
        history.half_attempted = true; return half;
    }
    long double const half_cost = cost(half, elapsed(half_samples.planning), elapsed(half_samples.encoding), extra_pack);
    long double const margin = group_grow::construction_margin(std::min(full.planning.count, half_samples.planning.count));
    // Retain the incumbent across small timing changes in either direction.
    if (history.selected == half)
    {
        if (full_cost < half_cost * (1 - margin)) history.selected = largest;
    }
    else if (half_cost < full_cost * (1 - margin)) history.selected = half;
    return history.selected;
}

// Sum the existing fresh-chunk bound before any members are generated. The
// pinned FSE bound is n + floor(n/128) plus a fixed per-chunk overhead. Summing
// its variable part once is conservative; low-bit rounding is per chunk.
uint64_t canonical_bound(Identity const& identity, uint16_t members)
{
    require(members, "canonical bound requires a member");
    auto const k = uint8_t(identity.params.get_k());
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    uint64_t const symbols = product(fragment_bound(identity.params) / sizeof(uint64_t), members);
    uint64_t const unary = (product(members, 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS)) - 1) /
        (178ull << (k - 8));
    uint64_t const bits = sum(product(symbols, k - 7), product(chunks, unary));
    uint64_t const ans = sum(sum(symbols, symbols / 128), product(chunks, POS2_FSE_compressBound(0)));
    return sum(sizeof(group_grow::Header) + identity.memo.size(),
        sum(sum(bits / 8 + chunks, ans), product(chunks, 18)));
}

uint64_t temporary_output_capacity(Arguments const& args, Identity const& identity,
                                   uint16_t maximum, uint64_t capacity)
{
    // A finite membership limit need not reserve the whole filesystem. All
    // encoded bytes are bounded above, and the admitted RAM also bounds the
    // checkpoint tables and ordinary handoff workspace. Exact engine and
    // physical allocation admission remain authoritative.
    return std::min(capacity, sum(canonical_bound(identity, maximum), args.max_ram));
}

std::optional<group_grow::ExternalConfig> temporary_grant(group_grow::ScratchPool& pool,
    uint64_t owner, fs::path const& directory, Identity const& identity, uint16_t maximum, uint16_t batch,
    group_grow::Config& config, bool automatic)
{
    uint64_t const batches = (uint64_t(maximum) + batch - 1) / batch;
    uint64_t const run = sum(canonical_bound(identity, batch),
        product(PlotGroupFile::getChunkCountForK(uint8_t(identity.params.get_k())), 8));
    // External admission can reduce the eventual batch. One run per member
    // bounds the reader set without relying on that provisional batch size.
    unsigned const packing_inputs = unsigned(std::min<uint64_t>(64, std::max<uint64_t>(4, maximum)));
    auto attempt = [&]() -> std::optional<group_grow::ExternalConfig>
    {
        uint64_t const segment = std::max(mib, group_grow::MergeEngine::block_bytes(config));
        uint64_t const desired = product(2, std::min(config.capacity_bytes, product(run, batches)));
        uint64_t const minimum = std::min(desired, sum(canonical_bound(identity, 1),
            product(PlotGroupFile::getChunkCountForK(uint8_t(identity.params.get_k())), 8)));
        if (!minimum) return {};
        auto grant = pool.grant(owner, desired, minimum, segment, 4, automatic, packing_inputs);
        if (grant && group_grow::ExternalEngine::metadata_bytes(uint8_t(identity.params.get_k()), *grant) +
                sizeof(group_grow::Header) + identity.memo.size() > config.capacity_bytes)
        { pool.release_scratch_after_cleanup(owner); grant.reset(); }
        return grant;
    };
    if (auto grant = attempt()) return grant;
    // On a shared filesystem leave room for the retained runs and a merging
    // replacement as well as the final output. This is a fresh reservation;
    // saved quotas never shrink. Ordinary handoff restores available capacity.
    uint64_t const original = config.capacity_bytes;
    uint64_t const reduced = original / 3;
    if (!reduced || !pool.try_reserve_output(owner, directory, reduced)) return {};
    config.capacity_bytes = reduced;
    if (auto grant = attempt()) return grant;
    require(pool.try_reserve_output(owner, directory, original), "cannot restore an unused destination reservation");
    config.capacity_bytes = original; return {};
}

uint64_t allocated_bytes(fs::path const& path)
{
    file_io::Stat value{};
    if (file_io::inspect_link(path.c_str(), &value))
    { require(errno == ENOENT, "cannot inspect destination allocation"); return 0; }
    require(file_io::is_regular(value.st_mode) && value.st_blocks >= 0, "invalid destination allocation");
    return product(uint64_t(value.st_blocks), 512);
}

// Retained members and their descriptors grow together. Resolve the admitted
// batch bound before reserving the engine's per-member tables; exact producer
// and engine admission still handle unusually large members and recovery.
template<class Memory>
uint16_t admitted_batch_cap(uint16_t maximum, uint16_t minimum, uint64_t room,
                           uint64_t member_bytes, Memory memory)
{
    auto fits = [&](uint16_t count)
    {
        uint64_t const bytes = memory(count);
        return bytes <= room && (!count || uint64_t(count - 1) <= (room - bytes) / std::max(uint64_t(1), member_bytes));
    };
    if (!fits(minimum)) return 0;
    uint32_t low = minimum, high = std::max(maximum, minimum);
    while (low < high)
    {
        uint16_t const middle = uint16_t(low + (high - low + 1) / 2);
        if (fits(middle)) low = middle;
        else high = middle - 1;
    }
    return uint16_t(low);
}

template<class Memory>
uint16_t successor_batch_cap(uint16_t maximum, uint64_t room, uint64_t released_bytes,
    uint64_t member_bytes, Memory memory)
{
    uint16_t const actual = admitted_batch_cap(maximum, 1, room, member_bytes, memory);
    if (!actual) return 0;
    // Keep the pre-transition working-batch ceiling while returning unused RAM
    // to other destinations. A newly fitting singleton still uses live admission.
    uint16_t const reference = admitted_batch_cap(maximum, 1,
        room - std::min(room, released_bytes), member_bytes, memory);
    return std::min(actual, std::max(uint16_t(1), reference));
}

struct DestinationPlan
{
    size_t active = 0;
    unsigned workers = 0;
    uint16_t batch = 0;
};
template<class Memory, class Cost>
DestinationPlan destination_plan(uint64_t budget, uint64_t fixed, uint64_t generation,
    uint64_t member_bytes, size_t drive_groups, unsigned workers, uint16_t maximum,
    uint16_t minimum, Memory memory, Cost cost, size_t active_limit = 0)
{
    DestinationPlan best;
    if (fixed >= budget || generation > budget - fixed) return best;
    uint64_t const room = budget - fixed - generation;
    long double best_cost = std::numeric_limits<long double>::infinity();
    for (size_t active = 1; active <= std::min(drive_groups, active_limit ? active_limit : size_t(workers)); ++active)
    {
        unsigned share = std::max(1u, unsigned(workers / active));
        // The next producer peak includes its own returned member. Every
        // other active group may retain a whole batch while that producer runs.
        if (active - 1 > room / std::max(uint64_t(1), member_bytes)) continue;
        uint64_t const queued = uint64_t(active - 1) * member_bytes;
        uint16_t batch = 0;
        for (;;)
        {
            batch = admitted_batch_cap(maximum, minimum, (room - queued) / active,
                member_bytes, [&](uint16_t count) { return memory(share, count); });
            if (batch || share == 1) break;
            --share;
        }
        if (!batch) continue;
        long double const projected = cost(active, share, batch);
        if (projected < best_cost)
        { best = {active, share, batch}; best_cost = projected; }
    }
    return best;
}

uint64_t prefetch_memory(Arguments const& args, Identity const& identity, group_grow::Config config,
    uint16_t count, bool nonrotating, std::optional<group_grow::ExternalConfig> const& scratch = {})
{
    // Match Construction's drive-specific reader envelope before opening it.
    if (nonrotating && config.window_bytes >= 256 * 1024) config.merge_inputs = 4;
    uint8_t const k = uint8_t(identity.params.get_k());
    uint64_t bytes = group_grow::Engine::memory_bytes(k, config, count, 0, count);
    if (args.construction == "merged")
        bytes = std::max(bytes, group_grow::MergeEngine::memory_bytes(k, config, count));
    if (args.construction == "staged")
        bytes = std::max(bytes, group_grow::StagingEngine::memory_bytes(k, config, count));
    if (scratch) bytes = std::max(bytes, group_grow::ExternalEngine::memory_bytes(k, config, *scratch, count));
    return sum(bytes, sum(product(count, 2 * sizeof(group_grow::Member)), batch_hash_bytes(args, count)));
}

// A successor owns its own recoverable external job and exact resource grant.
// It can stage several bounded batches while the preceding final pack runs;
// final packing always returns to the ordinary destination writer.
struct GroupSuccessor
{
    fs::path path;
    Identity requested;
    uint16_t maximum = 0, batch_cap = 0;
    uint64_t capacity = 0, fixed = 0, engine_bytes = 0, owner = 0, pool_growth = 0, metadata_bytes = 0;
    unsigned workers = 0;
    bool deferred = false, detached = false;
    std::optional<group_grow::ExternalConfig> scratch;
    std::optional<group_grow::ExternalEngine::Inspection> recovery;
    Output guard;
    std::optional<group_grow::ScratchPool::IoLease> io;
    std::unique_ptr<Construction> engine;
    Metrics metrics;

    GroupSuccessor(fs::path name, Identity identity, uint16_t count, uint64_t room)
        : path(std::move(name)), requested(std::move(identity)), maximum(count), capacity(room) {}
    void close()
    {
        if (engine)
        {
            static_cast<group_grow::Stats&>(metrics) = engine->stats();
            metrics.staged_batches = engine->staged_batches;
            engine.reset();
        }
        io.reset();
    }
};

// Keep the pending job and its capacity grant when a lane stops. Only its
// joined live consumer stops occupying a construction slot and workspace.
Metrics detach_successor(GroupSuccessor& next, uint64_t& fixed, size_t& active,
                         std::optional<uint64_t>& successor_ordinal)
{
    require(!next.detached && !next.io && !next.workers,
        "successor detached before its staging task joined or more than once");
    uint64_t scratch_peak = 0;
    if (next.engine)
    {
        for (auto bytes : next.engine->external->scratch_peak_bytes()) scratch_peak = sum(scratch_peak, bytes);
        next.close();
        fixed -= next.engine_bytes; --active; next.engine_bytes = 0;
    }
    else if (next.recovery)
    {
        for (auto bytes : next.recovery->peak_reserved_bytes) scratch_peak = sum(scratch_peak, bytes);
        next.metrics.recovered = true;
    }
    next.metrics.members = next.guard.pending_members;
    next.metrics.final_bytes = next.guard.bytes;
    next.metrics.peak_file_bytes = std::max(next.metrics.peak_file_bytes, next.guard.charge());
    next.metrics.workspace_bytes = std::max(next.metrics.workspace_bytes, next.guard.auxiliary);
    next.metrics.scratch_peak_bytes = std::max(next.metrics.scratch_peak_bytes, scratch_peak);
    next.metrics.scratch_released = false; next.metrics.construction_selected = "external";
    next.detached = true; successor_ordinal.reset();
    return next.metrics;
}

bool prepare_successor(GroupSuccessor& next, Arguments const& args, Backend& backend,
    group_grow::ScratchPool& pool, uint64_t owner, uint64_t current_fixed, uint64_t retained,
    unsigned workers, std::shared_ptr<std::recursive_mutex> const& gate, Fixtures const* fixture = nullptr,
    group_grow::ConstructionHistories* histories = nullptr,
    std::shared_ptr<group_grow::SharedConstructionHistory> const& demand = {}, uint64_t released_bytes = 0,
    std::shared_ptr<group_grow::CodecScheduler> const& cpu_scheduler = {})
{
    if (!workers || !next.maximum || !next.capacity ||
        (args.construction != "auto" && args.construction != "external") ||
        !group_prefetch_enabled(args, backend)) return false;
    file_io::Stat target{};
    require(file_io::inspect_link(next.path.c_str(), &target) && errno == ENOENT, "successor destination appeared during the run");
    auto const pool_before = pool.memory_bytes();
    uint64_t const capacity = temporary_output_capacity(args, next.requested, next.maximum, next.capacity);
    bool reserved = pool.try_reserve_output(owner, next.path.parent_path(), capacity);
    if (!reserved)
    {
        pool.cancel(owner); next.pool_growth = pool.memory_bytes() - pool_before; return false;
    }
    auto cancel = [&]
    {
        next.io.reset(); next.scratch.reset(); pool.cancel(owner);
        next.pool_growth = pool.memory_bytes() - pool_before;
        return false;
    };
    auto options = args; options.workers = workers;
    auto cfg = configuration(options, capacity, current_fixed, cpu_scheduler, backend.remote_encoder());
    next.scratch = temporary_grant(pool, owner, next.path.parent_path(), next.requested, next.maximum,
        uint16_t(std::min(args.batch, uint64_t(next.maximum))), cfg, false);
    if (!next.scratch) return cancel();
    next.scratch->target_gate = gate;
    next.io = pool.try_stage_io(owner);
    if (!next.io) return cancel();
    next.metadata_bytes = sum(sizeof(GroupSuccessor),
        sum(next.path.native().capacity() + 1, next.requested.memo.capacity()));
    uint64_t const metadata = sum(sum(pool.memory_bytes() - pool_before, external_metadata(*next.scratch)),
        next.metadata_bytes);
    if (metadata >= args.max_ram - std::min(args.max_ram, current_fixed)) return cancel();
    uint64_t const base = sum(current_fixed, metadata);
    cfg = configuration(options, cfg.capacity_bytes, base, cpu_scheduler, backend.remote_encoder());
    auto memory = [&](uint16_t count)
    { return prefetch_memory(args, next.requested, cfg, count,
        group_grow::storage_device(next.path.parent_path()).nonrotating, next.scratch); };
    Producer probe{next.requested, fixture, backend, args, args.max_ram, sum(base, memory(1)), 0};
    if (!probe.fits(0, retained) || !group_prefetch_enabled(args, backend)) return cancel();
    uint64_t const generation = std::max(probe.next_peak(0), backend.retained());
    uint64_t const occupied = sum(sum(base, retained), generation);
    if (occupied >= args.max_ram) return cancel();
    uint16_t const maximum = uint16_t(std::min(args.batch, uint64_t(next.maximum)));
    uint64_t const headroom = args.max_ram - occupied, storage = member_storage_bound(args, next.requested);
    uint16_t const batch = successor_batch_cap(maximum, headroom, released_bytes, storage, memory);
    if (!batch) return cancel();
    uint64_t const workspace = memory(batch);
    uint64_t const available = available_host_bytes();
    if (!fixture && available && workspace > available) return cancel();
    auto engine = std::make_unique<Construction>(file_io::path_text(next.path), cfg, false,
        initial_identity(next.requested), "external", next.scratch);
    engine->demand_history = construction_history(histories, args, next.path, engine->identity(),
        args.workers ? unsigned(args.workers) : std::max(1u, std::thread::hardware_concurrency()), false, &*next.scratch, demand);
    engine->attach_history(engine->demand_history);
    uint64_t const actual = sum(engine->memory_bytes(batch, true),
        sum(product(batch, 2 * sizeof(group_grow::Member)), batch_hash_bytes(args, batch)));
    require(actual <= workspace, "successor workspace exceeds its admitted envelope");
    engine->confirm_identity();
    next.owner = owner; next.workers = workers; next.batch_cap = batch;
    next.pool_growth = pool.memory_bytes() - pool_before;
    next.fixed = sum(base, actual); next.engine_bytes = actual; next.capacity = cfg.capacity_bytes;
    next.metrics.capacity_bytes = cfg.capacity_bytes;
    next.metrics.peak_file_bytes = engine->identity().pending_peak_bytes;
    next.metrics.engine_reserved_bytes = next.fixed;
    next.metrics.peak_modeled_bytes = sum(sum(next.fixed, retained), generation);
    next.engine = std::move(engine);
    return true;
}

uint32_t stage_successor(GroupSuccessor& next, Arguments const& args, Backend& backend,
    uint64_t retained, std::function<bool()> const& stop, Fixtures const* fixture = nullptr)
{
    require(next.engine && next.engine->external && next.io, "successor staging has no admitted owner");
    ControllerTiming timing("stage_successor", next.owner);
    Producer producer{next.requested, fixture, backend, args, args.max_ram, next.fixed, next.fixed};
    std::vector<group_grow::Member> prefetched;
    uint64_t observed_member_storage = 0;
    auto const source = next.engine->demand_history;
    if (source)
    {
        std::lock_guard lock(source->mutex);
        next.engine->external_batches = source->successor_batch_workers == next.workers ?
            source->successor_batches : ExternalBatchHistory{};
    }
    auto retain_batches = [&]
    {
        if (!source) return;
        std::lock_guard lock(source->mutex);
        source->successor_batches = next.engine->external_batches;
        source->successor_batch_workers = next.workers;
    };
    bool stopping = false;
    auto stop_generation = [&] { return stopping || (stopping = stop()); };
    auto join_consumer = [&](auto& pending, uint32_t reasons)
    {
        if (pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) return pending.get();
        reasons |= wait_bit(ControllerWait::consumer);
        auto const begin = std::chrono::steady_clock::now();
        ControllerTiming timing("wait", next.owner, reasons);
        auto result = pending.get();
        next.metrics.controller_waits.add(reasons, group_grow::elapsed_ns(begin));
        return result;
    };
    uint32_t reason = 0;
    // Drain owning members before returning the stage lease, including when
    // the preceding pack finishes during their generation.
    while (next.metrics.members < next.maximum && (!prefetched.empty() || !stop_generation()))
    {
        // Other destinations may have consumed transient headroom since the
        // successor was admitted. Keep its durable prefix and retry later.
        if (prefetched.empty() && !producer.fits(uint16_t(next.metrics.members), retained))
        { reason = wait_bit(ControllerWait::producer_memory); break; }
        uint16_t const selected_batch = external_batch_size(next.engine->external_batches,
            next.engine->external->batch_resources(), producer, next.engine->config,
            next.maximum, next.batch_cap, retained, observed_member_storage);
        retain_batches(); // A rejected or interrupted trial must not repeat.
        auto members = std::move(prefetched);
        prefetched = std::vector<group_grow::Member>{};
        // A completed observation may shrink the next batch target. Retain an
        // already generated owning prefix, including when packing has stopped.
        size_t const wanted = std::max(members.size(), size_t(selected_batch));
        members.reserve(wanted);
        require(members.capacity() <= next.batch_cap, "successor member container exceeds admitted capacity");
        if (!stop_generation())
            next.metrics.generated_members = sum(next.metrics.generated_members,
                producer.fill(members, uint16_t(next.metrics.members), wanted, retained, nullptr, nullptr,
                    file_io::path_text(next.path) + ": staging next group member ", stop_generation));
        if (members.empty()) { reason = fill_wait_reason(producer.fill_stop); break; }
        next.metrics.queued_peak_bytes = std::max(next.metrics.queued_peak_bytes, resident(members));
        group_grow::AppendResult result;
        for (;;)
        {
            auto const external_before = next.engine->external->stats();
            // Stage leases never authorize the ordinary Construction fallback,
            // which may finish the target when a singleton exhausts scratch.
            auto append = [&]
            {
                auto const before = next.engine->external->stats();
                auto const cpu = next.engine->external->contribution_cpu_time(), serial = next.engine->external->contribution_serial_cpu_time();
                auto const scratch = next.engine->external->scratch_service(), destination = next.engine->external->destination_service();
                auto const begin = std::chrono::steady_clock::now();
                auto result = next.engine->external->append(members);
                if (result.accepted)
                {
                    next.engine->observe_external_contribution(before, group_grow::elapsed_ns(begin), uint16_t(members.size()),
                        cpu, serial, scratch, destination);
                    next.engine->observe_demand(cpu, next.engine->external->contribution_cpu_time(),
                        group_grow::elapsed_ns(begin), next.workers, false);
                }
                return result;
            };
            uint64_t const first = next.metrics.members + members.size();
            if (!stop_generation() && first < next.maximum && group_prefetch_enabled(args, backend) &&
                producer.fits(uint16_t(first), sum(retained, resident(members))) && group_prefetch_enabled(args, backend))
            {
                // The CPU consumer borrows this immutable batch. GPU use stays
                // on its constructing thread, and the future drains on unwind.
                auto pending = std::async(std::launch::async, append);
                size_t const count = size_t(std::min<uint64_t>(selected_batch, next.maximum - first));
                prefetched.reserve(count);
                require(prefetched.capacity() <= next.batch_cap, "successor member container exceeds admitted capacity");
                next.metrics.generated_members = sum(next.metrics.generated_members,
                    producer.fill(prefetched, uint16_t(first), count, sum(retained, resident(members)), nullptr, nullptr,
                        file_io::path_text(next.path) + ": staging next group member ", [&]
                        { return stop_generation() || pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }));
                next.metrics.queued_peak_bytes = std::max(next.metrics.queued_peak_bytes,
                    sum(resident(members), resident(prefetched)));
                result = join_consumer(pending, fill_wait_reason(producer.fill_stop));
            }
            else result = append();
            // The CPU consumer owns the phase counters until it has joined.
            // Rejected or failed candidates do not train either batch size.
            if (result.accepted)
            {
                next.engine->observe_external(uint16_t(members.size()), external_before);
                retain_batches();
            }
            if (result.accepted || members.size() == 1) break;
            prefetched.clear(); // Rejected candidates change the next member index.
            members.pop_back();
        }
        if (!result.accepted) { next.deferred = true; reason = wait_bit(ControllerWait::scratch_capacity); break; }
        observed_member_storage = (resident(members) + members.size() - 1) / members.size();
        ++next.engine->staged_batches; ++next.metrics.batches;
        next.metrics.members = result.result.members; next.metrics.final_bytes = result.result.final_bytes;
        next.metrics.peak_file_bytes = std::max(next.metrics.peak_file_bytes, result.result.peak_file_bytes);
        next.metrics.workspace_bytes = std::max(next.metrics.workspace_bytes, result.result.workspace_bytes);
        next.metrics.max_batch_members = std::max<uint64_t>(next.metrics.max_batch_members, members.size());
        next.metrics.queued_peak_bytes = std::max(next.metrics.queued_peak_bytes, resident(members));
        ++next.metrics.generation_overlap_batches;
    }
    next.metrics.peak_modeled_bytes = std::max(next.metrics.peak_modeled_bytes, producer.peak);
    next.metrics.generation_overlap_members = sum(next.metrics.generation_overlap_members, producer.overlap_members);
    next.metrics.record_generation(backend.mode, producer.generation_peak);
    return next.metrics.members == next.maximum ? wait_bit(ControllerWait::group_full) : reason;
}

Metrics run_group(Arguments const& args, fs::path const& target, Fixtures const* fixture,
                  std::optional<Identity> const& expected, uint16_t maximum, uint64_t capacity,
                  uint64_t external_bytes, std::unique_ptr<Backend>& backend, group_grow::ConstructionHistories* histories = nullptr,
                  std::optional<group_grow::ExternalEngine::Inspection> recovery = {}, Output const& auxiliary = {},
                  GroupPrefetch* inherited = nullptr,
                  std::function<uint32_t(uint64_t, uint16_t, std::future<group_grow::Result>&)> const& prefetch_next = {},
                  GroupSuccessor* successor = nullptr,
                  std::shared_ptr<std::recursive_mutex> const& target_gate = {})
{
    bool const compact = args.member_storage == "compact";
    unsigned const cpu_workers = args.workers ? unsigned(args.workers) : std::max(1u, std::thread::hardware_concurrency());
    external_bytes = sum(external_bytes, group_grow::CodecScheduler::memory_bytes(cpu_workers));
    require(external_bytes < args.max_ram, "codec scheduler metadata exceeds --max-ram");
    if (backend && args.pool_broker.empty()) backend->cpu_scheduler.reset();
    auto cpu_scheduler = backend && backend->cpu_scheduler ? backend->cpu_scheduler :
        std::make_shared<group_grow::CodecScheduler>(cpu_workers);
    if (!backend) backend = std::make_unique<Backend>();
    backend->cpu_scheduler = cpu_scheduler;
    backend->configure_network(args);
    uint64_t inherited_bytes = inherited ? sum(inherited->resident_bytes(), inherited->metadata_bytes) : 0;
    uint64_t inherited_peak = 0;
    if (inherited && !inherited->members.empty())
    {
        require(expected && inherited->identity && !args.resume, "next group batch requires a fresh identified group");
        same_identity(*expected, *inherited->identity);
    }
    require(auxiliary.retired <= capacity, "retired metadata exceeds the destination reservation");
    uint64_t capacity_limit = capacity;
    auto initial = expected ? std::optional(initial_identity(*expected)) : std::nullopt;
    external_bytes = sum(external_bytes, path_metadata(args.temporary));
    if (!recovery && group_grow::ExternalEngine::pending(file_io::path_text(target)))
    {
        require(args.resume, "external construction exists; use --resume");
        recovery = group_grow::ExternalEngine::inspect(file_io::path_text(target), args.max_ram - std::min(args.max_ram, external_bytes));
    }
    std::unique_ptr<group_grow::ScratchPool> scratch_pool;
    struct SuccessorScope
    {
        GroupSuccessor* value;
        ~SuccessorScope()
        {
            // The caller retains the record, but leases borrow this local pool.
            // Join the later-declared final writer before destroying its peer.
            if (value) { value->engine.reset(); value->io.reset(); }
        }
    } successor_scope{successor};
    std::optional<group_grow::ExternalConfig> scratch;
    auto roots = args.temporary;
    if (args.destination_temp != "off")
        add_destination_temporary(roots, args.destination_temp, target.parent_path(),
            group_grow::storage_device(target.parent_path()));
    size_t const fresh_roots = roots.size();
    require(args.resume || args.construction != "external" || fresh_roots,
            "fresh external construction requires --temp-dir or an eligible --destination-temp directory");
    if (args.temporary.empty() && recovery)
        for (auto const& grant : recovery->external.scratch)
            if (std::find(roots.begin(), roots.end(), file_io::native_path(grant.directory)) == roots.end())
                roots.emplace_back(file_io::native_path(grant.directory));
    if (successor && successor->recovery && args.temporary.empty())
        for (auto const& grant : successor->recovery->external.scratch)
            if (std::find(roots.begin(), roots.end(), file_io::native_path(grant.directory)) == roots.end())
                roots.emplace_back(file_io::native_path(grant.directory));
    external_bytes = sum(external_bytes, sum(path_metadata(roots), recovery ? external_metadata(*recovery) : 0));
    if (successor && successor->recovery)
        external_bytes = sum(external_bytes, external_metadata(*successor->recovery));
    if (!roots.empty())
    {
        scratch_pool = std::make_unique<group_grow::ScratchPool>(roots, args.headroom, args.temporary_capacity);
        preserve_retired(*scratch_pool, target, args.max_ram - std::min(args.max_ram, external_bytes), args.group_members != 0);
        if (initial || recovery)
            capacity_limit = sum(auxiliary.retired, temporary_output_capacity(args,
                identity_from(initial ? *initial : recovery->identity), maximum, capacity - auxiliary.retired));
        uint64_t const allocated = sum(auxiliary.retired_allocated, recovery ?
            sum(recovery->target_allocated_bytes, recovery->metadata_allocated_bytes) : allocated_bytes(target));
        uint64_t const minimum = std::max(allocated, sum(auxiliary.retired, recovery ? recovery->peak_bytes : 0));
        require(scratch_pool->try_reserve_output(0, target.parent_path(), minimum, allocated, args.resume),
            "destination and temporary reservations exceed available capacity");
        if (recovery)
        {
            require(sum(recovery->peak_bytes, auxiliary.retired) <= capacity_limit,
                "saved external output and metadata exceed capacity");
            scratch_pool->restore(0, recovery->external, recovery->allocated_bytes);
            scratch = recovery->external;
        }
        if (successor && successor->recovery)
        {
            auto const& saved_next = *successor->recovery;
            uint64_t const allocated_next = sum(successor->guard.retired_allocated,
                sum(saved_next.target_allocated_bytes, saved_next.metadata_allocated_bytes));
            uint64_t const minimum_next = std::max(allocated_next,
                sum(successor->guard.retired, saved_next.peak_bytes));
            require(scratch_pool->try_reserve_output(1, successor->path.parent_path(), minimum_next, allocated_next, true),
                "saved successor output reservation exceeds available capacity");
            scratch_pool->restore(1, saved_next.external, saved_next.allocated_bytes);
        }
        capacity = std::min(capacity_limit, scratch_pool->available_for_output(0, target.parent_path()));
        require(capacity >= minimum && scratch_pool->try_reserve_output(0, target.parent_path(), capacity),
            "saved output and scratch exceed available capacity");
        scratch_pool->restrict_grants(std::span(roots).first(fresh_roots));
        scratch_pool->seal_recovery();
        external_bytes = sum(external_bytes, scratch_pool->memory_bytes());
    }
    if (scratch) external_bytes = sum(external_bytes, external_metadata(*scratch));
    auto const history_bytes = histories ? histories->memory_bytes() : 0;
    external_bytes = sum(external_bytes, sizeof(group_grow::ConstructionHistory));
    uint64_t caller_fixed = sum(sum(sum(sum(fixture ? fixture->bytes : 0, external_bytes), mib),
        2 * sizeof(group_grow::Member)), batch_hash_bytes(args, 1));
    auto config = configuration(args, capacity, caller_fixed, cpu_scheduler, backend->remote_encoder());
    if (fresh_roots && scratch_pool && !args.resume && initial && (args.construction == "auto" || args.construction == "external"))
    {
        if (!backend) backend = std::make_unique<Backend>();
        auto const identity = identity_from(*initial);
        uint64_t const ordinary = group_grow::Engine::memory_bytes(initial->header.k, config, 1, 0, compact ? 1 : 0);
        Producer probe{identity, fixture, *backend, args, args.max_ram, sum(caller_fixed, ordinary), 0};
        bool const borrow = probe.can_borrow();
        uint64_t const generation = std::max(probe.next_peak(0, borrow), backend->retained());
        uint16_t const batch = admitted_batch_cap(uint16_t(std::min(args.batch, uint64_t(maximum))), 1,
            args.max_ram - std::min(args.max_ram, sum(caller_fixed, generation)), member_storage_bound(args, identity),
            [&](uint16_t n) { return group_grow::Engine::memory_bytes(initial->header.k, config, n, 0, compact ? n : 0); });
        require(batch, "member producer and engine workspace exceed --max-ram");
        if (args.construction == "external" || batch < maximum)
        {
            uint64_t const before = scratch_pool->memory_bytes();
            uint64_t const prior_capacity = capacity;
            scratch = temporary_grant(*scratch_pool, 0, target.parent_path(), identity, maximum, batch, config,
                args.construction == "auto");
            capacity = config.capacity_bytes;
            external_bytes = sum(external_bytes, scratch_pool->memory_bytes() - before);
            caller_fixed = sum(caller_fixed, scratch_pool->memory_bytes() - before);
            config = configuration(args, capacity, caller_fixed, cpu_scheduler, backend->remote_encoder());
            bool admitted = true;
            if (scratch)
            {
                probe.fixed = sum(caller_fixed, std::max(ordinary,
                    group_grow::ExternalEngine::memory_bytes(initial->header.k, config, *scratch)));
                bool borrowed = probe.can_borrow();
                admitted = probe.fits(0, 0, borrowed);
                if (!admitted && !borrowed && probe.can_borrow()) admitted = probe.fits(0, 0, true);
            }
            if (!admitted)
            {
                require(args.construction != "external", "external construction and producer exceed --max-ram");
                scratch_pool->release_scratch_after_cleanup(0); scratch.reset();
                require(scratch_pool->try_reserve_output(0, target.parent_path(), prior_capacity),
                    "cannot restore an unused destination reservation");
                capacity = prior_capacity; config = configuration(args, capacity, caller_fixed, cpu_scheduler, backend->remote_encoder());
            }
        }
        require(scratch || args.construction != "external", "no temporary capacity available for external construction");
        if (scratch && !args.quiet) std::cerr << target << ": selected external contribution storage\n";
    }
    if (inherited && !inherited->members.empty())
    {
        // Construction retains the owning batch but generates no new member.
        uint64_t const workspace = prefetch_memory(args, *expected, config, 1,
            group_grow::storage_device(target.parent_path()).nonrotating, scratch);
        uint64_t const construction = sum(sum(sum(caller_fixed, inherited_bytes), backend->retained()), workspace);
        uint64_t const available = available_host_bytes();
        if (construction > args.max_ram || (available && workspace > available))
        { inherited->clear(); inherited_bytes = 0; }
        else inherited_peak = construction;
    }
    bool scratch_live = scratch.has_value();
    if (scratch) scratch->target_gate = target_gate;
    // Scratch selection is part of an external phase's physical context.
    // Resolve it before sharing completed CPU or storage observations.
    auto source = initial ? construction_history(histories, args, target, *initial,
        args.workers ? unsigned(args.workers) : std::max(1u, std::thread::hardware_concurrency()), true,
        scratch ? &*scratch : nullptr) : nullptr;
    uint64_t const history_growth = histories ? histories->memory_bytes() - history_bytes : 0;
    external_bytes = sum(external_bytes, history_growth); caller_fixed = sum(caller_fixed, history_growth);
    config = configuration(args, capacity, caller_fixed, cpu_scheduler, backend->remote_encoder());
    Construction engine(file_io::path_text(target), config, args.resume, std::move(initial), args.construction,
        std::move(scratch), auxiliary.retired);
    if (engine.external && !args.checkpoint_set)
        engine.external->tune_packing_checkpoints(group_grow::PackingCheckpointHistory::maximum_bytes);
    engine.attach_history(std::move(source));
    auto const saved = engine.identity();
    require(saved.bytes <= engine.config.capacity_bytes, "existing group exceeds destination capacity");
    require(!saved.pending || saved.pending_peak_bytes <= engine.config.capacity_bytes,
            "pending transaction exceeds destination capacity; restore its original capacity");
    Identity const identity = identity_from(saved);
    requested_identity(args, identity);
    if (expected) same_identity(*expected, identity);
    if (fixture) same_identity(fixture->identity, identity);
    require(engine.member_count() <= maximum,
            "existing membership exceeds requested member limit");
    uint16_t const pending_count = !engine.is_staging() && saved.pending ? saved.target_members - saved.old_members : 0;
    uint64_t const base_fixed = sum(sum(external_bytes, fixture ? fixture->bytes : 0), mib);
    if (!backend) backend = std::make_unique<Backend>();
    uint64_t generation_reserve = 0;
    uint16_t const after_recovery = saved.pending && !engine.is_staging() ? saved.target_members : engine.member_count();
    if (pending_count || (engine.can_append() && after_recovery < maximum))
    {
        uint16_t const minimum = std::max(uint16_t(1), pending_count);
        uint64_t const floor = sum(base_fixed, sum(engine.memory_bytes(minimum, compact),
            sum(product(minimum, 2 * sizeof(group_grow::Member)), batch_hash_bytes(args, minimum))));
        Producer probe{identity, fixture, *backend, args, args.max_ram, floor, 0};
        uint16_t const first = pending_count ? saved.old_members : after_recovery;
        uint64_t const prefix = product(minimum - 1, member_storage_bound(args, identity));
        bool borrowed = probe.can_borrow(prefix);
        bool admitted = probe.fits(first, 0, borrowed && minimum == 1);
        if (!admitted && minimum == 1 && !borrowed && probe.can_borrow())
        { borrowed = true; admitted = probe.fits(first, 0, true); }
        require(admitted,
            "member producer and engine workspace exceed --max-ram or available host memory");
        // The batch bound adds its owning prefix. Its last member may borrow
        // output, while the first generation above still needs its own copy.
        generation_reserve = std::max(probe.next_peak(first, borrowed), backend->retained());
    }
    uint16_t const batch_cap = admitted_batch_cap(uint16_t(std::min(args.batch, uint64_t(maximum - saved.old_members))),
        std::max(uint16_t(1), pending_count), args.max_ram - std::min(args.max_ram, sum(base_fixed, generation_reserve)),
        member_storage_bound(args, identity), [&](uint16_t count)
        { return sum(engine.memory_bytes(count, compact), sum(product(count, 2 * sizeof(group_grow::Member)), batch_hash_bytes(args, count))); });
    require(batch_cap, "engine workspace exceeds --max-ram");
    uint64_t engine_bytes = engine.memory_bytes(batch_cap, compact);
    uint64_t fixed = sum(sum(sum(sum(sum(engine_bytes, external_bytes),
        fixture ? fixture->bytes : 0), product(batch_cap, 2 * sizeof(group_grow::Member))), mib),
        batch_hash_bytes(args, batch_cap));
    require(fixed < args.max_ram, "engine workspace exceeds --max-ram");
    require(!saved.pending || saved.target_members <= maximum,
            "pending transaction membership exceeds requested limit or available fixtures");
    Producer producer{identity, fixture, *backend, args, args.max_ram, fixed, fixed};
    producer.admit_engine({});
    if (inherited && !inherited->members.empty())
    {
        // The inherited members become this batch's prefix. Charge their old
        // metadata beside the new reservation only until ownership transfers.
        uint64_t const transfer = sum(sum(fixed, inherited_bytes), backend->retained());
        if (transfer > args.max_ram) { inherited->clear(); inherited_bytes = 0; }
        else inherited_peak = std::max(inherited_peak, transfer);
    }
    if (pending_count)
        require(producer.fits(saved.old_members, 0, producer.can_borrow() && pending_count == 1),
                "member producer exceeds --max-ram or available host memory");
    engine.confirm_identity();
    Metrics metrics;
    metrics.peak_modeled_bytes = inherited_peak;
    metrics.capacity_bytes = capacity;
    metrics.peak_file_bytes = sum(saved.bytes, auxiliary.retired);
    metrics.final_bytes = saved.bytes;
    metrics.members = engine.member_count();
    metrics.engine_reserved_bytes = fixed;
    metrics.recovered = engine.staged_recovery;
    auto join_consumer = [&](auto& pending, uint32_t reasons)
    {
        if (pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) return pending.get();
        reasons |= wait_bit(ControllerWait::consumer);
        auto const begin = std::chrono::steady_clock::now();
        ControllerTiming timing("wait", 0, reasons);
        auto result = pending.get();
        metrics.controller_waits.add(reasons, group_grow::elapsed_ns(begin));
        return result;
    };
    auto release_scratch = [&]
    {
        if (!scratch_pool || !scratch_live || engine.external || !engine.scratch_cleanup_complete) return false;
        scratch_pool->release_scratch_after_cleanup(0); scratch_live = false;
        uint64_t const expanded = std::min(capacity_limit, scratch_pool->available_for_output(0, target.parent_path()));
        if (expanded <= capacity || !engine.incremental) return false;
        require(scratch_pool->try_reserve_output(0, target.parent_path(), expanded), "cannot expand destination after scratch cleanup");
        engine.resize_capacity(expanded); capacity = expanded; metrics.capacity_bytes = expanded; return true;
    };
    if (engine.is_staging() && !engine.can_append())
    {
        auto const result = engine.finish_staging(args.construction == "merged");
        metrics.members = result.members;
        metrics.final_bytes = result.final_bytes;
        metrics.peak_file_bytes = std::max(metrics.peak_file_bytes, result.peak_file_bytes);
        metrics.workspace_bytes = std::max(metrics.workspace_bytes, result.workspace_bytes);
        (void)release_scratch();
    }
    else if (!engine.is_staging() && saved.pending)
    {
        std::vector<group_grow::Member> incoming;
        auto hash = compact && args.pool_broker.empty() ? std::make_unique<group_grow::MemberBatchHash>(batch_cap) : nullptr;
        MemberLease borrowed;
        incoming.reserve(pending_count);
        require(incoming.capacity() <= batch_cap, "member container exceeds admitted capacity");
        metrics.generated_members += producer.fill(incoming, saved.old_members, pending_count, 0,
            hash.get(), &borrowed, "regenerating member ");
        require(incoming.size() == pending_count,
                "recovery member batch exceeds --max-ram or available host memory");
        producer.admit_engine(incoming);
        auto const result = engine.resume(incoming, hash.get());
        metrics.members = result.members;
        metrics.final_bytes = result.final_bytes;
        metrics.peak_file_bytes = std::max(metrics.peak_file_bytes, result.peak_file_bytes);
        metrics.workspace_bytes = std::max(metrics.workspace_bytes, result.workspace_bytes);
        ++metrics.batches;
        metrics.recovered = true;
    }

    uint64_t observed_member_bytes = metrics.members ?
        std::max(uint64_t(1), metrics.final_bytes / metrics.members) : 0;
    uint64_t observed_member_storage = 0;
    std::vector<group_grow::Member> prefetched;
    std::unique_ptr<group_grow::MemberBatchHash> prefetched_hash;
    if (inherited && !inherited->members.empty())
        inherited->take(prefetched, prefetched_hash, maximum, batch_cap);
    while (metrics.members < maximum)
    {
        check_cancel(args);
        engine.refresh_history();
        if (auto selected = automatic_selection(engine, producer, maximum, batch_cap, engine_bytes,
                0, observed_member_storage, true, resident(prefetched)))
        {
            engine_bytes = selected->engine_bytes; producer.fixed = fixed = selected->fixed;
            engine.retune(selected->config); engine.selected_batch = selected->batch;
            if (selected->merged) engine.select_merged(!args.quiet);
            producer.admit_engine(prefetched);
            metrics.engine_reserved_bytes = std::max(metrics.engine_reserved_bytes, fixed);
        }
        uint64_t selected_batch = engine.selected_batch;
        if (engine.external && engine.can_append())
            selected_batch = external_batch_size(engine.external_batches, engine.external->batch_resources(),
                producer, engine.config, maximum, batch_cap, 0, observed_member_storage);
        uint64_t wanted = std::min({args.batch, uint64_t(batch_cap), uint64_t(maximum - metrics.members)});
        if (selected_batch) wanted = std::min(wanted, selected_batch);
        if (!engine.is_staging() && observed_member_bytes && metrics.final_bytes < capacity)
        {
            uint64_t const remaining = capacity - metrics.final_bytes;
            // The extra candidate makes the boundary exact after a fit estimate.
            wanted = std::min(wanted, remaining / observed_member_bytes + 1);
        }
        auto incoming = std::move(prefetched);
        auto hash = std::move(prefetched_hash);
        if (compact && args.pool_broker.empty() && !engine.is_staging() && !hash && incoming.empty())
            hash = std::make_unique<group_grow::MemberBatchHash>(batch_cap);
        MemberLease borrowed;
        std::unique_ptr<group_grow::ExternalEngine::ContributionInput> contribution_input;
        uint16_t contribution_members = 0;
        prefetched = std::vector<group_grow::Member>{};
        if (incoming.size() > wanted) incoming.resize(size_t(wanted));
        if (hash) hash->truncate(incoming.size());
        (void)producer.can_borrow(); // Initialize the backend before reporting it.
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (backend->mode == "pool" && engine.external && engine.can_append() && incoming.empty())
        {
            uint16_t count = uint16_t(wanted);
            for (; count; --count)
            {
                uint64_t const admitted = producer.contribution_bytes(engine.config, count);
                uint64_t const planned = sum(sum(fixed, backend->retained()), admitted);
                uint64_t const available = available_host_bytes();
                if (planned > args.max_ram || (available && admitted > available)) continue;
                contribution_input = engine.external->prepare_contributions(count,
                    fragment_bound(identity.params) / sizeof(uint64_t));
                if (!contribution_input) continue;
                contribution_members = uint16_t(producer.fill_contributions(*contribution_input,
                    uint16_t(metrics.members), 0, admitted, "generating compressed pool member "));
                metrics.generated_members += contribution_members;
                metrics.stream_input_written_bytes = sum(metrics.stream_input_written_bytes, contribution_input->written_bytes());
                break;
            }
        }
#endif
        if (!contribution_input)
        {
            incoming.reserve(wanted);
            require(incoming.capacity() <= batch_cap, "member container exceeds admitted capacity");
            metrics.generated_members += producer.fill(incoming, uint16_t(metrics.members), size_t(wanted), 0,
                hash.get(), &borrowed, fixture ? "loading fixture member " : "generating " + backend->mode + " member ");
        }
        producer.admit_engine(incoming);
        std::optional<group_grow::AppendResult> appended;
        while (!incoming.empty() || contribution_members)
        {
            uint64_t const count = contribution_input ? contribution_members : incoming.size();
            uint64_t const next = metrics.members + count;
            // Snapshot and select before async append can mutate the engine.
            // Observe its completed phases only after the consumer has joined.
            auto const external_before = engine.external ?
                std::optional(engine.external->stats()) : std::optional<group_grow::Stats>{};
            bool const need_prefetch_hash = compact && args.pool_broker.empty() && !engine.is_staging();
            size_t const prefetch_count = size_t(std::min({args.batch, uint64_t(batch_cap), uint64_t(maximum - next),
                selected_batch ? selected_batch : uint64_t(batch_cap)}));
            uint64_t received = resident(incoming);
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            if (contribution_input) received = producer.contribution_bytes(engine.config, contribution_input->size());
#endif
            if (producer.overlap_enabled() && backend->mode != "cpu" && next < maximum &&
                producer.fits(uint16_t(next), received) && producer.overlap_enabled())
            {
                // Keep GPU use on its constructing thread. The CPU consumer
                // borrows immutable members until get (also drained on unwind).
                auto pending = std::async(std::launch::async, [&]
                {
                    return contribution_input ? engine.append(*contribution_input, contribution_members) :
                        engine.append(incoming, hash.get());
                });
                if (need_prefetch_hash) prefetched_hash = std::make_unique<group_grow::MemberBatchHash>(batch_cap);
                prefetched.reserve(prefetch_count);
                require(prefetched.capacity() <= batch_cap, "member container exceeds admitted capacity");
                // Stop prefetching as soon as assembly finishes. The producer
                // also drains its final pending member before either hash or
                // destination batch can be read on this thread.
                auto const generated = producer.fill(prefetched, uint16_t(next), prefetch_count, received,
                    prefetched_hash.get(), nullptr, "prefetching " + backend->mode + " member ", [&]
                    { return pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
                metrics.generated_members += generated;
                if (generated) ++metrics.generation_overlap_batches;
                appended = join_consumer(pending, fill_wait_reason(producer.fill_stop));
            }
            else appended = contribution_input ? engine.append(*contribution_input, contribution_members) : engine.append(incoming, hash.get());
            if (appended->accepted && external_before && engine.external && !contribution_input)
                engine.observe_external(uint16_t(incoming.size()), *external_before);
            if (release_scratch() && !appended->accepted)
                appended = engine.append(incoming, hash.get());
            if (!engine.is_staging() && !appended->fresh_single_pass) ++metrics.exact_plan_batches;
            metrics.attempted_final_bytes = appended->attempted_final_bytes;
            metrics.attempted_peak_bytes = appended->attempted_peak_bytes;
            if (appended->accepted) break;
            prefetched.clear(); // Rejected candidates change the next member index.
            prefetched_hash.reset();
            std::cerr << "candidate batch rejected: " << appended->reason << '\n';
            if (contribution_input)
            {
                if (contribution_members > 1) { --contribution_members; continue; }
                contribution_input.reset(); contribution_members = 0;
                auto const result = engine.fallback_input();
                metrics.members = result.members; metrics.final_bytes = result.final_bytes;
                metrics.peak_file_bytes = std::max(metrics.peak_file_bytes, result.peak_file_bytes);
                metrics.workspace_bytes = std::max(metrics.workspace_bytes, result.workspace_bytes);
                (void)release_scratch();
                incoming.reserve(1);
                metrics.generated_members += producer.fill(incoming, uint16_t(metrics.members), 1, 0,
                    hash.get(), &borrowed, "regenerating pool member after contribution admission ");
                continue;
            }
            if (incoming.size() == 1) break;
            incoming.pop_back();
            if (hash) hash->truncate(incoming.size());
            borrowed.reset(); // A borrowed member is always the final candidate.
        }
        require(appended.has_value(), "empty batch plan");
        if (!appended->accepted)
        {
            metrics.stop_reason = "capacity";
            break;
        }
        uint64_t const before = metrics.final_bytes;
        uint16_t const added = contribution_input ? contribution_members : uint16_t(incoming.size());
        if (contribution_input) ++metrics.stream_input_batches;
        else observed_member_storage = (resident(incoming) + added - 1) / added;
        metrics.max_batch_members = std::max<uint64_t>(metrics.max_batch_members, added);
        if (appended->fresh_single_pass) ++metrics.fresh_single_pass_batches;
        auto const result = appended->result;
        metrics.members = result.members;
        metrics.final_bytes = result.final_bytes;
        metrics.peak_file_bytes = std::max(metrics.peak_file_bytes, result.peak_file_bytes);
        metrics.workspace_bytes = std::max(metrics.workspace_bytes, result.workspace_bytes);
        if (metrics.final_bytes > before)
            observed_member_bytes = std::max(uint64_t(1), (metrics.final_bytes - before) / added);
        ++metrics.batches;
        if (args.command.empty() || args.verbose) std::cerr << (engine.is_staging() ? "staged " : "committed ") << metrics.members << " members, "
                  << metrics.final_bytes << (engine.is_staging() ? " byte canonical bound\n" : " bytes\n");
    }
    group_grow::Result finished;
    std::optional<group_grow::ScratchPool::IoLease> finish_io;
    bool successor_ready = false;
    { ControllerTiming timing("prepare_pack", 0); engine.prepare_finish(); }
    uint64_t const finishing_bytes = engine.memory_bytes(batch_cap, compact);
    require(finishing_bytes <= engine_bytes, "final packing exceeds the destination workspace lease");
    uint64_t const released_bytes = engine_bytes - finishing_bytes;
    if (engine.external)
    {
        metrics.packing_engine_bytes = finishing_bytes;
        metrics.packing_released_bytes = released_bytes;
    }
    fixed -= released_bytes; engine_bytes = finishing_bytes; producer.fixed = fixed;
    auto finish = [&] { ControllerTiming timing("finish", 0); return engine.finish(); };
    if (successor && !successor->recovery && scratch_pool && engine.external && engine.config.workers > 1 &&
        (args.construction == "auto" || args.construction == "external") &&
        metrics.members == maximum && metrics.stop_reason != "capacity" && group_prefetch_enabled(args, *backend))
    {
        // The completed prefix reserves its complete final-output envelope.
        // The successor receives only the remaining aggregate directory quota.
        if (successor->capacity > capacity)
        {
            successor->capacity -= capacity;
            finish_io = scratch_pool->try_io(0);
            if (finish_io)
            {
                unsigned const spare = engine.staging_workers(engine.config.workers);
                successor_ready = prepare_successor(*successor, args, *backend, *scratch_pool, 1,
                    producer.fixed, resident(prefetched), spare, target_gate, fixture, histories, engine.demand_history, released_bytes, cpu_scheduler);
                producer.fixed = sum(producer.fixed, successor->pool_growth);
                if (successor_ready) engine.admit_staging_workers(engine.config.workers, spare);
            }
        }
    }
    if (successor_ready)
    {
        auto pending = std::async(std::launch::async, finish);
        uint32_t const reason = stage_successor(*successor, args, *backend, resident(prefetched), [&]
            { return pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }, fixture);
        successor->close();
        finished = join_consumer(pending, reason);
        metrics.controller_waits.add(successor->metrics.controller_waits);
        metrics.peak_modeled_bytes = std::max(metrics.peak_modeled_bytes, successor->metrics.peak_modeled_bytes);
    }
    else if (prefetch_next && group_prefetch_enabled(args, *backend) && metrics.stop_reason != "capacity")
    {
        auto pending = std::async(std::launch::async, finish);
        // Device generation stays on the constructing thread; get also joins
        // final packing before engine ownership or scratch reservations change.
        uint32_t const reason = prefetch_next(producer.fixed, batch_cap, pending);
        finished = join_consumer(pending, reason);
    }
    else finished = finish();
    finish_io.reset();
    metrics.members = finished.members;
    metrics.final_bytes = finished.final_bytes;
    metrics.peak_file_bytes = std::max(metrics.peak_file_bytes, finished.peak_file_bytes);
    metrics.workspace_bytes = std::max(metrics.workspace_bytes, finished.workspace_bytes);
    metrics.staged_batches = engine.staged_batches;
    metrics.staging_fallbacks = engine.fallbacks;
    metrics.automatic_merges = engine.automatic_merges;
    metrics.automatic_fallbacks = engine.automatic_fallbacks;
    metrics.tuning_changes = engine.tuning_changes;
    metrics.selected_window_bytes = engine.config.window_bytes;
    metrics.selected_checkpoint_bytes = engine.selected_checkpoint_bytes();
    metrics.construction_selected = engine.mode();
    if (metrics.stop_reason.empty())
    {
        if (maximum == 65535) metrics.stop_reason = "format-limit";
        else if (fixture && maximum == fixture->paths.size() && maximum < args.max_members)
            metrics.stop_reason = "fixture-limit";
        else metrics.stop_reason = "member-limit";
    }
    metrics.peak_modeled_bytes = std::max(metrics.peak_modeled_bytes, producer.peak);
    metrics.generation_overlap_members = producer.overlap_members;
    if (!fixture && metrics.generated_members)
    {
        metrics.record_generation(backend->mode, producer.generation_peak);
    }
    engine.release_external();
    metrics.scratch_released = engine.scratch_cleanup_complete;
    metrics.scratch_peak_bytes = engine.peak_scratch_bytes;
    static_cast<group_grow::Stats&>(metrics) = engine.stats();
    if (scratch_pool && engine.scratch_cleanup_complete)
        scratch_pool->release_after_cleanup(0, sum(allocated_bytes(target), engine.remaining_metadata_bytes));
    return metrics;
}

void print_metrics(Arguments const& args, Metrics const& metrics, Backend const& backend,
                   uint64_t groups = 0, uint64_t current_group_members = 0,
                   uint64_t accounted_file_bytes = 0)
{
    if (!args.report_metrics) return;
    if (!args.command.empty() && !args.json)
    {
        if (!args.quiet)
            std::cerr << '[' << args.command << "] " << metrics.members << " members, "
                      << metrics.final_bytes << " bytes; " << metrics.stop_reason << '\n';
        return;
    }
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS usage{};
    require(::K32GetProcessMemoryInfo(::GetCurrentProcess(), &usage, sizeof(usage)), "cannot inspect process memory peak");
    uint64_t const peak_rss_bytes = usage.PeakWorkingSetSize;
#else
    struct rusage usage{}; require(!::getrusage(RUSAGE_SELF, &usage), "cannot inspect process memory peak");
    uint64_t const peak_rss_bytes = product(uint64_t(usage.ru_maxrss), 1024);
#endif
    std::cout << "{\"mode\":\"" << backend.mode
              << "\",\"members\":" << metrics.members << ",\"final_bytes\":" << metrics.final_bytes
              << ",\"capacity_bytes\":" << metrics.capacity_bytes
              << ",\"scratch_peak_bytes\":" << metrics.scratch_peak_bytes
              << ",\"scratch_released\":" << (metrics.scratch_released ? "true" : "false")
              << ",\"peak_file_bytes\":" << metrics.peak_file_bytes
              << ",\"workspace_bytes\":" << metrics.workspace_bytes
              << ",\"attempted_final_bytes\":" << metrics.attempted_final_bytes
              << ",\"attempted_peak_bytes\":" << metrics.attempted_peak_bytes
              << ",\"generated_members\":" << metrics.generated_members << ",\"batches\":" << metrics.batches
              << ",\"peak_modeled_bytes\":" << sum(metrics.peak_modeled_bytes, sum(args.pool_offer_ram, args.pool_codec_ram))
              << ",\"engine_reserved_bytes\":" << metrics.engine_reserved_bytes
              << ",\"cpu_generation_peak_bytes\":" << metrics.cpu_generation_peak_bytes
              << ",\"gpu_generation_peak_bytes\":" << metrics.gpu_generation_peak_bytes
              << ",\"pool_generation_peak_bytes\":" << metrics.pool_generation_peak_bytes
              << ",\"pool_offered_ram_bytes\":" << args.pool_offer_ram
              << ",\"pool_codec_ram_bytes\":" << args.pool_codec_ram
              << ",\"gpu_retained_host_bytes\":" << (args.pool_broker.empty() ? backend.retained() : 0)
              << ",\"gpu_device_resident_bytes\":" << backend.device_bytes()
              << ",\"backend_requested\":\"" << args.backend
              << "\",\"backend_fallback\":\"" << backend.fallback
              << "\",\"gpu_tier\":\"" << backend.tier
              << "\",\"member_storage\":\"" << args.member_storage
              << "\",\"cache_release\":" << (args.cache_release ? "true" : "false")
              << ",\"construction_requested\":\"" << args.construction << '"'
              << ",\"staged_batches\":" << metrics.staged_batches
              << ",\"staging_fallbacks\":" << metrics.staging_fallbacks
              << ",\"stream_input_batches\":" << metrics.stream_input_batches
              << ",\"stream_input_written_bytes\":" << metrics.stream_input_written_bytes
              << ",\"stream_input_read_bytes\":" << metrics.stream_input_read_bytes
              << ",\"stream_input_preflight_rejections\":" << metrics.stream_input_preflight_rejections
              << ",\"automatic_merges\":" << metrics.automatic_merges
              << ",\"automatic_fallbacks\":" << metrics.automatic_fallbacks
              << ",\"tuning_changes\":" << metrics.tuning_changes
              << ",\"selected_window_bytes\":" << metrics.selected_window_bytes
              << ",\"selected_checkpoint_bytes\":" << metrics.selected_checkpoint_bytes
              << ",\"max_batch_members\":" << metrics.max_batch_members
              << ",\"construction_selected\":\"" << metrics.construction_selected << '"'
              << ",\"fresh_single_pass_batches\":" << metrics.fresh_single_pass_batches
              << ",\"exact_plan_batches\":" << metrics.exact_plan_batches
              << ",\"encode_chunks\":" << metrics.encode_chunks
              << ",\"sized_chunks\":" << metrics.sized_chunks
              << ",\"write_calls\":" << metrics.write_calls
              << ",\"read_calls\":" << metrics.read_calls
              << ",\"read_bytes\":" << metrics.read_bytes
              << ",\"written_bytes\":" << metrics.written_bytes
              << ",\"read_bytes_per_final_byte\":" << (metrics.final_bytes ? double(metrics.read_bytes) / metrics.final_bytes : 0)
              << ",\"written_bytes_per_final_byte\":" << (metrics.final_bytes ? double(metrics.written_bytes) / metrics.final_bytes : 0)
              << ",\"storage_read_seconds\":" << double(metrics.read_ns) / 1e9
              << ",\"storage_write_seconds\":" << double(metrics.write_ns) / 1e9
              << ",\"storage_sync_seconds\":" << double(metrics.sync_ns) / 1e9
              << ",\"codec_seconds\":" << double(metrics.codec_ns) / 1e9
              << ",\"decode_seconds\":" << double(metrics.decode_ns) / 1e9
              << ",\"generation_seconds\":" << double(backend.generation_ns) / 1e9
              << ",\"peak_rss_bytes\":" << peak_rss_bytes
              << ",\"merge_passes\":" << metrics.merge_passes
              << ",\"reclaimed_bytes\":" << metrics.reclaimed_bytes
              << ",\"max_active_runs\":" << metrics.max_active_runs
              << ",\"merge_inputs\":" << metrics.merge_inputs
              << ",\"contribution_workers\":" << metrics.contribution_workers
              << ",\"packing_overlap_windows\":" << metrics.packing_overlap_windows
              << ",\"layout_moves\":" << metrics.layout_moves
              << ",\"layout_bytes\":" << metrics.layout_bytes
              << ",\"layout_batches\":" << metrics.layout_batches
              << ",\"placed_blocks\":" << metrics.placed_blocks
              << ",\"destinations\":" << metrics.destination_count
              << ",\"writer_groups\":" << metrics.writer_count
              << ",\"queued_peak_bytes\":" << metrics.queued_peak_bytes
              << ",\"drive_overlap_batches\":" << metrics.drive_overlap_batches
              << ",\"execution_active_destinations\":" << metrics.execution_active
              << ",\"execution_batch_members\":" << metrics.execution_batch
              << ",\"execution_estimate_known\":" << (metrics.execution_estimate.known ? "true" : "false")
              << ",\"stream_input_trial\":\"" << metrics.stream_input_trial << '"'
              << ",\"execution_estimated_seconds\":";
    if (metrics.execution_estimate.known) std::cout << double(metrics.execution_estimate.completion) / 1e9;
    else std::cout << "null";
    std::cout
              << ",\"storage_queue_wait_seconds\":" << double(metrics.queue_wait_ns) / 1e9
              << ",\"packing_engine_bytes\":" << metrics.packing_engine_bytes
              << ",\"packing_released_bytes\":" << metrics.packing_released_bytes
              << ",\"checkpoints\":" << metrics.checkpoints << ",\"syncs\":" << metrics.syncs
              << ",\"overlap_windows\":" << metrics.overlap_windows
              << ",\"generation_overlap_batches\":" << metrics.generation_overlap_batches
              << ",\"generation_overlap_members\":" << metrics.generation_overlap_members
              << ",\"recovered\":" << (metrics.recovered ? "true" : "false")
              << ",\"stop_reason\":\"" << metrics.stop_reason << '"';
    std::cout << ",\"controller_wait_seconds\":{";
    for (size_t i = 0; i < controller_wait_names.size(); ++i)
    {
        if (i) std::cout << ',';
        std::cout << '"' << controller_wait_names[i] << "\":" << double(metrics.controller_waits.nanoseconds[i]) / 1e9;
    }
    std::cout << '}';
    constexpr std::array phases{"plan", "encode", "merge", "pack", "layout"};
    std::cout << ",\"phases\":{";
    for (size_t i = 0; i < phases.size(); ++i)
    {
        if (i) std::cout << ',';
        auto const& phase = metrics.phases[i];
        std::cout << '"' << phases[i] << "\":{\"seconds\":" << double(phase.elapsed_ns) / 1e9
                  << ",\"read_seconds\":" << double(phase.read_ns) / 1e9
                  << ",\"write_seconds\":" << double(phase.write_ns) / 1e9
                  << ",\"sync_seconds\":" << double(phase.sync_ns) / 1e9
                  << ",\"codec_seconds\":" << double(phase.codec_ns) / 1e9
                  << ",\"decode_seconds\":" << double(phase.decode_ns) / 1e9 << '}';
    }
    std::cout << '}';
    if (args.group_members || args.directory_farm)
    {
        std::cout << ",\"rollover\":true,\"groups\":" << groups;
        if (args.group_members) std::cout << ",\"group_members\":" << args.group_members;
        else std::cout << ",\"group_members_mode\":\"auto\"";
        std::cout << ",\"current_group_members\":" << current_group_members
                  << ",\"accounted_file_bytes\":" << accounted_file_bytes;
    }
    if (args.directory_farm)
    {
        auto const precision = std::cout.precision();
        std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10)
                  << ",\"farming_response_bound_seconds\":" << metrics.farming_response_bound_seconds
                  << ",\"farming_dynamic_work_seconds_per_sp\":" << metrics.farming_dynamic_work_seconds_per_sp
                  << ",\"farming_certified_max_eligible_groups\":" << metrics.farming_certified_max_eligible_groups;
        std::cout.precision(precision);
    }
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
    if (!args.pool_broker.empty())
    {
        std::cout << ",\"pool_packed_members\":" << backend.pool_members.load()
                  << ",\"pool_contribution_members\":" << backend.pool_contribution_members.load()
                  << ",\"pool_local_fallbacks\":" << backend.pool_local_fallbacks.load()
                  << ",\"pool_retries\":" << backend.pool_retries.load()
                  << ",\"pool_retained_host_bytes\":" << backend.retained();
        if (backend.pool_remote)
        {
            auto const& remote = backend.pool_remote->counters;
            std::cout << ",\"pool_codec_windows\":" << remote.accepted_windows.load()
                      << ",\"pool_codec_chunks\":" << remote.accepted_chunks.load()
                      << ",\"pool_codec_fallback_windows\":" << remote.fallback_windows.load()
                      << ",\"pool_codec_calibration_windows\":" << remote.calibration_windows.load()
                      << ",\"pool_codec_input_bytes\":" << remote.input_bytes.load()
                      << ",\"pool_codec_output_bytes\":" << remote.output_bytes.load();
        }
        if (backend.pool_client)
        {
            auto const wire = backend.pool_client->stats();
            std::cout << ",\"pool_wire_input_bytes\":" << wire.input_bytes
                      << ",\"pool_wire_output_bytes\":" << wire.output_bytes
                      << ",\"pool_completed_generation_tasks\":" << wire.completed_generation
                      << ",\"pool_completed_contribution_tasks\":" << wire.completed_contributions
                      << ",\"pool_completed_codec_tasks\":" << wire.completed_codec
                      << ",\"pool_local_direct_tasks\":" << wire.local_tasks
                      << ",\"pool_task_elapsed_ns\":" << wire.task_elapsed_ns
                      << ",\"pool_peak_generation_members\":" << wire.peak_generation_members
                      << ",\"pool_peak_active_generation\":" << wire.peak_active_generation
                      << ",\"pool_failed_tasks\":" << wire.failed_tasks;
        }
    }
#endif
    std::cout << "}\n";
}

fs::path group_path(fs::path const& directory, uint64_t ordinal)
{
    std::ostringstream name;
    name << "grow-group-" << std::setfill('0') << std::setw(20) << ordinal << ".gplot";
    return directory / name.str();
}
std::optional<uint64_t> rollover_entry(fs::path const& directory, fs::path const& path, uint64_t memory_limit,
                                     std::optional<uint64_t> writer_owned = {})
{
    auto const name = file_io::path_text(path.filename());
    std::string const prefix = "grow-group-", suffix = ".gplot";
    if (!name.starts_with(prefix)) return {};
    size_t const length = prefix.size() + 20 + suffix.size();
    require(name.size() >= length && name.substr(prefix.size() + 20, suffix.size()) == suffix,
        "ambiguous rollover output name");
    uint64_t const ordinal = number(name.substr(prefix.size(), 20).c_str());
    auto const target = group_path(file_io::native_path(directory), ordinal);
    require(target.filename() == name.substr(0, length), "noncanonical rollover output name");
    if (name.size() == length) return ordinal;
    // A live engine owns its checkpoint and scratch turnover until its future
    // joins. Validate names here, then audit that metadata after ownership ends.
    bool const inspect = !writer_owned || *writer_owned != ordinal;
    if (name == file_io::path_text(target.filename()) + ".external")
    {
        if (inspect) (void)group_grow::ExternalEngine::inspect(file_io::path_text(target), memory_limit);
    }
    else if (name.starts_with(file_io::path_text(target.filename()) + ".external.retired-"))
    {
        auto const token = name.substr(length + std::string_view(".external.retired-").size());
        require(token.size() == 64 && std::all_of(token.begin(), token.end(), [](char c)
            { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }), "ambiguous rollover output name");
        if (inspect) (void)group_grow::ExternalEngine::inspect_retired(file_io::path_text(target), file_io::path_text(path), memory_limit);
    }
    else throw std::runtime_error("ambiguous rollover output name");
    return {};
}
void check_outputs(fs::path const& directory, std::span<Output const> outputs, uint64_t memory_limit)
{
    uint64_t seen = 0;
    for (auto const& entry : fs::directory_iterator(file_io::native_path(directory)))
    {
        auto const ordinal = rollover_entry(directory, entry.path(), memory_limit);
        if (!ordinal) continue;
        require(*ordinal < outputs.size(),
                "rollover output appeared during the run");
        ++seen;
    }
    require(seen == outputs.size(), "rollover output disappeared during the run");
    for (auto const& saved : outputs)
    {
        auto const now = snapshot(group_path(file_io::native_path(directory), saved.ordinal), saved.ordinal);
        require(now.device == saved.device && now.inode == saved.inode && now.bytes == saved.bytes &&
                    now.allocated == saved.allocated && now.modified_seconds == saved.modified_seconds &&
                    now.modified_nanoseconds == saved.modified_nanoseconds &&
                    now.changed_seconds == saved.changed_seconds && now.changed_nanoseconds == saved.changed_nanoseconds,
                "rollover output path or contents changed during the run");
    }
}
struct DirectoryLock
{
    int fd = -1;
    fs::path path;
    uint64_t device = 0, inode = 0;
    explicit DirectoryLock(fs::path value) : path(std::move(value))
    {
        fd = file_io::open_file(path.c_str(), file_io::read_only | file_io::directory | file_io::close_on_exec | file_io::no_follow);
        require(fd >= 0, "cannot open rollover destination directory");
        file_io::Stat info{};
        if (file_io::lock_file(fd, file_io::lock_exclusive | file_io::lock_nonblocking) != 0 || file_io::inspect_fd(fd, &info) != 0)
        {
            file_io::close_file(fd);
            fd = -1;
            throw std::runtime_error("rollover directory is already in use or unavailable");
        }
        device = info.st_dev;
        inode = info.st_ino;
    }
    ~DirectoryLock() { if (fd >= 0) file_io::close_file(fd); }
    void check() const
    {
        file_io::Stat info{};
        require(file_io::inspect_link(path.c_str(), &info) == 0 && file_io::is_directory(info.st_mode) &&
                    uint64_t(info.st_dev) == device && uint64_t(info.st_ino) == inode,
                "rollover destination directory changed during the run");
    }
};

int rollover(Arguments const& args, Keys const& keys)
{
    require(fs::is_directory(file_io::native_path(args.target)), "rollover destination directory must exist");
    DirectoryLock lock(args.target);
    std::vector<Output> outputs;
    for (auto const& entry : fs::directory_iterator(file_io::native_path(args.target)))
    {
        auto const ordinal = rollover_entry(args.target, entry.path(), args.max_ram);
        if (!ordinal) continue;
        require(product(sum(outputs.size(), 1), sizeof(Output) * 4) < args.max_ram / 16,
                "rollover output metadata exceeds --max-ram");
        outputs.push_back(snapshot(entry.path(), *ordinal));
    }
    std::sort(outputs.begin(), outputs.end(), [](auto const& a, auto const& b) { return a.ordinal < b.ordinal; });
    require(outputs.empty() || args.resume, "rollover outputs exist; use --resume");
    require(!outputs.empty() || !args.resume, "--resume found no rollover output");
    require(outputs.size() <= args.max_groups, "existing outputs exceed --max-groups");
    uint64_t allocated = 0, charged = 0, logical = 0, members = 0;
    group_grow::Stats audited{};
    std::optional<uint64_t> first_pending;
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        require(outputs[i].ordinal == i, "rollover output sequence has a hole");
        auto const expected = keys.derive(i);
        uint64_t const metadata = product(outputs.capacity(), sizeof(Output));
        auto config = configuration(args, uint64_t(INT64_MAX), sum(metadata, mib));
        Construction engine(file_io::path_text(group_path(args.target, i)), config, true);
        auto const saved = engine.identity();
        same_identity(expected, identity_from(saved));
        require(saved.header.group_size <= args.group_members &&
                    (!saved.pending || saved.target_members <= args.group_members),
                "saved membership differs from --group-members");
        bool const pending = saved.pending || engine.is_staging();
        bool const pair_prefix = i + 2 == outputs.size() && engine.external &&
            engine.member_count() == args.group_members;
        require(i + 1 == outputs.size() || (!pending && saved.header.group_size == args.group_members) || pair_prefix,
                "only a full external prefix may precede a staged rollover successor");
        if (first_pending)
            require(i + 1 == outputs.size() && i == *first_pending + 1 && engine.external && engine.can_append(),
                "rollover successor must be the adjacent external staging output");
        if (pending && !first_pending) first_pending = i;
        require(sum(members, saved.pending ? saved.target_members : saved.header.group_size) <=
                    args.max_total_members, "saved transaction exceeds --max-total-members");
        require(!saved.pending || saved.header.group_size || members < args.max_total_members,
                "empty pending output requires room for a member under --max-total-members");
        outputs[i].members = engine.member_count();
        outputs[i].pending = saved.pending || engine.is_staging();
        output_auxiliary(group_path(args.target, i), outputs[i], args.max_ram);
        audited = Construction::add_stats(audited, engine.stats());
        allocated = sum(allocated, sum(outputs[i].allocated, outputs[i].auxiliary_allocated));
        charged = sum(charged, outputs[i].charge());
        logical = sum(logical, outputs[i].bytes);
        members = sum(members, outputs[i].members);
    }
    require(members <= args.max_total_members, "existing outputs exceed --max-total-members");
    lock.check();
    check_outputs(args.target, outputs, args.max_ram);
    file_io::Space space{};
    require(file_io::space_fd(lock.fd, &space) == 0, "cannot inspect rollover filesystem space");
    uint64_t const available = sum(product(space.f_bavail, space.f_frsize), allocated);
    uint64_t const allocation_alignment = std::max(uint64_t(space.f_bsize), uint64_t(space.f_frsize));
    require(allocation_alignment > 0, "cannot determine rollover allocation alignment");
    require(available > args.headroom, "destination filesystem has no capacity above headroom");
    uint64_t const filesystem_limit = available - args.headroom;
    uint64_t const capacity = args.capacity ? std::min(args.capacity, filesystem_limit) : filesystem_limit;
    require(charged <= capacity, "existing outputs exceed aggregate destination capacity");

    Metrics total;
    static_cast<group_grow::Stats&>(total) = audited;
    total.members = members;
    total.final_bytes = logical;
    total.capacity_bytes = capacity;
    total.peak_file_bytes = charged;
    std::unique_ptr<Backend> backend;
    group_grow::ConstructionHistories histories(1);
    uint64_t ordinal = first_pending.value_or(outputs.empty() ? 0 : outputs.back().ordinal);
    if (!first_pending && !outputs.empty() && outputs.back().members == args.group_members) ++ordinal;
    auto target_gate = std::make_shared<std::recursive_mutex>();
    uint64_t last_members = outputs.empty() ? 0 : outputs.back().members;
    GroupPrefetch prefetched;
    while (ordinal < args.max_groups && (total.members < args.max_total_members ||
           (ordinal < outputs.size() && outputs[ordinal].pending)))
    {
        lock.check();
        check_outputs(args.target, outputs, args.max_ram);
        auto const path = group_path(args.target, ordinal);
        bool const existing = ordinal < outputs.size();
        uint64_t const before_charge = existing ? outputs[ordinal].charge() : 0;
        uint64_t const before_bytes = existing ? outputs[ordinal].bytes : 0;
        uint64_t const before_members = existing ? outputs[ordinal].members : 0;
        uint64_t const prior_charge = charged - before_charge;
        uint64_t const prior_bytes = total.final_bytes - before_bytes;
        uint64_t const prior_members = total.members - before_members;
        uint16_t const maximum = uint16_t(std::min(args.group_members, args.max_total_members - prior_members));
        uint64_t const active_capacity = std::min(capacity - prior_charge,
            group_grow::maximum_file_bytes(file_io::path_text(args.target))) / allocation_alignment * allocation_alignment;
        if (prior_charge >= capacity || active_capacity == 0)
        {
            total.stop_reason = "capacity";
            break;
        }
        require(existing || !fs::exists(file_io::native_path(path)), "new rollover output path appeared during the run");
        // Include both allocations if appending this output grows the snapshot
        // vector, as well as the directory scan's temporary bookkeeping.
        uint64_t const metadata = sum(sum(product(sum(outputs.capacity(), 2), sizeof(Output) * 4), mib),
            sizeof(std::recursive_mutex) + 2 * sizeof(void*));
        Arguments current_args = args;
        current_args.resume = existing;
        auto const next_members = args.max_total_members - sum(prior_members, maximum);
        auto const next_maximum = uint16_t(std::min(args.group_members, next_members));
        std::function<uint32_t(uint64_t, uint16_t, std::future<group_grow::Result>&)> prefetch_next;
        if (ordinal + 1 < args.max_groups && next_maximum && maximum == args.group_members)
            prefetch_next = [&](uint64_t current_fixed, uint16_t cap, std::future<group_grow::Result>& pending)
            {
                auto const identity = keys.derive(ordinal + 1);
                auto const next_config = configuration(args, active_capacity, sum(metadata, histories.memory_bytes()));
                bool const nonrotating = group_grow::storage_device(args.target).nonrotating;
                auto memory = [&](uint16_t count) { return prefetch_memory(args, identity, next_config, count, nonrotating); };
                Producer producer{identity, nullptr, *backend, args, args.max_ram,
                    sum(current_fixed, memory(1)), current_fixed};
                if (!producer.fits(0, 0)) return wait_bit(ControllerWait::producer_memory);
                uint64_t const generation = std::max(producer.next_peak(0), backend->retained());
                uint16_t const wanted = admitted_batch_cap(uint16_t(std::min({args.batch, uint64_t(cap), uint64_t(next_maximum)})),
                    1, args.max_ram - std::min(args.max_ram, sum(current_fixed, generation)),
                    member_storage_bound(args, identity), memory);
                if (!wanted) return wait_bit(ControllerWait::engine_memory);
                producer.fixed = sum(current_fixed, memory(wanted));
                auto const generated = prefetch_group(prefetched, producer, identity, ordinal + 1, wanted, 0,
                    "prefetching next group " + backend->mode + " member ", [&]
                    { return pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
                if (generated)
                    prefetched.metadata_bytes = sum(product(wanted, 2 * sizeof(group_grow::Member)),
                        batch_hash_bytes(args, wanted));
                total.generated_members = sum(total.generated_members, generated);
                if (generated) ++total.generation_overlap_batches;
                total.generation_overlap_members = sum(total.generation_overlap_members, producer.overlap_members);
                total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, producer.peak);
                total.gpu_generation_peak_bytes = std::max(total.gpu_generation_peak_bytes, producer.generation_peak);
                return fill_wait_reason(producer.fill_stop);
            };
        std::unique_ptr<GroupSuccessor> successor;
        if (ordinal + 1 < outputs.size())
        {
            successor = std::make_unique<GroupSuccessor>(group_path(args.target, ordinal + 1),
                keys.derive(ordinal + 1), uint16_t(args.group_members), 0);
            successor->guard = outputs[ordinal + 1];
            successor->recovery = group_grow::ExternalEngine::inspect(file_io::path_text(successor->path), args.max_ram - metadata);
            prefetch_next = {};
        }
        else if (ordinal + 1 < args.max_groups && next_maximum && maximum == args.group_members)
            successor = std::make_unique<GroupSuccessor>(group_path(args.target, ordinal + 1),
                keys.derive(ordinal + 1), next_maximum, active_capacity);
        auto const part = run_group(current_args, path, nullptr, keys.derive(ordinal), maximum,
                                    active_capacity, sum(metadata, histories.memory_bytes()), backend, &histories,
                                    {}, existing ? outputs[ordinal] : Output{}, &prefetched, prefetch_next,
                                    successor.get(), target_gate);
        total.members = sum(prior_members, part.members);
        total.final_bytes = sum(prior_bytes, part.final_bytes);
        total.peak_file_bytes = std::max(total.peak_file_bytes, sum(prior_charge, part.peak_file_bytes));
        total.workspace_bytes = std::max(total.workspace_bytes, part.workspace_bytes);
        total.attempted_final_bytes = sum(prior_charge, part.attempted_final_bytes);
        total.attempted_peak_bytes = sum(prior_charge, part.attempted_peak_bytes);
        total.generated_members = sum(total.generated_members, part.generated_members);
        total.batches = sum(total.batches, part.batches);
        total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, part.peak_modeled_bytes);
        total.engine_reserved_bytes = std::max(total.engine_reserved_bytes, part.engine_reserved_bytes);
        total.cpu_generation_peak_bytes = std::max(total.cpu_generation_peak_bytes, part.cpu_generation_peak_bytes);
        total.gpu_generation_peak_bytes = std::max(total.gpu_generation_peak_bytes, part.gpu_generation_peak_bytes);
        total.pool_generation_peak_bytes = std::max(total.pool_generation_peak_bytes, part.pool_generation_peak_bytes);
        total.fresh_single_pass_batches = sum(total.fresh_single_pass_batches, part.fresh_single_pass_batches);
        total.exact_plan_batches = sum(total.exact_plan_batches, part.exact_plan_batches);
        total.staged_batches = sum(total.staged_batches, part.staged_batches);
        total.staging_fallbacks = sum(total.staging_fallbacks, part.staging_fallbacks);
        total.stream_input_batches = sum(total.stream_input_batches, part.stream_input_batches);
        total.stream_input_written_bytes = sum(total.stream_input_written_bytes, part.stream_input_written_bytes);
        total.stream_input_read_bytes = sum(total.stream_input_read_bytes, part.stream_input_read_bytes);
        total.stream_input_preflight_rejections = sum(total.stream_input_preflight_rejections, part.stream_input_preflight_rejections);
        total.scratch_peak_bytes = sum(total.scratch_peak_bytes, part.scratch_peak_bytes);
        total.automatic_merges = sum(total.automatic_merges, part.automatic_merges);
        total.automatic_fallbacks = sum(total.automatic_fallbacks, part.automatic_fallbacks);
        total.tuning_changes = sum(total.tuning_changes, part.tuning_changes);
        total.selected_window_bytes = std::max(total.selected_window_bytes, part.selected_window_bytes);
        total.selected_checkpoint_bytes = std::max(total.selected_checkpoint_bytes, part.selected_checkpoint_bytes);
        total.max_batch_members = std::max(total.max_batch_members, part.max_batch_members);
        if (total.construction_selected.empty()) total.construction_selected = part.construction_selected;
        else if (total.construction_selected != part.construction_selected) total.construction_selected = "mixed";
        static_cast<group_grow::Stats&>(total) = Construction::add_stats(total, part);
        total.generation_overlap_batches = sum(total.generation_overlap_batches, part.generation_overlap_batches);
        total.generation_overlap_members = sum(total.generation_overlap_members, part.generation_overlap_members);
        total.controller_waits.add(part.controller_waits);
        total.packing_engine_bytes = std::max(total.packing_engine_bytes, part.packing_engine_bytes);
        total.packing_released_bytes = std::max(total.packing_released_bytes, part.packing_released_bytes);
        total.recovered = total.recovered || part.recovered;
        if (part.members)
        {
            last_members = part.members;
            auto output = snapshot(path, ordinal);
            output_auxiliary(path, output, args.max_ram);
            output.members = uint16_t(part.members);
            charged = sum(prior_charge, output.charge());
            if (existing) outputs[ordinal] = output;
            else outputs.push_back(output);
            require(charged <= capacity, "completed output allocation exceeds aggregate capacity");
        }
        else charged = prior_charge;
        if (successor && successor->batch_cap)
        {
            auto output = snapshot(successor->path, ordinal + 1);
            output_auxiliary(successor->path, output, args.max_ram - metadata);
            output.members = output.pending_members = uint16_t(successor->metrics.members); output.pending = true;
            require(outputs.size() == ordinal + 1, "successor output ordinal changed during publication");
            outputs.push_back(output);
            charged = sum(charged, output.charge());
            total.members = sum(total.members, output.members);
            total.final_bytes = sum(total.final_bytes, output.bytes);
            total.generated_members = sum(total.generated_members, successor->metrics.generated_members);
            total.batches = sum(total.batches, successor->metrics.batches);
            total.staged_batches = sum(total.staged_batches, successor->metrics.staged_batches);
            total.generation_overlap_batches = sum(total.generation_overlap_batches, successor->metrics.generation_overlap_batches);
            total.generation_overlap_members = sum(total.generation_overlap_members, successor->metrics.generation_overlap_members);
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, successor->metrics.peak_modeled_bytes);
            total.engine_reserved_bytes = std::max(total.engine_reserved_bytes, successor->metrics.engine_reserved_bytes);
            total.max_batch_members = std::max(total.max_batch_members, successor->metrics.max_batch_members);
            total.gpu_generation_peak_bytes = std::max(total.gpu_generation_peak_bytes, successor->metrics.gpu_generation_peak_bytes);
            total.peak_file_bytes = std::max(total.peak_file_bytes, charged);
            require(charged <= capacity, "successor output allocation exceeds aggregate capacity");
        }
        if (part.stop_reason == "capacity") { total.stop_reason = "capacity"; break; }
        if (!part.scratch_released) { total.scratch_released = false; total.stop_reason = "preserved-scratch"; break; }
        ++ordinal;
    }
    if (total.stop_reason.empty())
        total.stop_reason = total.members >= args.max_total_members ? "total-member-limit" : "group-limit";
    if (!backend)
    {
        uint64_t const metadata = sum(product(outputs.capacity(), sizeof(Output) * 4), mib);
        uint64_t const fixed = sum(group_grow::Engine::memory_bytes(keys.k,
            configuration(args, capacity, metadata), 1, 0,
            args.member_storage == "compact" ? 1 : 0), metadata);
        backend = std::make_unique<Backend>();
        total.peak_modeled_bytes = sum(fixed, backend->retained());
        total.engine_reserved_bytes = fixed;
        require(total.peak_modeled_bytes <= args.max_ram, "engine and retained GPU runtime exceed --max-ram");
    }
    print_metrics(args, total, *backend, outputs.size(), last_members, charged);
    return total.members ? 0 : 2;
}

struct FarmingRun
{
    struct Lane
    {
        std::vector<Output> outputs;
        uint64_t ordinal = 0, completed_charge = 0;
        uint64_t completed_members = 0, completed_bytes = 0;
        uint16_t reserved_members = 0, next_reserved_members = 0;
        std::optional<uint64_t> successor_ordinal; // Registered only while its engine owns the next output.
        bool no_capacity = false;
    };
    group_grow::FarmingState state;
    std::vector<Lane> lanes;
    std::vector<std::unique_ptr<DirectoryLock>> locks;
    fs::path manifest;
    std::vector<std::pair<fs::path, stream_state::Stamp>> manifests;
    std::vector<uint8_t> available;
    size_t previous_destinations = 0;
    group_grow::Stats audited;
    uint64_t metadata = 0, memory_limit = 0, manifest_charge = 0, planning_peak = 0;
    std::optional<group_grow::FarmingAssessment> admission;
    std::vector<std::vector<uint16_t>> certified_members;

    bool feasible() const { return state.estimate.feasible && (!admission || admission->feasible); }
    std::string const& infeasibility_reason() const
    { return state.estimate.feasible && admission ? admission->reason : state.estimate.reason; }
    void assessment_metrics(Metrics& metrics) const
    {
        metrics.farming_response_bound_seconds = state.estimate.response_bound_seconds;
        metrics.farming_dynamic_work_seconds_per_sp = state.estimate.dynamic_work_seconds_per_sp;
        metrics.farming_certified_max_eligible_groups = state.estimate.max_eligible_families_per_resource;
        if (admission)
        {
            metrics.farming_response_bound_seconds = std::max(metrics.farming_response_bound_seconds, admission->response_bound_seconds);
            metrics.farming_dynamic_work_seconds_per_sp = std::max(metrics.farming_dynamic_work_seconds_per_sp, admission->dynamic_work_seconds_per_sp);
            metrics.farming_certified_max_eligible_groups = std::max(metrics.farming_certified_max_eligible_groups,
                admission->max_eligible_families_per_resource);
        }
    }
    uint16_t membership(size_t lane, uint64_t ordinal) const
    {
        auto const& current = lanes[lane];
        auto const& layout = state.directories[lane].layout;
        // Extensions retain the complete promised forecast in every lane;
        // a faster lane cannot spend another lane's future service budget.
        uint16_t members = ordinal < layout.files ? layout.file_members(ordinal) : 0;
        if (ordinal < current.outputs.size())
            members = std::max({members, current.outputs[ordinal].members, current.outputs[ordinal].pending_members});
        if (ordinal == current.ordinal) members = std::max(members, current.reserved_members);
        if (ordinal == sum(current.ordinal, 1)) members = std::max(members, current.next_reserved_members);
        return members;
    }
    uint64_t membership_files(size_t lane) const
    {
        auto const& current = lanes[lane];
        uint64_t files = std::max(uint64_t(current.outputs.size()), state.directories[lane].layout.files);
        if (current.reserved_members) files = std::max(files, sum(current.ordinal, 1));
        if (current.next_reserved_members) files = std::max(files, sum(current.ordinal, 2));
        while (files && !membership(lane, files - 1)) --files;
        return files;
    }
    bool covered() const
    {
        bool planned = state.estimate.feasible, certified = certified_members.size() == lanes.size();
        for (size_t lane = 0; lane < lanes.size(); ++lane)
        {
            auto const files = membership_files(lane); auto const& layout = state.directories[lane].layout;
            planned = planned && files <= layout.files;
            certified = certified && files <= certified_members[lane].size();
            for (uint64_t ordinal = 0; ordinal < files && (planned || certified); ++ordinal)
            {
                auto const members = membership(lane, ordinal);
                planned = planned && members && members <= layout.file_members(ordinal);
                certified = certified && members && members <= certified_members[lane][ordinal];
            }
        }
        return planned || certified;
    }
    uint64_t certificate_memory() const
    {
        uint64_t bytes = product(certified_members.capacity(), sizeof(std::vector<uint16_t>));
        for (auto const& members : certified_members) bytes = sum(bytes, product(members.capacity(), sizeof(uint16_t)));
        return bytes;
    }
    void certify(uint64_t occupied)
    {
        if (covered()) return;
        uint64_t inputs = product(lanes.size(), 2 * (sizeof(group_grow::FarmingDestination) +
            sizeof(group_grow::FarmingDrivePlan) + sizeof(std::vector<uint16_t>)));
        for (size_t lane = 0; lane < lanes.size(); ++lane)
            inputs = sum(inputs, product(membership_files(lane), 2 * sizeof(uint16_t)));
        if (occupied >= memory_limit || inputs >= memory_limit - occupied)
            throw group_grow::FarmingResourceError("frozen farming assessment input exceeds --max-ram", true);
        std::vector<group_grow::FarmingDestination> destinations; destinations.reserve(lanes.size());
        std::vector<group_grow::FarmingDrivePlan> layouts; layouts.reserve(lanes.size());
        std::vector<std::vector<uint16_t>> members(lanes.size());
        for (size_t lane = 0; lane < lanes.size(); ++lane)
        {
            destinations.push_back(state.directories[lane].storage); layouts.push_back(state.directories[lane].layout);
            auto const files = membership_files(lane); members[lane].reserve(size_t(files));
            for (uint64_t ordinal = 0; ordinal < files; ++ordinal) members[lane].push_back(membership(lane, ordinal));
        }
        admission = group_grow::farming_assess(state.config, destinations, layouts, members, memory_limit - occupied - inputs);
        planning_peak = std::max(planning_peak, sum(sum(occupied, inputs), admission->planning_memory_bytes));
        require(planning_peak <= memory_limit, "frozen farming assessment exceeds --max-ram");
        if (admission->feasible)
        {
            auto const old = certificate_memory(); certified_members = std::move(members);
            metadata = sum(metadata - old, certificate_memory());
        }
    }

    Identity identity(Keys const& keys, size_t lane, uint64_t ordinal) const
    {
        auto const& directory = state.directories[lane];
        auto const& layout = directory.layout;
        auto const resources = directory.identity_resources, resource = directory.identity_resource;
        // Check the interleave before the policy accessor multiplies its ordinal.
        (void)sum(product(ordinal, layout.resource_lanes), layout.resource_rank);
        uint64_t const family = sum(product(layout.family(ordinal), resources), resource);
        auto farming_keys = keys;
        struct sha256 hash; sha256_init(&hash);
        std::string_view const domain = "xchplot2 directory farming v1";
        sha256_update(&hash, domain.data(), domain.size());
        sha256_update(&hash, keys.seed.data(), keys.seed.size());
        if (directory.identity_epoch)
        {
            // Expansion gets its own key namespace without changing the
            // original family interleave, including on an offline member.
            std::string const epoch = " expansion " + std::to_string(directory.identity_epoch);
            sha256_update(&hash, epoch.data(), epoch.size());
        }
        sha256_sum(&hash, farming_keys.seed.data());
        auto value = farming_keys.derive(family);
        value.params = PlotGroupParams(value.params.get_plot_group_id(), keys.k, keys.strength, layout.meta(ordinal));
        return value;
    }
    std::string stop_reason(Arguments const& args) const
    {
        if (std::any_of(lanes.begin(), lanes.end(), [](auto const& lane) { return lane.no_capacity; })) return "capacity";
        if (std::all_of(lanes.begin(), lanes.end(), [&](auto const& lane) { return lane.completed_members >= args.max_total_members; }))
            return "total-member-limit";
        return "group-limit";
    }
    fs::path path(size_t lane) const { return group_path(file_io::native_path(state.directories[lane].directory), lanes[lane].ordinal); }
    uint16_t file_maximum(size_t lane, uint64_t ordinal) const
    {
        auto const& layout = state.directories[lane].layout;
        if (state.fixed_extent) return ordinal < layout.files ? layout.file_members(ordinal) : 0;
        // The forecast terminal file may be partial. A fixed rollover size
        // still allows that same file to grow when the total limit increases.
        return layout.fixed_members ? layout.group_members : layout.file_members(ordinal);
    }
    uint16_t maximum(Arguments const& args, size_t lane) const
    {
        auto const& current = lanes[lane];
        if (state.fixed_extent && current.ordinal >= state.directories[lane].layout.files) return 0;
        if (current.no_capacity || current.ordinal >= args.max_groups || current.completed_members >= args.max_total_members ||
            current.completed_charge >= state.directories[lane].capacity_bytes) return 0;
        return uint16_t(std::min<uint64_t>(file_maximum(lane, current.ordinal),
            args.max_total_members - current.completed_members));
    }
    void check(std::span<uint8_t const> active = {}) const
    {
        for (auto const& [path, stamp] : manifests)
            require(stream_state::stamp(path) == stamp, "farming plan changed during the run");
        for (size_t i = 0; i < lanes.size(); ++i)
        {
            if (!available.empty() && !available[i]) continue;
            if (!locks.empty() && locks[i]) locks[i]->check();
            auto const& lane = lanes[i];
            auto const& directory = state.directories[i].directory;
            bool const writing = !active.empty() && active[i];
            require(!lane.successor_ordinal || *lane.successor_ordinal == sum(lane.ordinal, 1),
                "live farming successor is not the next ordinal");
            auto const successor = writing ? lane.successor_ordinal : std::nullopt;
            std::string const successor_name = successor ? file_io::path_text(group_path(file_io::native_path(directory), *successor).filename()) : "";
            size_t seen = 0;
            for (auto const& entry : fs::directory_iterator(file_io::native_path(directory)))
            {
                auto owned = writing ? std::optional(lane.ordinal) : std::nullopt;
                if (successor && file_io::path_text(entry.path().filename()).starts_with(successor_name)) owned = successor;
                auto const ordinal = rollover_entry(directory, entry.path(), memory_limit,
                    owned);
                if (!ordinal) continue;
                require(*ordinal < lane.outputs.size() ||
                    (writing && *ordinal == lane.ordinal) || (successor && *ordinal == *successor),
                    "unplanned farming output appeared during the run");
                ++seen;
            }
            size_t expected = lane.outputs.size();
            if (writing && lane.ordinal == expected) ++expected;
            if (successor && *successor == expected) ++expected;
            require(seen == expected, "farming output disappeared during the run");
            for (auto const& saved : lane.outputs)
            {
                if ((writing && saved.ordinal == lane.ordinal) || (successor && saved.ordinal == *successor)) continue;
                auto const now = snapshot(group_path(file_io::native_path(directory), saved.ordinal), saved.ordinal);
                require(now.device == saved.device && now.inode == saved.inode && now.bytes == saved.bytes &&
                    now.allocated == saved.allocated && now.modified_seconds == saved.modified_seconds &&
                    now.modified_nanoseconds == saved.modified_nanoseconds && now.changed_seconds == saved.changed_seconds &&
                    now.changed_nanoseconds == saved.changed_nanoseconds, "saved farming output changed during the run");
            }
        }
    }

    void select_available()
    {
        for (size_t i = available.size(); i-- > 0;) if (!available[i])
        {
            state.directories.erase(state.directories.begin() + i);
            lanes.erase(lanes.begin() + i);
            if (!locks.empty()) locks.erase(locks.begin() + i);
            available.erase(available.begin() + i);
        }
    }
};

group_grow::FarmingDestination farming_destination_profile(group_grow::StorageDevice const& device,
    group_grow::FarmingDestination destination = {})
{
    destination.known_device = !device.drives.empty() && (device.rotating || device.nonrotating);
    destination.nonrotating = destination.known_device && device.nonrotating;
    return group_grow::farming_storage_profile(destination);
}
group_grow::FarmingConfig farming_configuration(Arguments const& args)
{
    group_grow::FarmingConfig config;
    config.k = uint8_t(args.k.value_or(28)); config.strength = uint8_t(args.strength.value_or(2));
    config.group_members_override = uint16_t(args.group_members);
    config.fixed_meta = args.meta.has_value(); config.meta_group_override = uint8_t(args.meta.value_or(0));
    if (args.farming_reader) config.reader = *args.farming_reader == "cached" ?
        group_grow::FarmingReader::cached : group_grow::FarmingReader::stateless;
    if (args.farming_objective) config.objective = *args.farming_objective == "latency" ?
        group_grow::FarmingObjective::latency : group_grow::FarmingObjective::energy;
    if (args.farming_cpu_work_weight) config.cpu_work_weight = static_cast<long double>(*args.farming_cpu_work_weight);
    if (args.farming_io_work_weight) config.io_work_weight = static_cast<long double>(*args.farming_io_work_weight);
    if (args.farming_max_eligible_groups) config.max_eligible_families_per_resource = *args.farming_max_eligible_groups;
    if (args.farming_lookup_workers) config.lookup_workers = unsigned(*args.farming_lookup_workers);
    if (args.farming_memory) config.memory_bytes = *args.farming_memory;
    if (args.farming_cores) config.harvester_cores = unsigned(*args.farming_cores);
    if (args.farming_index_cache) config.index_cache_bytes = *args.farming_index_cache;
    if (args.farming_filter_bits) config.filter_bits = uint8_t(*args.farming_filter_bits);
    if (args.farming_future_filter_bits) config.future_filter_bits = uint8_t(*args.farming_future_filter_bits);
    if (args.farming_chain_us) config.chain_seconds_per_member = static_cast<long double>(*args.farming_chain_us) / 1000000;
    if (args.farming_index_ns) config.index_seconds_per_entry = static_cast<long double>(*args.farming_index_ns) / 1000000000;
    if (args.farming_deadline_ms) config.deadline_seconds = static_cast<long double>(*args.farming_deadline_ms) / 1000;
    return config;
}
void farming_requested(Arguments const& args, group_grow::FarmingConfig const& saved)
{
    require(!args.k || *args.k == saved.k, "--k differs from the saved farming plan");
    require(!args.strength || *args.strength == saved.strength, "--strength differs from the saved farming plan");
    require(!args.farming_reader || (*args.farming_reader == "cached") == (saved.reader == group_grow::FarmingReader::cached),
        "--farming-reader differs from the saved farming plan");
    require(!args.farming_objective || (*args.farming_objective == "latency") == (saved.objective == group_grow::FarmingObjective::latency),
        "--farming-objective differs from the saved farming plan");
    require(!args.farming_cpu_work_weight || static_cast<long double>(*args.farming_cpu_work_weight) == saved.cpu_work_weight,
        "--farming-cpu-work-weight differs from the saved farming plan");
    require(!args.farming_io_work_weight || static_cast<long double>(*args.farming_io_work_weight) == saved.io_work_weight,
        "--farming-io-work-weight differs from the saved farming plan");
    require(!args.farming_max_eligible_groups || *args.farming_max_eligible_groups == saved.max_eligible_families_per_resource,
        "--farming-max-eligible-groups differs from the saved farming plan");
    require(!args.farming_cores || *args.farming_cores == saved.harvester_cores, "--farming-cores differs from the saved farming plan");
    require(!args.farming_lookup_workers || *args.farming_lookup_workers == saved.lookup_workers,
        "--farming-lookup-workers differs from the saved farming plan");
    require(!args.farming_memory || *args.farming_memory == saved.memory_bytes, "--farming-memory-bytes differs from the saved farming plan");
    require(!args.farming_index_cache || *args.farming_index_cache == saved.index_cache_bytes,
        "--farming-index-cache-bytes differs from the saved farming plan");
    require(!args.farming_filter_bits || *args.farming_filter_bits == group_grow::farming_filter_bits(saved),
        "--farming-filter-bits differs from the saved farming plan");
    require(!args.farming_future_filter_bits || *args.farming_future_filter_bits == group_grow::farming_filter_bits(saved, true),
        "--farming-future-filter-bits differs from the saved farming plan");
    require(!args.farming_chain_us || static_cast<long double>(*args.farming_chain_us) / 1000000 == saved.chain_seconds_per_member,
        "--farming-chain-us-per-member differs from the saved farming plan");
    require(!args.farming_index_ns || static_cast<long double>(*args.farming_index_ns) / 1000000000 == saved.index_seconds_per_entry,
        "--farming-index-ns-per-entry differs from the saved farming plan");
    require(!args.farming_deadline_ms || static_cast<long double>(*args.farming_deadline_ms) / 1000 == saved.deadline_seconds,
        "--farming-deadline-ms differs from the saved farming plan");
    require(!args.group_auto || !saved.group_members_override, "--group-members auto differs from the saved farming plan");
    require(!args.group_members || args.group_members == saved.group_members_override,
        "--group-members differs from the saved farming plan");
    require(!args.meta || (saved.fixed_meta && *args.meta == saved.meta_group_override),
        "--meta-group differs from the saved farming plan");
}
std::string farming_fingerprint(Keys const& keys)
{
    auto const identity = keys.derive(0);
    std::ostringstream value;
    value << group_tools::hex(identity.params.get_plot_group_id().span()) << ' ' << unsigned(keys.k) << ' ' << unsigned(keys.strength)
        << ' ' << group_tools::hex(identity.memo);
    return stream_state::digest(value.str());
}

struct FarmingJob
{
    struct Copy
    {
        fs::path path;
        std::optional<stream_state::Stamp> stamp;
        std::string digest;
        bool pending = false;

        void check() const
        {
            if (stamp)
                require(stream_state::stamp(path) == *stamp, "farming job changed while acquiring its directories");
            else
            {
                file_io::Stat info{};
                require(file_io::inspect_link(path.c_str(), &info) != 0 && errno == ENOENT,
                    "farming job appeared while acquiring its directories");
            }
        }
    };
    std::optional<group_grow::FarmingState> saved;
    std::vector<Copy> copies;
    std::vector<fs::path> directories;
    std::set<size_t> relocated;
    std::string digest;
    bool pending = false, published = false, changed = false;

    void check() const
    {
        for (auto const& copy : copies) copy.check();
    }
};

FarmingJob farming_arguments(Arguments& args)
{
    FarmingJob job;
    std::vector<fs::path> requested;
    if (!args.target.empty()) requested.push_back(args.target);
    requested.insert(requested.end(), args.destinations.begin(), args.destinations.end());
    if (requested.empty() && args.manifest.empty() && args.relocations.empty()) requested.push_back(fs::current_path());
    for (auto& path : requested) path = fs::weakly_canonical(path);
    std::sort(requested.begin(), requested.end());
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    auto same_directory = [](fs::path const& a, fs::path const& b)
    {
        if (a == b) return true;
        std::error_code error;
        return fs::equivalent(a, b, error) && !error;
    };

    std::vector<fs::path> directories = requested;
    for (auto const& [old, replacement] : args.relocations) directories.push_back(fs::canonical(replacement));
    std::map<std::string, group_grow::FarmingState> revisions;
    std::set<fs::path> visited;
    auto read = [&](fs::path path, bool pending)
    {
        path = fs::absolute(path).lexically_normal();
        if (!visited.insert(path).second) return;
        file_io::Stat info{};
        if (file_io::inspect_link(path.c_str(), &info))
        {
            require(errno == ENOENT, "cannot inspect farming job: " + file_io::path_text(path));
            job.copies.push_back({path, {}, {}, pending}); return;
        }
        auto const stamp = stream_state::stamp(path);
        auto state = group_grow::load_farming_state(path, args.max_ram);
        require(stream_state::stamp(path) == stamp, "farming job changed while loading");
        auto const digest = stream_state::digest(group_grow::farming_state_text(state));
        job.copies.push_back({path, stamp, digest, pending});
        if (!revisions.empty()) require(revisions.begin()->second.job_id == state.job_id,
            "destinations contain different farming jobs; select one job with --manifest and its own destinations");
        if (revisions.contains(digest)) return;
        require(revisions.size() < 2, "conflicting farming revisions; restore a consistent job before continuing");
        for (auto const& d : state.directories) directories.push_back(file_io::native_path(d.directory));
        revisions.emplace(digest, std::move(state));
    };
    if (!args.manifest.empty())
    {
        require(fs::is_regular_file(args.manifest), "--manifest requires an existing farming job file");
        read(args.manifest, args.manifest.extension() == ".next");
    }
    std::set<fs::path> seen;
    for (size_t i = 0; i < directories.size(); ++i)
    {
        auto const directory = directories[i];
        if (!seen.insert(directory).second) continue;
        read(directory / "grow-farm.plan", false);
        read(directory / "grow-farm.plan.next", true);
    }
    if (!revisions.empty())
    {
        auto selected = revisions.begin();
        for (auto at = revisions.begin(); at != revisions.end(); ++at)
            if (at->second.revision > selected->second.revision) selected = at;
        if (revisions.size() == 2)
        {
            auto const previous = selected == revisions.begin() ? std::next(selected) : revisions.begin();
            auto const& before = previous->second; auto const& after = selected->second;
            require(before.revision < UINT64_MAX && after.revision == before.revision + 1 &&
                after.parent_digest == previous->first && after.seed == before.seed && after.seed_owner == before.seed_owner &&
                after.farmer == before.farmer && after.pool == before.pool && after.pool_public_key == before.pool_public_key &&
                after.config == before.config && after.fixed_extent == before.fixed_extent &&
                after.directories.size() >= before.directories.size(), "conflicting farming revisions; no files were changed");
            for (size_t i = 0; i < before.directories.size(); ++i)
            {
                auto const& a = before.directories[i]; auto const& b = after.directories[i];
                require(a.layout == b.layout && a.identity_resource == b.identity_resource &&
                    a.identity_resources == b.identity_resources && a.identity_epoch == b.identity_epoch,
                    "farming revision changed an existing identity assignment");
            }
        }
        job.digest = selected->first; job.saved = std::move(selected->second);
        auto const& saved = *job.saved;
        for (auto const& copy : job.copies) if (copy.stamp)
        {
            job.pending = job.pending || copy.pending || copy.digest != job.digest;
            job.published = job.published || (!copy.pending && copy.digest == job.digest);
        }
        require(args.resume || args.plan_only, "farming job exists; use --resume or --extend");
        farming_requested(args, saved.config);
        if (!args.seed) args.seed = saved.seed;
        if (!args.farmer) args.farmer = saved.farmer;
        if (!args.pool_pk && !args.pool_ph && !args.pool_address)
        {
            if (saved.pool_public_key) args.pool_pk = saved.pool;
            else args.pool_ph = saved.pool;
        }
        args.seed_owner = saved.seed_owner;
        args.k = saved.config.k; args.strength = saved.config.strength;
        if (saved.config.fixed_meta) args.meta = saved.config.meta_group_override;
        if (!args.group_members && !args.group_auto) args.group_members = saved.config.group_members_override;
        auto limit = [&](bool specified, uint64_t& value, uint64_t original, char const* name)
        {
            require(!saved.fixed_extent || !specified || value == original,
                std::string(name) + " differs from the saved job; resume preserves its planned extent");
            if (!specified) value = original;
        };
        limit(args.capacity_set, args.capacity, saved.capacity, "--capacity-bytes");
        limit(args.headroom_set, args.headroom, saved.headroom, "--headroom-bytes");
        limit(args.max_groups_set, args.max_groups, saved.max_groups, "--max-groups");
        limit(args.max_total_members_set, args.max_total_members, saved.max_total_members, "--max-total-members");
        for (auto const& d : saved.directories) job.directories.push_back(file_io::native_path(d.directory));
        for (auto const& [old, replacement] : args.relocations)
        {
            auto at = std::find_if(job.directories.begin(), job.directories.end(),
                [&](auto const& directory) { return same_directory(directory, old); });
            require(at != job.directories.end(), "--relocate source does not belong to this job: " + file_io::path_text(old));
            auto path = fs::canonical(replacement);
            require(std::any_of(job.copies.begin(), job.copies.end(), [&](auto const& copy)
                { return copy.stamp && same_directory(copy.path.parent_path(), path); }),
                "relocated destination must contain its saved grow-farm.plan or grow-farm.plan.next");
            require(!fs::exists(old) || fs::equivalent(old, path),
                "old destination is still present; take it offline before relocating its job");
            auto const index = size_t(at - job.directories.begin());
            require(job.relocated.insert(index).second, "destination relocated more than once");
            *at = std::move(path); job.changed = true;
        }
        for (auto const& path : requested)
        {
            if (std::any_of(job.directories.begin(), job.directories.end(),
                [&](auto const& directory) { return same_directory(directory, path); })) continue;
            bool const old_location = std::any_of(args.relocations.begin(), args.relocations.end(),
                [&](auto const& relocation) { return relocation.first == path; });
            if (old_location) continue;
            require(args.extend || args.plan_only,
                "destination is not in the saved job; preview with plan, then add it with fill --extend: " + file_io::path_text(path));
            require(saved.fixed_extent, "reference farming jobs cannot add destinations; start a managed fill job");
            job.directories.push_back(path); job.changed = true;
        }
        require(!job.changed || !job.pending, "recover the pending farming revision with --resume before changing destinations");
    }
    else
    {
        require(!args.resume && !args.extend && args.relocations.empty(), "no saved farming job found; supply any member directory or --manifest FILE");
        job.directories = std::move(requested);
        if (!args.seed && !args.plan_only)
        {
            std::array<uint8_t, 32> seed{}; pos2gpu::read_random_bytes(seed.data(), seed.size());
            args.seed = group_tools::hex(seed);
        }
    }
    require(!job.directories.empty() && job.directories.size() <= 1024, "farming job requires 1..1024 destinations");
    std::set<fs::path> unique;
    std::set<std::string> unique_identities;
    for (auto& path : job.directories)
    {
        path = fs::weakly_canonical(path);
        require(unique.insert(path).second, "duplicate destination in saved farming job");
        if (fs::is_directory(path)) require(unique_identities.insert(group_grow::farming_directory_identity(path)).second,
            "destination paths refer to the same directory");
    }
    args.target = job.directories.front();
    args.destinations.assign(job.directories.begin() + 1, job.directories.end());
    return job;
}
std::unique_ptr<FarmingRun> farming_run(Arguments const& args, Keys const* keys, FarmingJob const& job)
{
    auto run = std::make_unique<FarmingRun>();
    auto const& directories = job.directories;
    size_t const previous = job.saved ? job.saved->directories.size() : 0;
    run->previous_destinations = previous;
    run->lanes.resize(directories.size()); run->available.resize(directories.size());
    run->metadata = sum(product(directories.size(), mib), path_metadata(directories));
    run->memory_limit = args.max_ram;
    require(run->metadata < args.max_ram, "farming destination metadata exceeds --max-ram");
    std::vector<group_grow::StorageDevice> devices(directories.size());
    std::vector<std::string> identities(directories.size());
    std::vector<fs::path> missing;
    for (size_t i = 0; i < directories.size(); ++i)
    {
        std::error_code error;
        bool present = fs::is_directory(directories[i], error);
        require(!error || error == std::errc::no_such_file_or_directory,
            "cannot inspect farming destination: " + file_io::path_text(directories[i]));
        if (present)
        {
            identities[i] = group_grow::farming_directory_identity(directories[i]);
            if (i < previous && !job.relocated.contains(i) &&
                identities[i] != job.saved->directories[i].directory_identity) present = false;
        }
        if (!present)
        {
            require(i < previous, "new destination must be an existing directory: " + file_io::path_text(directories[i]));
            missing.push_back(directories[i]);
        }
        else
        {
            run->available[i] = 1;
            devices[i] = group_grow::storage_device(directories[i]);
        }
    }
    if (!missing.empty())
    {
        std::string message = "unavailable or replaced farming destinations:";
        for (auto const& path : missing) message += "\n  " + file_io::path_text(path);
        message += "\nMount the original directories, or use --relocate OLD NEW for an intentional move.";
        require(args.plan_only || (args.allow_missing && job.saved->fixed_extent && !job.changed &&
            (!job.pending || job.published)), message + " Independent recovery can use --resume --allow-missing.");
    }
    if (!args.plan_only)
    {
        std::vector<size_t> order;
        for (size_t i = 0; i < directories.size(); ++i) if (run->available[i]) order.push_back(i);
        require(!order.empty(), "no saved farming destination is available");
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return directories[a] < directories[b]; });
        run->locks.resize(directories.size());
        for (auto i : order) run->locks[i] = std::make_unique<DirectoryLock>(directories[i]);
    }
    job.check();
    for (size_t i = 0; i < directories.size(); ++i) if (run->available[i])
        require(group_grow::farming_directory_identity(directories[i]) == identities[i],
            "farming destination changed while acquiring its lock");
    std::map<std::string, uint64_t> allocated, space_available;
    std::vector<uint64_t> plan_charges(directories.size());
    uint64_t alignment = 1;
    for (size_t i = 0; i < directories.size(); ++i)
    {
        if (!run->available[i]) continue;
        auto& lane = run->lanes[i];
        for (auto const& entry : fs::directory_iterator(directories[i]))
        {
            auto const ordinal = rollover_entry(directories[i], entry.path(), args.max_ram - run->metadata);
            if (!ordinal) continue;
            require(i < previous, "new destination contains rollover outputs without its saved farming job");
            lane.outputs.push_back(snapshot(entry.path(), *ordinal));
            run->metadata = sum(run->metadata, sizeof(Output) * 4);
            require(run->metadata < args.max_ram / 2, "saved farming output metadata exceeds --max-ram");
        }
        std::sort(lane.outputs.begin(), lane.outputs.end(), [](auto const& a, auto const& b) { return a.ordinal < b.ordinal; });
        for (size_t ordinal = 0; ordinal < lane.outputs.size(); ++ordinal)
        {
            require(lane.outputs[ordinal].ordinal == ordinal, "saved farming output sequence has a hole");
            output_auxiliary(group_path(directories[i], ordinal), lane.outputs[ordinal], args.max_ram - run->metadata);
            allocated[devices[i].filesystem] = sum(allocated[devices[i].filesystem],
                sum(lane.outputs[ordinal].allocated, lane.outputs[ordinal].auxiliary_allocated));
        }
        for (auto const& copy : job.copies) if (copy.stamp && copy.path.parent_path() == directories[i])
        {
            file_io::Stat info{};
            require(!file_io::inspect_link(copy.path.c_str(), &info) && file_io::is_regular(info.st_mode) &&
                info.st_blocks >= 0 && info.st_size >= 0, "cannot inspect farming-plan allocation");
            plan_charges[i] = sum(plan_charges[i], std::max(uint64_t(info.st_size), product(uint64_t(info.st_blocks), 512)));
        }
        allocated[devices[i].filesystem] = sum(allocated[devices[i].filesystem], plan_charges[i]);
        file_io::Space space{};
        require(!file_io::space_path(directories[i].c_str(), &space), "cannot inspect farming destination capacity");
        uint64_t const bytes = product(space.f_bavail, space.f_frsize);
        auto [where, inserted] = space_available.emplace(devices[i].filesystem, bytes);
        if (!inserted) where->second = std::min(where->second, bytes);
        alignment = std::max({alignment, uint64_t(space.f_bsize), uint64_t(space.f_frsize)});
    }
    if (job.saved)
    {
        run->state = *job.saved;
        require(!keys || run->state.fingerprint == farming_fingerprint(*keys), "keys or plot parameters differ from the saved farming job");
    }
    else
    {
        run->state.config = farming_configuration(args);
        run->state.fingerprint = keys ? farming_fingerprint(*keys) : std::string(64, '0');
        std::array<uint8_t, 32> id{};
        if (!args.plan_only) pos2gpu::read_random_bytes(id.data(), id.size());
        run->state.job_id = group_tools::hex(id);
        run->state.seed = args.seed.value_or(std::string(64, '0'));
        run->state.seed_owner = args.seed_owner.value_or(args.pool_broker.empty() ? "" : args.pool_owner);
        run->state.farmer = keys ? group_tools::hex(keys->farmer) : std::string(96, '0');
        run->state.pool = keys ? group_tools::hex(keys->pool) : std::string(96, '0');
        run->state.pool_public_key = keys ? keys->pool_kind == POS2_POOL_PK : true;
        run->state.fixed_extent = !args.command.empty();
        run->state.capacity = args.capacity; run->state.headroom = args.headroom;
        run->state.max_groups = args.max_groups; run->state.max_total_members = args.max_total_members;
    }
    // Audit every saved output before writing the plan or confirming a group.
    for (size_t i = 0; i < previous; ++i)
    {
        if (!run->available[i]) continue;
        auto& lane = run->lanes[i];
        require(lane.outputs.size() <= args.max_groups, "saved farming outputs exceed --max-groups");
        uint64_t audited_members = 0, audited_charge = 0;
        std::optional<uint64_t> unfinished;
        bool prior_external_complete = false;
        for (size_t ordinal = 0; ordinal < lane.outputs.size(); ++ordinal)
        {
            auto& output = lane.outputs[ordinal];
            Construction audit(file_io::path_text(group_path(directories[i], ordinal)),
                configuration(args, uint64_t(INT64_MAX), run->metadata), true);
            auto const saved = audit.identity();
            if (keys) same_identity(run->identity(*keys, i, ordinal), identity_from(saved));
            uint16_t const maximum = uint16_t(std::min<uint64_t>(run->file_maximum(i, ordinal),
                args.max_total_members - std::min(args.max_total_members, audited_members)));
            require(saved.header.group_size <= maximum && (!saved.pending || saved.target_members <= maximum),
                "saved membership differs from the farming plan");
            output.members = audit.member_count(); output.pending = saved.pending || audit.is_staging();
            output.pending_members = saved.pending ? saved.target_members : output.members;
            if (!audit.external) output.reservation = std::max(saved.bytes, saved.pending_peak_bytes);
            require(!output.pending || maximum, "pending farming output requires room under --max-total-members");
            bool const full = !output.pending && output.members == maximum;
            // Finish may not have saved its pack phase before the successor
            // starts. Full committed external membership is the durable fence;
            // publication and cleanup still resume at this earlier ordinal.
            if (unfinished)
                require(ordinal == sum(*unfinished, 1) && ordinal + 1 == lane.outputs.size() &&
                    prior_external_complete && audit.external && audit.external->can_append(),
                    "only consecutive external farming outputs with a full prior membership may be incomplete");
            else if (!full)
            {
                unfinished = ordinal; lane.ordinal = ordinal;
                prior_external_complete = audit.external && output.members == run->file_maximum(i, ordinal);
            }
            run->audited = Construction::add_stats(run->audited, audit.stats());
            audited_members = sum(audited_members, output.pending_members);
            audited_charge = sum(audited_charge, output.charge());
            require(audited_members <= args.max_total_members, "saved farming transactions exceed --max-total-members");
            if (!unfinished)
            {
                lane.completed_charge = sum(lane.completed_charge, output.charge());
                lane.completed_members = sum(lane.completed_members, output.members);
                lane.completed_bytes = sum(lane.completed_bytes, output.bytes); lane.ordinal = ordinal + 1;
            }
            else if (ordinal == *unfinished) lane.reserved_members = output.pending_members;
            else lane.next_reserved_members = output.pending_members;
        }
        require(audited_charge <= run->state.directories[i].capacity_bytes, "saved farming outputs exceed planned capacity");
        lane.no_capacity = lane.ordinal == lane.outputs.size() &&
            (space_available.at(devices[i].filesystem) <= args.headroom || !run->state.directories[i].layout.total_members);
    }
    if (job.changed)
    {
        require(run->state.revision < UINT64_MAX, "farming job revision exhausted");
        ++run->state.revision; run->state.parent_digest = job.digest;
    }
    bool const planning = !job.saved || job.changed;
    if (planning)
    {
        require(missing.empty(), "all existing destinations must be available to plan an expansion or relocation");
        auto const resources = group_grow::writer_groups(devices);
        auto const resource_count = *std::max_element(resources.begin(), resources.end()) + 1;
        std::map<std::string, uint64_t> assigned;
        std::map<std::string, size_t> shares;
        std::vector<bool> completed(previous);
        for (size_t i = 0; i < previous; ++i)
        {
            completed[i] = run->state.fixed_extent && run->lanes[i].ordinal >= run->state.directories[i].layout.files;
            if (completed[i]) run->state.directories[i].capacity_bytes =
                sum(run->lanes[i].completed_charge, run->state.manifest_reserved_bytes);
            assigned[devices[i].filesystem] = sum(assigned[devices[i].filesystem], run->state.directories[i].capacity_bytes);
        }
        for (size_t i = previous; i < directories.size(); ++i) ++shares[devices[i].filesystem];
        std::vector<group_grow::FarmingDrivePlan> frozen;
        for (auto const& d : run->state.directories) frozen.push_back(d.layout);
        for (size_t i = 0; i < directories.size(); ++i)
        {
            if (i >= previous)
            {
                uint64_t const bytes = sum(space_available.at(devices[i].filesystem), allocated[devices[i].filesystem]);
                require(bytes > args.headroom, "farming destination has no capacity above headroom");
                uint64_t const total = args.capacity ? std::min(args.capacity, bytes - args.headroom) : bytes - args.headroom;
                require(total > assigned[devices[i].filesystem],
                    "new destination has no unassigned capacity on its filesystem; another directory does not add disk space");
                uint64_t const capacity = (total - assigned[devices[i].filesystem]) / shares.at(devices[i].filesystem);
                group_grow::FarmingDirectory d;
                d.capacity_bytes = capacity;
                d.identity_resources = resource_count; d.identity_resource = resources[i];
                d.identity_epoch = run->state.revision - 1;
                d.storage.max_file_bytes = group_grow::maximum_file_bytes(file_io::path_text(directories[i]));
                d.storage.max_total_members = args.max_total_members; d.storage.max_groups = args.max_groups;
                run->state.directories.push_back(std::move(d));
            }
            auto& d = run->state.directories[i];
            d.directory = file_io::path_text(directories[i]); d.directory_identity = identities[i];
            d.filesystem = devices[i].filesystem; d.resource = resources[i];
            d.storage.resource = d.resource; d.storage = farming_destination_profile(devices[i], d.storage);
        }
        for (;;)
        {
            // Finished fixed assignments need only their actual output and
            // revision storage. Release unused quota for additional directories
            // on that filesystem without changing any completed plot.
            assigned.clear();
            for (size_t i = 0; i < previous; ++i)
            {
                if (completed[i]) run->state.directories[i].capacity_bytes =
                    sum(run->lanes[i].completed_charge, run->state.manifest_reserved_bytes);
                assigned[devices[i].filesystem] = sum(assigned[devices[i].filesystem], run->state.directories[i].capacity_bytes);
            }
            for (size_t i = previous; i < directories.size(); ++i)
            {
                uint64_t const bytes = sum(space_available.at(devices[i].filesystem), allocated[devices[i].filesystem]);
                uint64_t const total = args.capacity ? std::min(args.capacity, bytes - args.headroom) : bytes - args.headroom;
                require(total > assigned[devices[i].filesystem], "filesystem has no unassigned capacity for additional destinations");
                run->state.directories[i].capacity_bytes = (total - assigned[devices[i].filesystem]) / shares.at(devices[i].filesystem);
            }
            std::vector<group_grow::FarmingDestination> destinations;
            for (auto& d : run->state.directories)
            {
                require(d.capacity_bytes > run->state.manifest_reserved_bytes, "destination cannot hold its farming job");
                d.storage.capacity_bytes = d.capacity_bytes - run->state.manifest_reserved_bytes;
                destinations.push_back(d.storage);
            }
            run->state.estimate = group_grow::farming_plan(run->state.config, destinations,
                args.max_ram - run->metadata, frozen);
            run->planning_peak = std::max(run->planning_peak, sum(run->metadata, run->state.estimate.planning_memory_bytes));
            require(run->planning_peak <= args.max_ram, "farming planner exceeds --max-ram");
            for (size_t i = 0; i < directories.size(); ++i) run->state.directories[i].layout = run->state.estimate.destinations[i];
            // Reserve the published copy and one durable replacement on each
            // destination. Keyless previews allow for the largest identity form.
            auto reservation = run->state;
            reservation.pool.assign(96, '0'); reservation.seed_owner.assign(128, 'x');
            reservation.parent_digest.assign(64, '0');
            uint64_t const bytes = sum(group_grow::farming_state_text(reservation).size(), 65 + 1024);
            uint64_t const needed = product(product(bytes / alignment + bool(bytes % alignment), alignment), 2);
            if (needed <= run->state.manifest_reserved_bytes) break;
            run->state.manifest_reserved_bytes = needed;
        }
    }
    else
    {
        std::vector<group_grow::StorageDevice> present;
        std::vector<size_t> indices;
        for (size_t i = 0; i < directories.size(); ++i)
            if (run->available[i]) { present.push_back(devices[i]); indices.push_back(i); }
        auto const resources = group_grow::writer_groups(present);
        for (size_t i = 0; i < indices.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                require((resources[i] == resources[j]) ==
                    (run->state.directories[indices[i]].resource == run->state.directories[indices[j]].resource),
                    "destination storage topology changed; inspect the saved job before continuing");
    }
    std::map<std::string, uint64_t> preflight_charge, replacement_bytes;
    for (size_t i = 0; i < directories.size(); ++i) if (run->available[i])
    {
        auto const path = directories[i] / "grow-farm.plan";
        auto copy = std::find_if(job.copies.begin(), job.copies.end(), [&](auto const& value) { return value.path == path; });
        bool const publishing = planning || job.pending || copy == job.copies.end() || !copy->stamp;
        uint64_t const metadata_charge = publishing ? std::max(plan_charges[i], run->state.manifest_reserved_bytes) : plan_charges[i];
        uint64_t charge = metadata_charge;
        for (auto const& output : run->lanes[i].outputs) charge = sum(charge, output.charge());
        require(charge <= run->state.directories[i].capacity_bytes, "farming job and outputs exceed destination capacity");
        preflight_charge[devices[i].filesystem] = sum(preflight_charge[devices[i].filesystem], charge);
        replacement_bytes[devices[i].filesystem] = sum(replacement_bytes[devices[i].filesystem], metadata_charge - plan_charges[i]);
    }
    for (auto const& [filesystem, charge] : preflight_charge)
    {
        require(!args.capacity || charge <= args.capacity, "saved farming outputs and job exceed --capacity-bytes");
        require(replacement_bytes[filesystem] <= space_available.at(filesystem), "destination cannot hold the next farming-job revision");
    }
    if (run->state.estimate.feasible)
    {
        for (size_t lane = 0; lane < run->lanes.size(); ++lane)
            if (run->available[lane]) run->lanes[lane].reserved_members = run->maximum(args, lane);
        run->certify(run->metadata);
    }
    run->check();
    if (!args.plan_only && run->feasible())
    {
        job.check();
        auto const digest = stream_state::digest(group_grow::farming_state_text(run->state));
        std::vector<std::optional<stream_state::Stamp>> pending_stamps(directories.size());
        // Prepare all copies before publishing the first. The first published
        // revision commits the addition; an interruption leaves complete .next
        // copies for the remaining members. Partial recovery never changes scope.
        for (size_t i = 0; i < directories.size(); ++i) if (run->available[i])
        {
            auto const path = directories[i] / "grow-farm.plan";
            auto const pending = directories[i] / "grow-farm.plan.next";
            auto copy = std::find_if(job.copies.begin(), job.copies.end(), [&](auto const& value) { return value.path == path; });
            auto next = std::find_if(job.copies.begin(), job.copies.end(), [&](auto const& value) { return value.path == pending; });
            if (next != job.copies.end()) pending_stamps[i] = next->stamp;
            if (copy != job.copies.end() && copy->stamp && copy->digest == digest) continue;
            if (next != job.copies.end() && next->stamp)
                require(next->digest == digest, "pending farming revision differs from the admitted job");
            else
            {
                group_grow::stage_farming_state(path, run->state);
                pending_stamps[i] = stream_state::stamp(pending);
            }
        }
        group_grow::grow_failpoint("farm-plan-all-staged");
        // Staging may take time on a large farm. Recheck every published
        // parent before committing, then guard each replacement and its source.
        for (auto const& copy : job.copies) if (!copy.pending) copy.check();
        for (size_t i = 0; i < directories.size(); ++i) if (pending_stamps[i])
            require(stream_state::stamp(directories[i] / "grow-farm.plan.next") == *pending_stamps[i],
                "pending farming revision changed before publication");
        for (size_t i = 0; i < directories.size(); ++i) if (run->available[i])
        {
            auto const path = directories[i] / "grow-farm.plan";
            auto const pending = directories[i] / "grow-farm.plan.next";
            auto copy = std::find_if(job.copies.begin(), job.copies.end(), [&](auto const& value) { return value.path == path; });
            if (copy != job.copies.end()) copy->check();
            if (pending_stamps[i]) require(stream_state::stamp(pending) == *pending_stamps[i],
                "pending farming revision changed before publication");
            if (copy == job.copies.end() || !copy->stamp || copy->digest != digest)
                group_grow::publish_farming_state(pending, path, copy != job.copies.end() && copy->stamp.has_value());
            else if (fs::exists(pending))
            {
                auto next = group_grow::load_farming_state(pending, args.max_ram);
                require(stream_state::digest(group_grow::farming_state_text(next)) == digest, "conflicting pending farming copy");
                require(!file_io::unlink_file(pending), "cannot retire pending farming copy");
                stream_state::sync_path(directories[i], true);
            }
            run->manifests.emplace_back(path, stream_state::stamp(path));
            if (run->manifest.empty()) run->manifest = path;
            file_io::Stat plan{};
            require(!file_io::inspect_link(path.c_str(), &plan) && file_io::is_regular(plan.st_mode) &&
                plan.st_nlink == 1 && plan.st_blocks >= 0, "invalid published farming-job allocation");
            plan_charges[i] = std::max(uint64_t(plan.st_size), product(uint64_t(plan.st_blocks), 512));
            require(plan_charges[i] <= run->state.manifest_reserved_bytes, "farming job exceeds its metadata reservation");
            run->metadata = sum(run->metadata, product(uint64_t(plan.st_size), 4));
        }
    }
    else
    {
        for (auto const& copy : job.copies) if (copy.stamp)
        {
            run->manifests.emplace_back(copy.path, *copy.stamp);
            if (run->manifest.empty()) run->manifest = copy.path;
        }
    }
    std::map<std::string, uint64_t> charged;
    for (size_t i = 0; i < directories.size(); ++i) if (run->available[i])
    {
        auto& lane = run->lanes[i];
        uint64_t charge = plan_charges[i];
        for (auto const& output : lane.outputs) charge = sum(charge, output.charge());
        require(charge <= run->state.directories[i].capacity_bytes, "farming job and outputs exceed destination capacity");
        charged[devices[i].filesystem] = sum(charged[devices[i].filesystem], charge);
        lane.completed_charge = sum(lane.completed_charge, plan_charges[i]);
        run->manifest_charge = sum(run->manifest_charge, plan_charges[i]);
    }
    if (args.capacity)
        for (auto const& [filesystem, charge] : charged)
            require(charge <= args.capacity, "saved farming outputs and job exceed --capacity-bytes");
    require(run->metadata < args.max_ram, "farming job and output metadata exceed --max-ram");
    return run;
}
std::string json_string(std::string_view value)
{
    std::ostringstream out; out << '"';
    constexpr char digits[] = "0123456789abcdef";
    for (unsigned char c : value)
    {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u00" << digits[c >> 4] << digits[c & 15];
        else out << c;
    }
    out << '"'; return out.str();
}
std::string json_estimate(long double value)
{
    if (!std::isfinite(value)) return "null";
    std::ostringstream out; out << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
    return out.str();
}
void print_farming_plan(FarmingRun const& run, Arguments const& args)
{
    auto const& c = run.state.config; auto const& e = run.state.estimate;
    if (!args.command.empty() && !args.json)
    {
        if (args.quiet && !args.plan_only) return;
        auto& out = args.plan_only ? std::cout : std::cerr;
        out << "Farming layout: " << (run.feasible() ? "feasible" : "infeasible") << '\n'
            << "Harvester: " << c.harvester_cores << " cores, " << c.lookup_workers
            << " lookup workers, " << c.memory_bytes << " bytes working memory, "
            << c.index_cache_bytes << " bytes index cache\n"
            << "Objective: " << (c.objective == group_grow::FarmingObjective::latency ? "latency" : "energy")
            << "; reader: " << (c.reader == group_grow::FarmingReader::cached ? "cached" : "stateless")
            << "; response bound: " << json_estimate(e.response_bound_seconds)
            << " s; deadline: " << json_estimate(c.deadline_seconds) << " s\n"
            << "Modeled active work: " << json_estimate(e.dynamic_work_seconds_per_sp)
            << " s per signage point; filter bits: " << group_grow::farming_filter_bits(c)
            << " now, " << group_grow::farming_filter_bits(c, true) << " future\n";
        for (auto const& d : run.state.directories)
        {
            size_t const index = size_t(&d - run.state.directories.data());
            out << d.directory << ": " << d.layout.files << " files, up to " << d.layout.group_members
                << " members/file, " << d.layout.total_members << " members, " << d.layout.families
                << " identity families, " << d.layout.meta_slots << " meta slots; storage resource " << d.resource
                << "; capacity " << d.capacity_bytes << " bytes; "
                << (!run.available[index] ? "unavailable" : index < run.previous_destinations ? "saved assignment" : "new destination") << '\n';
        }
        if (!run.feasible()) out << "Reason: " << run.infeasibility_reason() << '\n';
        out << "Estimates use the declared harvester profile and modeled storage costs.\n";
        return;
    }
    std::cout << "{\"farming_plan\":true,\"model\":" << json_string(group_grow::farming_model_identity)
        << ",\"reader\":" << json_string(c.reader == group_grow::FarmingReader::cached ? "cached" : "stateless")
        << ",\"objective\":" << json_string(c.objective == group_grow::FarmingObjective::latency ? "latency" : "energy")
        << ",\"energy_model\":\"expected_dynamic_work\""
        << ",\"cpu_work_weight\":" << json_estimate(c.cpu_work_weight)
        << ",\"io_work_weight\":" << json_estimate(c.io_work_weight)
        << ",\"max_eligible_groups\":" << c.max_eligible_families_per_resource
        << ",\"certified_max_eligible_groups\":" << e.max_eligible_families_per_resource
        << ",\"harvester_cores\":" << c.harvester_cores << ",\"lookup_workers\":" << c.lookup_workers
        << ",\"memory_bytes\":" << c.memory_bytes << ",\"index_cache_bytes\":" << c.index_cache_bytes
        << ",\"filter_bits\":" << group_grow::farming_filter_bits(c)
        << ",\"future_filter_bits\":" << group_grow::farming_filter_bits(c, true)
        << ",\"deadline_seconds\":" << json_estimate(c.deadline_seconds)
        << ",\"challenge_seconds\":" << json_estimate(c.challenge_seconds)
        << ",\"chain_seconds_per_member\":" << json_estimate(c.chain_seconds_per_member)
        << ",\"index_seconds_per_entry\":" << json_estimate(c.index_seconds_per_entry)
        << ",\"compressed_index_bytes_per_entry\":" << json_estimate(c.compressed_index_bytes_per_entry)
        << ",\"lookup_memory_per_member\":" << c.lookup_memory_per_member
        << ",\"expected_cpu_utilization\":" << json_estimate(e.cpu_utilization)
        << ",\"expected_worker_utilization\":" << json_estimate(e.worker_utilization)
        << ",\"feasible\":" << (run.feasible() ? "true" : "false")
        << ",\"infeasibility_reason\":" << json_string(run.infeasibility_reason())
        << ",\"response_bound_seconds\":" << json_estimate(e.response_bound_seconds)
        << ",\"dynamic_work_seconds_per_sp\":" << json_estimate(e.dynamic_work_seconds_per_sp)
        << ",\"cpu_work_seconds_per_sp\":" << json_estimate(e.cpu_work_seconds_per_sp)
        << ",\"io_work_seconds_per_sp\":" << json_estimate(e.io_work_seconds_per_sp)
        << ",\"reader_memory_bytes\":" << e.reader_memory_bytes << ",\"index_bytes\":" << e.index_bytes
        << ",\"planning_memory_bytes\":" << e.planning_memory_bytes
        << ",\"candidates_evaluated\":" << e.candidates_evaluated
        << ",\"manifest_reserved_bytes\":" << run.state.manifest_reserved_bytes;
    if (run.admission)
    {
        auto const& a = *run.admission;
        std::cout << ",\"extent_assessment\":{\"feasible\":" << (a.feasible ? "true" : "false")
            << ",\"infeasibility_reason\":" << json_string(a.reason)
            << ",\"response_bound_seconds\":" << json_estimate(a.response_bound_seconds)
            << ",\"dynamic_work_seconds_per_sp\":" << json_estimate(a.dynamic_work_seconds_per_sp)
            << ",\"cpu_work_seconds_per_sp\":" << json_estimate(a.cpu_work_seconds_per_sp)
            << ",\"io_work_seconds_per_sp\":" << json_estimate(a.io_work_seconds_per_sp)
            << ",\"certified_max_eligible_groups\":" << a.max_eligible_families_per_resource << '}';
    }
    std::cout << ",\"destinations\":[";
    bool comma = false;
    for (auto const& d : run.state.directories)
    {
        if (comma) std::cout << ','; comma = true;
        size_t const index = size_t(&d - run.state.directories.data());
        auto const& p = d.layout; auto const& storage = d.storage;
        std::cout << "{\"directory\":" << json_string(d.directory) << ",\"capacity_bytes\":" << d.capacity_bytes
            << ",\"available\":" << (run.available[index] ? "true" : "false")
            << ",\"resource\":" << d.resource << ",\"resource_rank\":" << p.resource_rank << ",\"resource_lanes\":" << p.resource_lanes
            << ",\"members\":" << p.total_members << ",\"files\":" << p.files << ",\"group_bytes\":" << p.group_bytes
            << ",\"group_members\":" << p.group_members << ",\"meta_slots\":" << p.meta_slots << ",\"families\":" << p.families
            << ",\"known_device\":" << (storage.known_device ? "true" : "false")
            << ",\"nonrotating\":" << (storage.nonrotating ? "true" : "false")
            << ",\"read_bytes_per_second\":" << json_estimate(storage.read_bytes_per_second)
            << ",\"read_latency_seconds\":" << json_estimate(storage.read_latency_seconds)
            << ",\"read_concurrency\":" << storage.read_concurrency << '}';
    }
    std::cout << "]}\n";
}

int print_destination_candidates(Arguments const& args)
{
    auto const paths = group_grow::farming_destination_candidates();
    std::vector<group_grow::StorageDevice> devices;
    for (auto const& path : paths) devices.push_back(group_grow::storage_device(path));
    auto const resources = group_grow::writer_groups(devices);
    if (args.json) std::cout << "{\"destination_candidates\":true,\"destinations\":[";
    else std::cout << "Select destination directories on these mounted filesystems. No files have been written.\n";
    bool comma = false;
    for (size_t i = 0; i < paths.size(); ++i)
    {
        file_io::Space space{};
        if (file_io::space_path(paths[i].c_str(), &space)) continue;
        auto const bytes = product(space.f_bavail, space.f_frsize);
        if (bytes <= args.headroom) continue;
        auto const media = devices[i].nonrotating ? "SSD" : devices[i].rotating ? "HDD" : "unknown";
        if (args.json)
        {
            if (comma) std::cout << ','; comma = true;
            std::cout << "{\"directory\":" << json_string(file_io::path_text(paths[i])) << ",\"available_bytes\":" << bytes
                << ",\"media\":" << json_string(media) << ",\"resource\":" << resources[i] << '}';
        }
        else std::cout << file_io::path_text(paths[i]) << ": " << bytes << " bytes available; " << media
            << "; storage resource " << resources[i] << '\n';
    }
    if (args.json) std::cout << "]}\n";
    else
    {
        if (paths.empty()) std::cout << "No eligible data-volume candidates found. Supply the intended directories explicitly.\n";
        std::cout << "Preview the selected directories with: xchplot2 plan [OPTIONS] -- DIR...\n"
            << "Suggestions exclude system mounts, RAM filesystems and unsupported volumes.\n";
    }
    return 0;
}

struct NamedGroup
{
    Identity identity;
    uint16_t members;
};

int multiple_destinations(Arguments const& args, Keys const* keys, FarmingRun* farm = nullptr,
                          std::span<NamedGroup const> named = {})
{
    if (farm && std::all_of(farm->lanes.begin(), farm->lanes.end(), [&](auto const& lane)
        { return farm->maximum(args, size_t(&lane - farm->lanes.data())) == 0; }))
    {
        Metrics total; static_cast<group_grow::Stats&>(total) = farm->audited;
        total.destination_count = farm->lanes.size();
        uint64_t groups = 0, accounted = farm->manifest_charge, last_members = 0;
        std::map<std::string, uint64_t> capacities;
        for (size_t i = 0; i < farm->lanes.size(); ++i)
        {
            auto const& lane = farm->lanes[i];
            total.members = sum(total.members, lane.completed_members);
            total.final_bytes = sum(total.final_bytes, lane.completed_bytes);
            auto const& directory = farm->state.directories[i];
            auto const filesystem = group_grow::storage_device(file_io::native_path(directory.directory)).filesystem;
            capacities[filesystem] = sum(capacities[filesystem], directory.capacity_bytes);
            total.writer_count = std::max(total.writer_count, farm->state.directories[i].resource + 1);
            groups = sum(groups, lane.outputs.size());
            for (auto const& output : lane.outputs) accounted = sum(accounted, output.charge());
            if (!lane.outputs.empty()) last_members = lane.outputs.back().members;
            if (lane.no_capacity) total.stop_reason = "capacity";
        }
        for (auto const& [filesystem, capacity] : capacities)
            total.capacity_bytes = sum(total.capacity_bytes, args.capacity ? std::min(args.capacity, capacity) : capacity);
        total.peak_file_bytes = accounted; total.peak_modeled_bytes = std::max(farm->metadata, farm->planning_peak);
        if (total.stop_reason.empty()) total.stop_reason = farm->stop_reason(args);
        Backend backend;
        farm->assessment_metrics(total);
        for (auto const& lane : farm->lanes)
            for (auto const& output : lane.outputs)
                report_output(args, group_path(file_io::native_path(farm->state.directories[&lane - farm->lanes.data()].directory),
                    output.ordinal), uint16_t(output.members), output.bytes, true);
        print_metrics(args, total, backend, groups, last_members, accounted);
        return total.members ? 0 : 2;
    }
    std::vector<fs::path> paths;
    if (farm) for (size_t i = 0; i < farm->lanes.size(); ++i) paths.push_back(farm->path(i));
    else { paths.push_back(args.target); paths.insert(paths.end(), args.destinations.begin(), args.destinations.end()); }
    require((keys && named.empty()) || (!farm && named.size() == paths.size()), "missing destination identities");
    uint64_t metadata = sum(sum(farm ? farm->metadata : product(paths.size(), mib), path_metadata(args.temporary)),
        product(paths.size(), 64 * sizeof(group_grow::ConstructionRun)));
    require(metadata < args.max_ram, "destination metadata exceeds --max-ram");
    std::vector<group_grow::StorageDevice> devices; devices.reserve(paths.size());
    std::vector<std::optional<Output>> guards; guards.reserve(paths.size());
    std::vector<file_io::Stat> directories; directories.reserve(paths.size());
    struct Space { uint64_t available = UINT64_MAX, allocated = 0; size_t outputs = 0; };
    std::map<std::string, Space> spaces;
    std::set<fs::path> names; bool existing = false;
    std::vector<std::optional<group_grow::ExternalEngine::Inspection>> inspections(paths.size());
    std::vector<std::unique_ptr<GroupSuccessor>> saved_successors(paths.size());
    for (auto const& path : paths)
    {
        require(fs::is_directory(file_io::native_path(path.parent_path())), "destination directory must exist: " + file_io::path_text(path.parent_path()));
        require(names.insert(fs::canonical(file_io::native_path(path.parent_path())) / path.filename()).second, "duplicate destination path");
        bool const exists = fs::exists(file_io::native_path(path)); existing = existing || exists;
        require(!exists || args.resume, "destination exists; use --resume: " + file_io::path_text(path));
        file_io::Stat directory{};
        require(!file_io::inspect_path(path.parent_path().c_str(), &directory), "cannot inspect destination directory identity");
        directories.push_back(directory);
        guards.push_back(exists ? std::optional(snapshot(path, 0)) : std::nullopt);
        devices.push_back(group_grow::storage_device(path.parent_path()));
        auto& space = spaces[devices.back().filesystem]; ++space.outputs;
        file_io::Space allocation{};
        require(!file_io::space_path(path.parent_path().c_str(), &allocation), "cannot inspect destination capacity");
        space.available = std::min(space.available, product(allocation.f_bavail, allocation.f_frsize));
        if (exists)
        {
            space.allocated = sum(space.allocated, guards.back()->allocated);
        }
    }
    require(farm || !named.empty() || !args.resume || existing, "--resume found no destination output");
    auto const groups = group_grow::writer_groups(devices);
    size_t const writer_count = *std::max_element(groups.begin(), groups.end()) + 1;
    auto roots = args.temporary;
    for (size_t i = 0; i < paths.size(); ++i)
        add_destination_temporary(roots, args.destination_temp, paths[i].parent_path(), devices[i]);
    size_t const fresh_roots = roots.size();
    require(args.construction != "external" || fresh_roots ||
        [&]
        {
            for (size_t i = 0; i < guards.size(); ++i)
                if (!guards[i] && (!farm || farm->maximum(args, i))) return false;
            return true;
        }(), "fresh external construction requires --temp-dir or an eligible --destination-temp directory");
    for (size_t i = 0; i < paths.size(); ++i)
    {
        Output absent;
        auto& output = guards[i] ? *guards[i] : absent;
        inspections[i] = output_auxiliary(paths[i], output, args.max_ram - metadata);
        spaces[devices[i].filesystem].allocated = sum(spaces[devices[i].filesystem].allocated,
            output.auxiliary_allocated);
        if (inspections[i])
        {
            require(guards[i].has_value() && args.resume, "external construction requires its destination and --resume");
            metadata = sum(metadata, external_metadata(*inspections[i]));
            require(metadata < args.max_ram, "saved destination metadata exceeds --max-ram");
            if (args.temporary.empty())
                for (auto const& grant : inspections[i]->external.scratch)
                    if (std::find(roots.begin(), roots.end(), file_io::native_path(grant.directory)) == roots.end())
                        roots.emplace_back(file_io::native_path(grant.directory));
        }
    }
    if (farm) for (size_t i = 0; i < paths.size(); ++i)
    {
        auto const& lane = farm->lanes[i];
        if (lane.ordinal + 1 >= lane.outputs.size()) continue;
        auto next = std::make_unique<GroupSuccessor>(group_path(paths[i].parent_path(), lane.ordinal + 1),
            farm->identity(*keys, i, lane.ordinal + 1), farm->file_maximum(i, lane.ordinal + 1), 0);
        next->owner = paths.size() + i; next->guard = lane.outputs[lane.ordinal + 1];
        next->recovery = group_grow::ExternalEngine::inspect(file_io::path_text(next->path), args.max_ram - metadata);
        next->metadata_bytes = sum(external_metadata(*next->recovery), sum(sizeof(GroupSuccessor),
            sum(next->path.native().capacity() + 1, next->requested.memo.capacity())));
        metadata = sum(metadata, next->metadata_bytes);
        require(metadata < args.max_ram, "saved farming successor metadata exceeds --max-ram");
        spaces[devices[i].filesystem].allocated = sum(spaces[devices[i].filesystem].allocated,
            sum(next->guard.allocated, next->guard.auxiliary_allocated));
        if (args.temporary.empty()) for (auto const& grant : next->recovery->external.scratch)
            if (std::find(roots.begin(), roots.end(), file_io::native_path(grant.directory)) == roots.end())
                roots.emplace_back(file_io::native_path(grant.directory));
        saved_successors[i] = std::move(next);
    }
    metadata = sum(metadata, path_metadata(roots));
    require(metadata < args.max_ram, "destination metadata exceeds --max-ram");
    std::unique_ptr<group_grow::ScratchPool> scratch_pool;
    std::vector<uint64_t> minimum_peaks(paths.size(), 0);
    std::map<std::string, uint64_t> output_room;
    if (!roots.empty())
    {
        scratch_pool = std::make_unique<group_grow::ScratchPool>(roots, args.headroom, args.temporary_capacity);
        for (auto const& path : paths) preserve_retired(*scratch_pool, path, args.max_ram - metadata, farm != nullptr);
        for (size_t i = 0; i < paths.size(); ++i)
        {
            if (farm && !farm->maximum(args, i) && !guards[i]) continue;
            auto const& saved = inspections[i];
            uint64_t const retired = guards[i] ? guards[i]->retired : 0;
            uint64_t const allocated = sum(guards[i] ? guards[i]->retired_allocated : 0,
                saved ? sum(saved->target_allocated_bytes, saved->metadata_allocated_bytes) : guards[i] ? guards[i]->allocated : 0);
            minimum_peaks[i] = saved ? std::max(sum(saved->peak_bytes, retired), allocated) : guards[i] ? guards[i]->charge() : 0;
            require(scratch_pool->try_reserve_output(i, paths[i].parent_path(), minimum_peaks[i], allocated, bool(guards[i])),
                "saved output reservations exceed available capacity");
            if (saved) scratch_pool->restore(i, saved->external, saved->allocated_bytes);
        }
        for (auto const& next : saved_successors) if (next)
        {
            auto const& saved = *next->recovery;
            uint64_t const allocated = sum(next->guard.retired_allocated,
                sum(saved.target_allocated_bytes, saved.metadata_allocated_bytes));
            uint64_t const peak = std::max(allocated, sum(next->guard.retired, saved.peak_bytes));
            require(scratch_pool->try_reserve_output(next->owner, next->path.parent_path(), peak, allocated, true),
                "saved farming successor reservation exceeds available capacity");
            scratch_pool->restore(next->owner, saved.external, saved.allocated_bytes);
        }
        for (size_t i = 0; i < paths.size(); ++i)
            output_room[devices[i].filesystem] = scratch_pool->available_for_output(i, paths[i].parent_path()) - minimum_peaks[i];
    }
    enum class Work { append, recover, prepare_finish, finish, recover_finish, fallback_input };
    struct Task
    {
        Work work = Work::append;
        bool select_merged = false;
        Producer::FillStop fill_stop = Producer::FillStop::requested;
        std::vector<group_grow::Member> members;
        std::unique_ptr<group_grow::ExternalEngine::PreparedInput> prepared_input;
        std::unique_ptr<group_grow::ExternalEngine::ContributionInput> contribution_input;
        uint64_t contribution_bytes = 0;
        uint16_t prepared_members = 0;
        uint64_t prepared_read_accounted = 0;
        MemberLease lease;
        std::optional<group_grow::ScratchPool::IoLease> io;
        std::unique_ptr<group_grow::MemberBatchHash> hash;
        group_grow::AppendResult appended;
        group_grow::Result finished;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        struct Generated
        {
            size_t members;
            uint64_t peak, generation_peak, elapsed;
            Producer::FillStop stop;
        };
        std::atomic<bool> cancel_generation{false};
        std::future<Generated> generating;
        uint64_t receiving_bytes = 0;
        // A producer can still own provisional scratch or its member vector.
        // Join before any of those fields, or the engine, are destroyed.
        ~Task()
        {
            cancel_generation.store(true);
            if (generating.valid()) generating.wait();
        }
#endif
    };
    struct Destination
    {
        fs::path path;
        Identity requested;
        group_grow::Identity saved;
        std::unique_ptr<Construction> engine;
        std::vector<group_grow::ConstructionRun> execution_runs;
        std::optional<group_grow::ExternalConfig> scratch;
        Metrics metrics;
        std::unique_ptr<Task> pending;
        std::future<void> ready;
        size_t writer = 0, owner = 0, scratch_owner = 0;
        std::unique_ptr<GroupSuccessor> successor;
        size_t successor_target = 0, prefunded_rows = 0;
        bool successor_staged = false, staging_successor = false;
        uint32_t successor_wait_reason = 0, prefetch_wait_reason = 0;
        uint16_t batch_cap = 0, recover_count = 0, maximum = 0;
        bool stream_input_candidate = false;
        uint16_t stream_input_batch = 0;
        uint64_t stream_input_bytes = 0;
        size_t stream_input_active = 0;
        uint64_t engine_bytes = 0, lease_bytes = 0, observed_member_storage = 0, capacity_limit = 0, farm_capacity = 0;
        uint64_t packing_released_bytes = 0;
        size_t generation_deferrals = 0;
        unsigned workers = 0;
        std::string mode;
        std::optional<Output> guard;
        file_io::Stat directory{};
        bool needs_resume = false, needs_finish = false, complete = false, farm_counted = false, output_reserved = false;
        Destination(fs::path name, Identity value) : path(std::move(name)), requested(std::move(value))
        { execution_runs.reserve(64); }
    };
    // Receiver futures and ordered writers may both wake the controller.
    // These objects outlive every task, including exception unwinding.
    std::mutex wake_mutex; std::condition_variable wake;
    std::unique_ptr<Backend> backend = std::make_unique<Backend>();
    std::vector<std::unique_ptr<Destination>> destinations; destinations.reserve(paths.size());
    group_grow::ConstructionHistories histories(paths.size());
    group_grow::StreamingInputCohort stream_cohort(paths.size());
    metadata = sum(metadata, stream_cohort.memory_bytes());
    auto const execution_begin = std::chrono::steady_clock::now();
    auto execution_now = [&] { return sum(1, group_grow::elapsed_ns(execution_begin)); };
    size_t history_contexts = 0;
    uint64_t pool_memory = scratch_pool ? scratch_pool->memory_bytes() : 0;
    uint64_t fixed = sum(sum(metadata, histories.memory_bytes()), pool_memory);
    auto update_pool_memory = [&]
    {
        if (!scratch_pool) return;
        uint64_t const now = scratch_pool->memory_bytes();
        uint64_t copies = 0;
        for (auto const& d : destinations)
        {
            if (d->scratch) copies = sum(copies, external_metadata(*d->scratch));
            if (d->successor && d->successor->scratch) copies = sum(copies, external_metadata(*d->successor->scratch));
        }
        fixed = sum(fixed - pool_memory, sum(now, copies)); pool_memory = sum(now, copies);
    };
    unsigned const workers = args.workers ? unsigned(args.workers) : std::max(1u, std::thread::hardware_concurrency());
    uint64_t const scheduler_memory = group_grow::CodecScheduler::memory_bytes(workers);
    metadata = sum(metadata, scheduler_memory); fixed = sum(fixed, scheduler_memory);
    require(fixed < args.max_ram, "codec scheduler metadata exceeds --max-ram");
    auto cpu_scheduler = std::make_shared<group_grow::CodecScheduler>(workers);
    backend->cpu_scheduler = cpu_scheduler;
    backend->configure_network(args);
    auto batch_metadata = [&](uint16_t count)
    { return sum(product(count, 2 * sizeof(group_grow::Member)), batch_hash_bytes(args, count)); };
    auto check_guard = [](Destination const& d)
    {
        file_io::Stat directory{};
        require(!file_io::inspect_path(d.path.parent_path().c_str(), &directory) && file_io::is_directory(directory.st_mode) &&
            directory.st_dev == d.directory.st_dev && directory.st_ino == d.directory.st_ino,
            "destination directory changed during the run");
        if (d.engine || d.staging_successor) return;
        if (d.guard)
        {
            auto const now = snapshot(d.path, 0); auto const& before = *d.guard;
            require(now.device == before.device && now.inode == before.inode && now.bytes == before.bytes &&
                now.allocated == before.allocated && now.modified_seconds == before.modified_seconds &&
                now.modified_nanoseconds == before.modified_nanoseconds && now.changed_seconds == before.changed_seconds &&
                now.changed_nanoseconds == before.changed_nanoseconds,
                "inactive destination path or contents changed during the run");
        }
        else
        {
            file_io::Stat file{};
            require(file_io::inspect_link(d.path.c_str(), &file) != 0 && errno == ENOENT,
                "inactive destination appeared during the run");
        }
    };
    auto check_farm = [&]
    {
        if (!farm) return;
        std::vector<uint8_t> active(destinations.size());
        for (size_t i = 0; i < destinations.size(); ++i) active[i] = bool(destinations[i]->engine) ||
            bool(destinations[i]->successor && destinations[i]->successor->engine);
        if (active.size() < farm->lanes.size()) active.resize(farm->lanes.size());
        farm->check(active);
    };
    std::vector<uint64_t> farm_capacities(paths.size(), 0);
    if (farm)
    {
        for (auto const& [filesystem, space] : spaces)
        {
            uint64_t completed = 0, minimum = 0;
            std::vector<std::pair<uint64_t, size_t>> shares;
            for (size_t i = 0; i < paths.size(); ++i) if (devices[i].filesystem == filesystem)
            {
                auto const& lane = farm->lanes[i]; completed = sum(completed, lane.completed_charge);
                if (!farm->maximum(args, i) && !guards[i]) continue;
                uint64_t const saved = sum(lane.ordinal < lane.outputs.size() ? lane.outputs[lane.ordinal].charge() : 0,
                    saved_successors[i] ? saved_successors[i]->guard.charge() : 0);
                uint64_t const quota = farm->state.directories[i].capacity_bytes - lane.completed_charge;
                require(saved <= quota, "saved farming output exceeds its remaining directory quota");
                farm_capacities[i] = saved; minimum = sum(minimum, saved);
                shares.emplace_back(quota - saved, i);
            }
            if (shares.empty()) continue;
            uint64_t const available = sum(space.available, space.allocated);
            require(available > args.headroom, "destination filesystem has no capacity above headroom");
            uint64_t remaining = available - args.headroom;
            if (args.capacity)
            {
                require(completed <= args.capacity, "completed farming outputs exceed --capacity-bytes");
                remaining = std::min(remaining, args.capacity - completed);
            }
            require(minimum <= remaining, "saved farming transactions exceed available aggregate capacity");
            remaining -= minimum;
            std::sort(shares.begin(), shares.end());
            for (size_t i = 0; i < shares.size(); ++i)
            {
                uint64_t const capacity = std::min(shares[i].first, remaining / (shares.size() - i));
                farm_capacities[shares[i].second] = sum(farm_capacities[shares[i].second], capacity); remaining -= capacity;
            }
        }
    }
    uint64_t audit_peak = fixed;
    // Discover every saved protocol and identity before any confirmation or
    // generation. Only this transient engine is resident during the audit.
    for (size_t i = 0; i < paths.size(); ++i)
    {
        if (farm && (!farm->maximum(args, i) || !farm_capacities[i]) && !guards[i])
        {
            bool const reserved = scratch_pool && farm->maximum(args, i);
            if (!farm_capacities[i]) farm->lanes[i].no_capacity = true;
            auto destination = std::make_unique<Destination>(paths[i], farm->identity(*keys, i, farm->lanes[i].ordinal));
            destination->writer = groups[i]; destination->owner = destination->scratch_owner = i; destination->directory = directories[i];
            destination->saved = initial_identity(destination->requested); destination->mode = args.construction;
            destination->complete = true; destination->output_reserved = reserved; destinations.push_back(std::move(destination)); continue;
        }
        auto const& space = spaces.at(devices[i].filesystem);
        uint64_t const available = sum(space.available, space.allocated);
        require(available > args.headroom, "destination filesystem has no capacity above headroom");
        uint64_t const aggregate = args.capacity ? std::min(args.capacity, available - args.headroom) : available - args.headroom;
        uint64_t const share = farm ? farm_capacities[i] : aggregate / space.outputs;
        uint64_t const file_limit = group_grow::maximum_file_bytes(file_io::path_text(paths[i].parent_path()));
        require(farm || !args.capacity || share <= file_limit,
                "destination share exceeds the filesystem or process single-file limit; use a smaller capacity");
        uint64_t capacity = std::min(share, file_limit);
        auto destination = std::make_unique<Destination>(paths[i], farm ? farm->identity(*keys, i, farm->lanes[i].ordinal) : named.empty() ? keys->derive(i) : named[i].identity); auto& d = *destination;
        d.writer = groups[i]; d.owner = d.scratch_owner = i; d.maximum = farm ? farm->maximum(args, i) : (named.empty() ? uint16_t(args.max_members) : named[i].members);
        if (farm) d.farm_capacity = sum(farm->lanes[i].completed_charge, farm_capacities[i]);
        d.directory = directories[i]; d.guard = guards[i]; d.output_reserved = bool(scratch_pool); check_guard(d);
        d.saved = initial_identity(d.requested); d.mode = args.construction; d.complete = !d.maximum;
        uint64_t const retired = d.guard ? d.guard->retired : 0;
        require(retired <= capacity, "retired metadata exceeds the destination capacity share");
        d.capacity_limit = scratch_pool ? sum(retired, temporary_output_capacity(args, d.requested,
            uint16_t(d.maximum), capacity - retired)) : capacity;
        if (scratch_pool)
        {
            capacity = std::min(capacity, sum(minimum_peaks[i], output_room.at(devices[i].filesystem) / space.outputs));
            capacity = std::max(minimum_peaks[i], sum(retired, temporary_output_capacity(args, d.requested,
                uint16_t(d.maximum), capacity - retired)));
            require(capacity <= share && scratch_pool->try_reserve_output(i, d.path.parent_path(), capacity),
                "saved output and temporary reservations exceed the destination capacity share");
            if (inspections[i])
            {
                d.scratch = inspections[i]->external;
                uint64_t const bytes = external_metadata(*d.scratch);
                metadata = sum(metadata, bytes); fixed = sum(fixed, bytes);
            }
            update_pool_memory();
        }
        d.metrics.capacity_bytes = capacity;
        if (d.guard)
        {
            auto cfg = configuration(args, capacity, fixed, cpu_scheduler, backend->remote_encoder());
            Construction audit(file_io::path_text(d.path), cfg, true, d.saved, "auto", d.scratch, retired);
            d.saved = audit.identity(); d.mode = audit.mode();
            if (audit.external) audit.external->run_shapes(d.execution_runs);
            d.metrics.members = audit.member_count();
            d.needs_resume = !audit.is_staging() && d.saved.pending;
            d.needs_finish = audit.is_staging() && !audit.can_append();
            d.metrics.recovered = audit.staged_recovery || d.needs_resume;
            d.complete = !d.saved.pending && !audit.is_staging() && d.metrics.members == d.maximum;
            audit_peak = std::max(audit_peak, sum(fixed, audit.memory_bytes(
                std::max(uint16_t(1), uint16_t(d.saved.target_members - d.saved.old_members)), args.member_storage == "compact")));
            require(audit_peak <= args.max_ram, "saved destination workspace exceeds --max-ram");
            static_cast<group_grow::Stats&>(d.metrics) = audit.stats();
        }
        same_identity(d.requested, identity_from(d.saved));
        if (!farm) requested_identity(args, d.requested);
        require(d.saved.bytes <= capacity - retired && (!d.saved.pending || d.saved.pending_peak_bytes <= capacity - retired),
                "saved destination exceeds its capacity share");
        require(d.metrics.members <= d.maximum && (!d.saved.pending || d.saved.target_members <= d.maximum),
                "saved destination membership exceeds --max-members");
        d.recover_count = d.needs_resume ? uint16_t(d.saved.target_members - d.saved.old_members) : 0;
        d.metrics.final_bytes = d.saved.bytes;
        d.metrics.peak_file_bytes = sum(std::max(d.saved.bytes, d.saved.pending_peak_bytes), retired);
        destinations.push_back(std::move(destination));
    }
    for (size_t i = 0; i < destinations.size(); ++i)
        destinations[i]->successor = std::move(saved_successors[i]);
    check_farm();
    for (auto const& destination : destinations) check_guard(*destination);
    if (scratch_pool)
    {
        scratch_pool->restrict_grants(std::span(roots).first(fresh_roots));
        scratch_pool->seal_recovery();
    }
    if (scratch_pool) for (auto const& d : destinations) if (d->complete && d->output_reserved)
        scratch_pool->release_after_cleanup(d->scratch_owner, d->guard ? sum(allocated_bytes(d->path), d->guard->retired_allocated) : 0);
    update_pool_memory();
    inspections.clear(); inspections.shrink_to_fit();

    // Writers are destroyed before destinations or the GPU backend, including
    // exception unwinding. Every queued task therefore retains valid members,
    // hash state and engine ownership until its consumer has stopped.
    std::vector<std::unique_ptr<group_grow::OrderedWriter>> writers(writer_count);
    std::vector<std::shared_ptr<std::recursive_mutex>> target_gates(writer_count);
    for (auto& gate : target_gates) gate = std::make_shared<std::recursive_mutex>();
    uint64_t const gate_memory = product(writer_count,
        sizeof(std::recursive_mutex) + 2 * sizeof(void*) + sizeof(std::shared_ptr<std::recursive_mutex>));
    metadata = sum(metadata, gate_memory); fixed = sum(fixed, gate_memory);
    GroupPrefetch prefetched;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
    struct PoolPrefetch
    {
        GroupPrefetch batch;
        std::atomic<bool> cancel{false};
        std::future<Task::Generated> ready;
        uint64_t receiving_bytes = 0;
        ~PoolPrefetch() { cancel.store(true); if (ready.valid()) ready.wait(); }
    } pool_prefetch;
#endif
    Metrics total; total.destination_count = destinations.size(); total.writer_count = writer_count;
    total.engine_reserved_bytes = fixed; total.peak_modeled_bytes = std::max(audit_peak, farm ? farm->planning_peak : 0);
    if (farm)
    {
        static_cast<group_grow::Stats&>(total) = farm->audited;
        for (size_t i = 0; i < farm->lanes.size(); ++i)
        {
            auto const& lane = farm->lanes[i];
            total.members = sum(total.members, lane.completed_members);
            total.final_bytes = sum(total.final_bytes, lane.completed_bytes);
            total.peak_file_bytes = sum(total.peak_file_bytes, lane.completed_charge);
            total.capacity_bytes = sum(total.capacity_bytes, sum(lane.completed_charge, farm_capacities[i]));
        }
    }
    auto accumulate = [&](Metrics const& part, std::string const& fallback)
    {
        total.controller_waits.add(part.controller_waits);
        total.packing_engine_bytes = std::max(total.packing_engine_bytes, part.packing_engine_bytes);
        total.packing_released_bytes = std::max(total.packing_released_bytes, part.packing_released_bytes);
        total.members = sum(total.members, part.members); total.final_bytes = sum(total.final_bytes, part.final_bytes);
        if (!farm) total.capacity_bytes = sum(total.capacity_bytes, part.capacity_bytes);
        if (!farm) total.peak_file_bytes = sum(total.peak_file_bytes, part.peak_file_bytes);
        total.workspace_bytes = sum(total.workspace_bytes, part.workspace_bytes);
        total.generated_members = sum(total.generated_members, part.generated_members); total.batches = sum(total.batches, part.batches);
        total.fresh_single_pass_batches = sum(total.fresh_single_pass_batches, part.fresh_single_pass_batches);
        total.exact_plan_batches = sum(total.exact_plan_batches, part.exact_plan_batches);
        total.staged_batches = sum(total.staged_batches, part.staged_batches);
        total.staging_fallbacks = sum(total.staging_fallbacks, part.staging_fallbacks);
        total.stream_input_batches = sum(total.stream_input_batches, part.stream_input_batches);
        total.stream_input_written_bytes = sum(total.stream_input_written_bytes, part.stream_input_written_bytes);
        total.stream_input_read_bytes = sum(total.stream_input_read_bytes, part.stream_input_read_bytes);
        total.stream_input_preflight_rejections = sum(total.stream_input_preflight_rejections, part.stream_input_preflight_rejections);
        total.scratch_peak_bytes = sum(total.scratch_peak_bytes, part.scratch_peak_bytes);
        total.scratch_released = total.scratch_released && part.scratch_released;
        total.automatic_merges = sum(total.automatic_merges, part.automatic_merges);
        total.automatic_fallbacks = sum(total.automatic_fallbacks, part.automatic_fallbacks);
        total.tuning_changes = sum(total.tuning_changes, part.tuning_changes);
        total.selected_window_bytes = std::max(total.selected_window_bytes, part.selected_window_bytes);
        total.selected_checkpoint_bytes = std::max(total.selected_checkpoint_bytes, part.selected_checkpoint_bytes);
        total.max_batch_members = std::max(total.max_batch_members, part.max_batch_members);
        auto const mode = part.construction_selected.empty() ? fallback : part.construction_selected;
        if (total.construction_selected.empty()) total.construction_selected = mode;
        else if (total.construction_selected != mode) total.construction_selected = "mixed";
        static_cast<group_grow::Stats&>(total) = Construction::add_stats(total, part);
        total.recovered = total.recovered || part.recovered;
        if (part.stop_reason == "capacity") total.stop_reason = "capacity";
        if (farm && !part.scratch_released) total.stop_reason = "preserved-scratch";
    };
    auto retained = [&]
    {
        uint64_t bytes = prefetched.resident_bytes();
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (pool_prefetch.ready.valid()) bytes = sum(bytes, pool_prefetch.receiving_bytes);
#endif
        for (auto const& d : destinations) if (d->pending)
        {
            uint64_t members;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            if (d->pending->generating.valid()) members = d->pending->receiving_bytes;
            else
#endif
                members = resident(d->pending->members);
            bytes = sum(bytes, sum(members, d->pending->contribution_bytes));
        }
        total.queued_peak_bytes = std::max(total.queued_peak_bytes, bytes); return bytes;
    };
    auto farm_peak = [&]
    {
        if (!farm) return;
        uint64_t peak = 0;
        for (auto const& d : destinations)
        {
            peak = sum(peak, farm->lanes[d->owner].completed_charge);
            if (!d->farm_counted) peak = sum(peak, d->metrics.peak_file_bytes);
            if (d->successor) peak = sum(peak, d->successor->recovery ?
                d->successor->guard.charge() : d->successor->metrics.peak_file_bytes);
        }
        total.peak_file_bytes = std::max(total.peak_file_bytes, peak);
    };
    size_t active = 0;
    uint32_t wait_reasons = 0;
    auto blocked = [&](ControllerWait reason) { wait_reasons |= wait_bit(reason); return false; };
    std::string farming_resource_error;
    auto activate = [&](Destination& d)
    {
        if (d.engine) return true;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (pool_prefetch.ready.valid() && pool_prefetch.batch.owner == d.owner)
            return blocked(ControllerWait::generation_order);
#endif
        if (!stream_cohort.may_admit(d.owner)) return false;
        if (farm)
        {
            auto& lane = farm->lanes[d.owner]; auto const previous = lane.reserved_members;
            lane.reserved_members = std::max(lane.reserved_members, d.maximum);
            auto const before = farm->metadata;
            try { farm->certify(sum(sum(fixed, retained()), backend->retained())); }
            catch (group_grow::FarmingResourceError const& error)
            {
                lane.reserved_members = previous;
                if (!error.memory_budget) throw;
                farming_resource_error = error.what(); return blocked(ControllerWait::engine_memory);
            }
            farming_resource_error.clear();
            metadata = sum(metadata - before, farm->metadata); fixed = sum(fixed - before, farm->metadata);
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, farm->planning_peak);
            if (!farm->feasible()) lane.reserved_members = previous;
            require(farm->feasible(), "cannot extend frozen farming layout: " + farm->infeasibility_reason());
        }
        bool fresh_scratch = false;
        uint64_t const prior_capacity = d.metrics.capacity_bytes;
        auto defer = [&]
        {
            if (fresh_scratch)
            {
                scratch_pool->release_scratch_after_cleanup(d.scratch_owner); d.scratch.reset();
                require(scratch_pool->try_reserve_output(d.scratch_owner, d.path.parent_path(), prior_capacity),
                    "cannot restore an unused destination reservation");
                d.metrics.capacity_bytes = prior_capacity; fresh_scratch = false; update_pool_memory();
            }
            return false;
        };
        bool const pooled_generation = !args.pool_broker.empty();
        // A remote receiver owns bounded RAM independently of its final
        // physical writer. The writer queue still serializes all publication.
        if (!pooled_generation)
            for (auto const& other : destinations)
                if (other->engine && other->writer == d.writer) return blocked(ControllerWait::writer_queue);
        uint64_t const held = retained();
        bool const owns_prefetch = !prefetched.members.empty() && prefetched.owner == d.owner;
        uint64_t const incoming_metadata = owns_prefetch ? prefetched.metadata_bytes : 0;
        uint64_t const other_held = held - (owns_prefetch ? prefetched.resident_bytes() : 0);
        bool use_stream_input = false;
        // Keep fixed/held truthful through construction. Only the ordinary
        // generation view replaces the owner's old batch with its new prefix.
        auto generation_fixed = [&] { return fixed - incoming_metadata; };
        auto memory = [&](unsigned count, uint16_t batch)
        {
            auto options = args; options.workers = count;
            auto cfg = configuration(options, d.metrics.capacity_bytes, fixed, cpu_scheduler, backend->remote_encoder());
            if (devices[d.owner].nonrotating && cfg.window_bytes >= 256 * 1024) cfg.merge_inputs = 4;
            bool const compact = args.member_storage == "compact";
            uint64_t bytes = group_grow::Engine::memory_bytes(d.requested.params.get_k(), cfg, batch, 0, compact ? batch : 0);
            std::string_view const mode = d.saved.pending ? d.mode : args.construction;
            if (mode == "merged") bytes = std::max(bytes, group_grow::MergeEngine::memory_bytes(d.requested.params.get_k(), cfg, batch));
            if (mode == "staged") bytes = std::max(bytes, group_grow::StagingEngine::memory_bytes(d.requested.params.get_k(), cfg, batch));
            if (d.scratch) bytes = std::max(bytes, group_grow::ExternalEngine::memory_bytes(d.requested.params.get_k(), cfg, *d.scratch, batch));
            if (d.scratch && args.construction == "merged")
                bytes = std::max(bytes, group_grow::MergeEngine::memory_bytes(d.requested.params.get_k(), cfg, batch));
            return sum(sum(bytes, batch_metadata(batch)), use_stream_input ?
                group_grow::ExternalEngine::input_memory_bytes(cfg, batch) : 0);
        };
        uint16_t const minimum = std::max(uint16_t(1), d.recover_count);
        uint64_t const floor = memory(1, minimum);
        if (floor > args.max_ram - std::min(args.max_ram, sum(sum(fixed, held), backend->retained())))
            return blocked(ControllerWait::engine_memory);
        Producer probe{d.requested, nullptr, *backend, args, args.max_ram, sum(generation_fixed(), floor), 0};
        uint64_t const after_recovery = d.needs_resume ? d.saved.target_members : d.metrics.members;
        bool const needs_generation = d.recover_count || (!d.needs_finish && after_recovery < d.maximum);
        uint64_t const recovery_prefix = product(minimum - 1, member_storage_bound(args, d.requested));
        uint64_t generation = backend->retained();
        auto admit_producer = [&](uint64_t workspace)
        {
            if (!needs_generation) return true;
            probe.fixed = sum(generation_fixed(), workspace);
            bool borrowed = probe.can_borrow(sum(other_held, recovery_prefix));
            if (!probe.fits(d.saved.old_members, other_held, borrowed && minimum == 1))
            {
                if (minimum != 1 || borrowed || !probe.can_borrow(sum(other_held, recovery_prefix)) ||
                    !probe.fits(d.saved.old_members, other_held, true)) return false;
                borrowed = true;
            }
            generation = std::max(probe.next_peak(d.saved.old_members, borrowed), backend->retained());
            return true;
        };
        if (!admit_producer(floor)) return blocked(ControllerWait::producer_memory);
        size_t unfinished_groups = 0;
        std::vector<bool> seen(writer_count, false);
        for (auto const& other : destinations) if (!other->complete && !seen[other->writer])
        { seen[other->writer] = true; ++unfinished_groups; }
        size_t const receiving_lanes = pooled_generation ? size_t(args.pool_inflight) : 0;
        if (pooled_generation)
        {
            // Remote receivers can make progress independently even when
            // their final outputs share one ordered physical writer. The
            // serial local-producer forecast cannot price those task lanes.
            unfinished_groups = size_t(std::count_if(destinations.begin(), destinations.end(),
                [](auto const& other) { return !other->complete; }));
        }
        auto const shared_fixed = sum(sum(metadata, histories.memory_bytes()), pool_memory);
        uint64_t const storage = d.observed_member_storage ? d.observed_member_storage : member_storage_bound(args, d.requested);
        uint16_t const maximum = uint16_t(std::max<uint64_t>(minimum,
            std::min(args.batch, d.maximum - d.metrics.members)));
        auto estimate = [&](size_t count, unsigned grant, uint16_t batch)
        {
            if (std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
                { return other->engine && other->stream_input_bytes; })) return group_grow::ExecutionEstimate{};
            std::vector<group_grow::ExecutionLane> lanes;
            std::vector<group_grow::StorageDevice> resources;
            lanes.reserve(destinations.size()); resources.reserve(2 * destinations.size());
            long double const member_ns = backend->generation_history.member_ns();
            if (!member_ns) return group_grow::ExecutionEstimate{};
            for (auto const& other : destinations) if (!other->complete)
            {
                if (other->needs_resume || other->needs_finish ||
                    (other->scratch && other->scratch->scratch.size() != 1)) return group_grow::ExecutionEstimate{};
                auto options = args; options.workers = grant;
                auto source = construction_history(&histories, options, other->path,
                    initial_identity(other->requested), grant, false, other->scratch ? &*other->scratch : nullptr, {}, true);
                if (!source) return group_grow::ExecutionEstimate{};
                auto const observed = source->snapshot();
                long double const shape = group_grow::construction_shape(observed.shapes);
                if (!shape || other->metrics.members >= other->maximum) return group_grow::ExecutionEstimate{};
                auto cfg = configuration(options, other->metrics.capacity_bytes, shared_fixed, cpu_scheduler, backend->remote_encoder());
                if (other->scratch) cfg.merge_inputs = other->scratch->fan_in;
                auto const forecast = other->scratch ? group_grow::construction_external_forecast(other->requested.params.get_k(), cfg,
                    uint16_t(other->metrics.members), batch, other->maximum, shape,
                    group_grow::construction_shape(observed.execution_contributions), other->scratch->fan_in,
                    other->scratch->packing_inputs ? other->scratch->packing_inputs : other->scratch->fan_in, other->execution_runs) :
                    group_grow::construction_forecast(other->requested.params.get_k(), cfg, uint16_t(other->metrics.members),
                        other->metrics.final_bytes, batch, other->maximum, shape);
                bool const merging = other->scratch || other->mode == "merged" ||
                    (args.construction == "auto" && group_grow::prefer_merged(forecast, cfg, observed));
                auto work = group_grow::construction_execution_work(forecast, observed,
                    member_ns * (forecast.members > other->metrics.members ? forecast.members - other->metrics.members : 0), merging, bool(other->scratch), args.io_overlap);
                if (!work.known) return group_grow::ExecutionEstimate{};
                group_grow::ExecutionLane lane{work, resources.size()};
                resources.push_back(devices[other->owner]);
                if (other->scratch)
                {
                    lane.scratch = resources.size();
                    resources.push_back(group_grow::storage_device(file_io::native_path(other->scratch->scratch.front().directory)));
                }
                lanes.push_back(lane);
            }
            if (std::any_of(resources.begin(), resources.end(), [](auto const& device) { return device.drives.empty(); }))
                return group_grow::ExecutionEstimate{};
            auto const physical = group_grow::writer_groups(resources);
            for (auto& lane : lanes)
            {
                lane.destination = physical[lane.destination];
                if (lane.scratch != SIZE_MAX) lane.scratch = physical[lane.scratch];
            }
            return group_grow::pipeline_execution_estimate(lanes, count, workers, grant,
                *std::max_element(physical.begin(), physical.end()) + 1,
                args.generation_overlap && backend->generation_overlap);
        };
        auto cost = [&](size_t count, unsigned grant, uint16_t batch)
        {
            auto const projected = estimate(count, grant, batch);
            return projected.known ? projected.completion : std::numeric_limits<long double>::infinity();
        };
        auto neutral = [](size_t count, unsigned, uint16_t) { return 1.0L / count; };
        auto plan = destination_plan(args.max_ram, shared_fixed, generation, storage,
            unfinished_groups, workers, maximum, minimum, memory, neutral, receiving_lanes);
        // Every physical destination must have matched completed observations.
        // Unknown contexts keep ordinary exact RAM/topology admission.
        if (!pooled_generation && plan.active && std::isfinite(cost(plan.active, plan.workers, plan.batch)))
            if (auto priced = destination_plan(args.max_ram, shared_fixed, generation, storage,
                unfinished_groups, workers, maximum, minimum, memory, cost); priced.active) plan = priced;
        if (!plan.active) return false;
        // Saved roots only recover their existing jobs. Fresh roots come from
        // explicit directories and the enabled destination policy.
        if (fresh_roots && scratch_pool && !d.guard && !d.scratch && plan.active &&
            (args.construction == "external" || (args.construction == "auto" && plan.batch < d.maximum)))
        {
            auto options = args; options.workers = plan.workers;
            auto config = configuration(options, d.metrics.capacity_bytes, shared_fixed, cpu_scheduler, backend->remote_encoder());
            d.scratch = temporary_grant(*scratch_pool, d.scratch_owner, d.path.parent_path(), d.requested,
                uint16_t(d.maximum), plan.batch, config, args.construction == "auto");
            d.metrics.capacity_bytes = config.capacity_bytes; fresh_scratch = d.scratch.has_value();
            update_pool_memory();
            if (d.scratch)
            {
                DestinationPlan external_plan;
                if (admit_producer(memory(1, minimum)))
                    external_plan = destination_plan(args.max_ram, sum(sum(metadata, histories.memory_bytes()), pool_memory),
                        generation, storage, unfinished_groups, workers, maximum, minimum, memory,
                        neutral, receiving_lanes);
                if (external_plan.active)
                {
                    plan = external_plan;
                    if (!pooled_generation && std::isfinite(cost(plan.active, plan.workers, plan.batch)))
                        if (auto priced = destination_plan(args.max_ram,
                            sum(sum(metadata, histories.memory_bytes()), pool_memory), generation, storage,
                            unfinished_groups, workers, maximum, minimum, memory, cost); priced.active) plan = priced;
                }
                else
                {
                    (void)defer();
                    if (args.construction == "external") return false;
                }
            }
            else if (args.construction == "external") return false;
        }
        // A compact-input cohort may trade retained batches for another
        // independent physical writer. Its transport cost is unknown until
        // complete durable work has been compared with an ordinary cohort.
        auto const stream_fixed = sum(sum(metadata, histories.memory_bytes()), pool_memory);
        bool const live_stream = std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
            { return other->engine && other->stream_input_active; });
        bool const compatible_active = scratch_pool && std::all_of(destinations.begin(), destinations.end(), [&](auto const& other)
            { return !other->engine || scratch_pool->io_compatible(d.scratch_owner, other->scratch_owner); });
        if (live_stream && !compatible_active) return defer();
        bool const ordinary_peer = std::any_of(destinations.begin(), destinations.end(), [&](auto const& other)
            { return other->engine && other->stream_input_candidate && active < other->stream_input_active; });
        uint64_t const fragment_count = fragment_bound(d.requested.params) / sizeof(uint64_t);
        bool const trial_boundary = prefetched.members.empty() &&
            std::none_of(destinations.begin(), destinations.end(), [](auto const& other) { return bool(other->successor); });
        auto candidate_context = construction_history(&histories, args, d.path, initial_identity(d.requested),
            workers, false, d.scratch ? &*d.scratch : nullptr, {}, true);
        if (!pooled_generation && d.scratch && d.scratch->scratch.size() == 1 && !d.guard && !d.needs_resume && !d.needs_finish &&
            !d.metrics.members && args.member_storage == "compact" && args.io_overlap && args.generation_overlap &&
            backend->generation_overlap && fragment_count && fragment_count <= UINT32_MAX && trial_boundary &&
            compatible_active && stream_cohort.stream_allowed(active) && stream_cohort.context_allowed(d.owner, d.maximum,
                reinterpret_cast<uintptr_t>(candidate_context.get())))
        {
            auto const source_device = group_grow::storage_device(file_io::native_path(d.scratch->scratch.front().directory));
            bool independent = source_device.nonrotating && !source_device.drives.empty() && !devices[d.owner].drives.empty();
            for (auto const& drive : source_device.drives)
                if (std::find(devices[d.owner].drives.begin(), devices[d.owner].drives.end(), drive) != devices[d.owner].drives.end())
                    independent = false;
            // Distinct outputs do not imply independent consumers: one
            // destination can overlap another owner's scratch. Match normal
            // I/O admission and recheck peers after their grants are known.
            std::vector<uint64_t> cohort; cohort.reserve(destinations.size());
            cohort.push_back(d.scratch_owner);
            bool compatible_cohort = true;
            for (auto const& other : destinations) if (other->engine)
            {
                compatible_cohort = compatible_cohort && std::all_of(cohort.begin(), cohort.end(), [&](uint64_t owner)
                    { return scratch_pool->io_compatible(owner, other->scratch_owner); });
                cohort.push_back(other->scratch_owner);
            }
            for (auto const& other : destinations) if (!other->complete && !other->engine && other.get() != &d &&
                std::all_of(cohort.begin(), cohort.end(), [&](uint64_t owner)
                    { return scratch_pool->io_compatible(owner, other->scratch_owner); }))
                cohort.push_back(other->scratch_owner);
            auto stream_memory = [&](unsigned grant, uint16_t count)
            {
                auto options = args; options.workers = grant;
                auto cfg = configuration(options, d.metrics.capacity_bytes, fixed, cpu_scheduler, backend->remote_encoder());
                return sum(memory(grant, count), group_grow::ExternalEngine::input_memory_bytes(cfg, count));
            };
            auto stream_cost = [&](size_t count, unsigned grant, uint16_t batch)
            {
                uint64_t const workspace = stream_memory(grant, batch);
                // An ordinary singleton can wait for pending consumers to join;
                // every selected engine still remains resident during fallback.
                bool const fallback = stream_fixed < args.max_ram && generation <= args.max_ram - stream_fixed &&
                    workspace <= (args.max_ram - stream_fixed - generation) / count;
                return fallback ? 1.0L / count : std::numeric_limits<long double>::infinity();
            };
            // Preserve the ordinary plan's useful batch ceiling. Scratch can
            // reduce it before the zero-retention memory plan selects a batch;
            // exact preparation still admits segment headers and alignment.
            uint16_t const scratch_batch = admitted_batch_cap(std::min(maximum, plan.batch), minimum,
                d.scratch->scratch.front().capacity_bytes, 0, [&](uint16_t count)
                { return group_grow::ExternalEngine::input_scratch_bytes(d.requested.params.get_k(), count, fragment_count); });
            auto const streamed = scratch_batch && compatible_cohort ? destination_plan(args.max_ram, stream_fixed,
                std::max(probe.input_peak(uint16_t(after_recovery)), backend->retained()), 0,
                cohort.size(), workers, scratch_batch, minimum, stream_memory, stream_cost) : DestinationPlan{};
            bool const capacity = streamed.active && group_grow::ExternalEngine::input_scratch_bytes(d.requested.params.get_k(),
                streamed.batch, fragment_count) <= d.scratch->scratch.front().capacity_bytes;
            if (group_grow::streamed_input_candidate(plan.active, streamed.active, independent, capacity,
                streamed.active && std::isfinite(stream_cost(streamed.active, streamed.workers, streamed.batch)), false, !trial_boundary))
            { plan = streamed; use_stream_input = true; }
        }
        // A peer using ordinary singleton input can join a live compact-input
        // plan. The exact live checks below still admit every byte and lease.
        if (!use_stream_input && ordinary_peer)
        {
            for (auto const& other : destinations) if (other->engine && other->stream_input_candidate &&
                active < other->stream_input_active && plan.active < other->stream_input_active)
            {
                plan.active = other->stream_input_active;
                plan.workers = std::max(1u, unsigned(workers / plan.active)); plan.batch = 1;
            }
        }
        if (!plan.active || active >= plan.active) return defer();
        total.execution_active = plan.active; total.execution_batch = plan.batch;
        total.execution_estimate = use_stream_input || pooled_generation ? group_grow::ExecutionEstimate{} :
            estimate(plan.active, plan.workers, plan.batch);
        unsigned const share = plan.workers;
        if (!admit_producer(memory(share, minimum))) return defer();
        if (fixed >= args.max_ram) return defer();
        uint64_t const available = args.max_ram - generation_fixed();
        if (other_held > available || generation > available - other_held) return defer();
        uint16_t const batch = admitted_batch_cap(std::min(maximum, plan.batch), minimum,
            available - other_held - generation, use_stream_input ? 0 : storage,
            [&](uint16_t n) { return memory(share, n); });
        if (!batch) return defer();
        uint64_t const lease = memory(share, batch);
        if (!admit_producer(lease)) return defer();
        if (sum(sum(sum(fixed, lease), held), backend->retained()) > args.max_ram) return defer();
        uint64_t const host = available_host_bytes();
        uint64_t const growth = generation > backend->retained() ? generation - backend->retained() : 0;
        if (host && sum(lease, growth) > host) return defer();
        check_farm();
        for (auto const& other : destinations) check_guard(*other);
        auto options = args; options.workers = share;
        std::optional<group_grow::ScratchPool::IoLease> io;
        if (scratch_pool)
        {
            io = scratch_pool->try_io(d.scratch_owner);
            if (!io) { blocked(ControllerWait::storage_lease); return defer(); }
        }
        auto const history_before = histories.memory_bytes();
        // Rotating farming meta values introduce more contexts than lanes.
        // Reuse matching observations, but keep the existing bounded table;
        // a new context can use ordinary resource admission without a history.
        auto source = construction_history(&histories, options, d.path, initial_identity(d.requested), share,
            !farm || history_contexts < paths.size(), d.scratch ? &*d.scratch : nullptr);
        if (farm && histories.memory_bytes() > history_before) ++history_contexts;
        fixed = sum(fixed, histories.memory_bytes() - history_before);
        auto generation_peak = [&](uint64_t workspace)
        { return sum(sum(sum(generation_fixed(), workspace), sum(other_held, product(batch - 1, use_stream_input ? 0 : storage))), generation); };
        auto transfer_peak = [&](uint64_t workspace)
        { return sum(sum(sum(fixed, workspace), held), backend->retained()); };
        if (std::max(generation_peak(memory(share, batch)), transfer_peak(memory(share, batch))) > args.max_ram) return defer();
        // Preserve every compact-input owner's ordinary singleton fallback
        // when a later peer needs a different engine envelope. Queued member
        // payloads may drain first, but all engine workspaces remain resident.
        bool preserve_fallback = use_stream_input;
        uint64_t fallback_generation = generation;
        for (auto const& other : destinations) if (other->engine && other->stream_input_bytes)
        {
            preserve_fallback = true;
            fallback_generation = std::max(fallback_generation, backend->generation_peak(other->requested, args));
        }
        if (preserve_fallback && !group_grow::execution_singleton_fallback(args.max_ram, generation_fixed(),
            memory(share, batch), fallback_generation)) return defer();
        auto cfg = configuration(options, d.metrics.capacity_bytes, sum(fixed, batch_metadata(batch)), cpu_scheduler, backend->remote_encoder());
        if (d.scratch) d.scratch->target_gate = target_gates[d.writer];
        auto engine = std::make_unique<Construction>(file_io::path_text(d.path), cfg, bool(d.guard),
            initial_identity(d.requested), args.construction, d.scratch, d.guard ? d.guard->retired : 0);
        if (engine->external && !args.checkpoint_set)
            engine->external->tune_packing_checkpoints(group_grow::PackingCheckpointHistory::maximum_bytes);
        engine->attach_history(std::move(source));
        Construction::same_saved(d.saved, engine->identity());
        if (d.guard) check_guard(d);
        uint64_t const input_bytes = use_stream_input ? group_grow::ExternalEngine::input_memory_bytes(cfg, batch) : 0;
        uint64_t const engine_bytes = sum(engine->memory_bytes(batch, args.member_storage == "compact"), input_bytes);
        uint64_t const actual = sum(engine_bytes, batch_metadata(batch));
        uint64_t const peak = std::max(generation_peak(actual), transfer_peak(actual));
        require(peak <= args.max_ram,
            "active destination workspace exceeds --max-ram");
        require(!preserve_fallback || group_grow::execution_singleton_fallback(args.max_ram,
            generation_fixed(), actual, fallback_generation), "active destination consumed compact-input fallback headroom");
        engine->confirm_identity();
        d.engine = std::move(engine); d.batch_cap = batch;
        d.stream_input_candidate = use_stream_input; d.stream_input_batch = use_stream_input ? batch : 0;
        d.stream_input_bytes = input_bytes; d.stream_input_active = use_stream_input ? plan.active : 0;
        if (d.engine->external) d.engine->external->run_shapes(d.execution_runs);
        auto context = construction_history(&histories, options, d.path, initial_identity(d.requested), share,
            false, d.scratch ? &*d.scratch : nullptr, {}, true);
        stream_cohort.admit(d.owner, d.maximum, reinterpret_cast<uintptr_t>(context.get()),
            d.stream_input_candidate, execution_now(), d.stream_input_active ? d.stream_input_active : 1);
        d.engine_bytes = engine_bytes; d.lease_bytes = actual;
        fixed = sum(fixed, actual); ++active;
        total.engine_reserved_bytes = std::max(total.engine_reserved_bytes, fixed);
        total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, peak);
        if (!writers[d.writer]) writers[d.writer] = std::make_unique<group_grow::OrderedWriter>([&] { wake.notify_one(); });
        return true;
    };
    auto release = [&](Destination& d)
    {
        auto& engine = *d.engine;
        d.metrics.staged_batches = engine.staged_batches; d.metrics.staging_fallbacks = engine.fallbacks;
        d.metrics.automatic_merges = engine.automatic_merges; d.metrics.automatic_fallbacks = engine.automatic_fallbacks;
        d.metrics.tuning_changes = engine.tuning_changes;
        d.metrics.selected_window_bytes = engine.config.window_bytes;
        d.metrics.selected_checkpoint_bytes = engine.selected_checkpoint_bytes();
        d.metrics.construction_selected = engine.mode();
        engine.confirm_identity(); d.guard = snapshot(d.path, 0); engine.release_external();
        d.metrics.scratch_peak_bytes = engine.peak_scratch_bytes;
        d.metrics.scratch_released = engine.scratch_cleanup_complete;
        static_cast<group_grow::Stats&>(d.metrics) = Construction::add_stats(d.metrics, engine.stats());
        if (scratch_pool && engine.scratch_cleanup_complete)
            scratch_pool->release_after_cleanup(d.scratch_owner, sum(allocated_bytes(d.path), engine.remaining_metadata_bytes));
        update_pool_memory();
        require(!d.workers, "destination released before its worker task joined");
        fixed -= d.lease_bytes; --active;
        d.engine.reset(); d.engine_bytes = d.lease_bytes = d.packing_released_bytes = 0; d.workers = 0;
        d.stream_input_candidate = false; d.stream_input_batch = 0; d.stream_input_bytes = 0; d.stream_input_active = 0;
        if (!d.metrics.final_bytes)
        {
            file_io::Stat file{};
            if (file_io::inspect_link(d.path.c_str(), &file) != 0 && errno == ENOENT) d.guard.reset();
        }
    };
    auto adopt_successor = [&](Destination& d, GroupSuccessor& next)
    {
        require(!next.io && !next.workers, "successor transferred before its staging task joined");
        d.path = std::move(next.path); d.requested = std::move(next.requested);
        d.scratch_owner = next.owner; d.maximum = std::min(d.maximum, next.maximum);
        d.staging_successor = false; d.needs_resume = false; d.recover_count = 0;
        d.packing_released_bytes = 0;
        bool const live = bool(next.engine);
        if (live)
        {
            d.scratch = std::move(next.scratch); d.engine = std::move(next.engine);
            d.engine->external->run_shapes(d.execution_runs);
            d.saved = d.engine->identity(); d.guard = snapshot(d.path, 0);
            d.metrics = std::move(next.metrics); d.metrics.staged_batches = d.engine->staged_batches;
            d.batch_cap = next.batch_cap; d.lease_bytes = next.engine_bytes;
            d.engine_bytes = d.lease_bytes - batch_metadata(d.batch_cap);
            d.needs_finish = false; d.mode = "external";
        }
        else
        {
            require(next.recovery.has_value(), "saved successor lost its audited checkpoint");
            auto& saved = *next.recovery;
            d.scratch = std::move(saved.external); d.saved = std::move(saved.identity); d.guard = next.guard;
            d.metrics.members = saved.members; d.metrics.final_bytes = d.saved.bytes;
            d.metrics.peak_file_bytes = sum(saved.peak_bytes, d.guard->retired);
            d.metrics.recovered = true; d.needs_finish = saved.finishing; d.mode = "external";
        }
        fixed -= next.metadata_bytes; metadata -= next.metadata_bytes;
        return live;
    };
    auto advance = [&](Destination& d)
    {
        if (!farm) return false;
        auto& lane = farm->lanes[d.owner];
        auto discard_future = [&]
        {
            if (d.successor && !d.successor->detached)
                accumulate(detach_successor(*d.successor, fixed, active, lane.successor_ordinal), "external");
            // The guarded output rows retain every acknowledged member. Drop
            // only the ungenerated promise and keep the record for disk peaks.
            lane.reserved_members = lane.next_reserved_members = 0;
            if (!prefetched.members.empty() && prefetched.owner == d.owner)
            { fixed -= prefetched.metadata_bytes; prefetched.clear(); }
        };
        if (!d.metrics.members) { discard_future(); return false; }
        farm_peak();
        auto output = snapshot(d.path, lane.ordinal);
        output_auxiliary(d.path, output, args.max_ram - metadata);
        output.members = output.pending_members = uint16_t(d.metrics.members);
        if (lane.ordinal < lane.outputs.size()) lane.outputs[lane.ordinal] = output;
        else
        {
            uint64_t const extra = d.prefunded_rows ? 0 : sizeof(Output) * 4;
            require(sum(sum(sum(fixed, extra), retained()), backend->retained()) <= args.max_ram,
                "farming output metadata and retained buffers exceed --max-ram");
            if (d.prefunded_rows) --d.prefunded_rows;
            lane.outputs.push_back(output); metadata = sum(metadata, extra); fixed = sum(fixed, extra);
            farm->metadata = sum(farm->metadata, extra);
        }
        if (d.metrics.stop_reason == "capacity" || !d.metrics.scratch_released) { discard_future(); return false; }
        lane.completed_charge = sum(lane.completed_charge, output.charge());
        lane.completed_members = sum(lane.completed_members, output.members);
        lane.completed_bytes = sum(lane.completed_bytes, output.bytes);
        lane.successor_ordinal.reset(); ++lane.ordinal; d.farm_counted = true;
        lane.reserved_members = lane.next_reserved_members; lane.next_reserved_members = 0;
        uint16_t const maximum = farm->maximum(args, d.owner);
        auto const quota = std::min(farm->state.directories[d.owner].capacity_bytes, d.farm_capacity);
        require(lane.completed_charge <= quota, "completed farming outputs exceed their planned capacity");
        uint64_t const room = std::min(quota - lane.completed_charge,
            group_grow::maximum_file_bytes(file_io::path_text(d.path.parent_path())));
        if (!room) { lane.no_capacity = true; d.metrics.stop_reason = "capacity"; discard_future(); return false; }
        if (!maximum) { discard_future(); return false; }
        file_io::Space available{};
        require(!file_io::space_path(d.path.parent_path().c_str(), &available), "cannot inspect next farming-output capacity");
        if (product(available.f_bavail, available.f_frsize) <= args.headroom)
        { lane.no_capacity = true; d.metrics.stop_reason = "capacity"; discard_future(); return false; }
        check_farm();
        accumulate(d.metrics, d.mode);
        d.path = farm->path(d.owner); d.requested = farm->identity(*keys, d.owner, lane.ordinal);
        d.saved = initial_identity(d.requested); d.guard.reset(); d.scratch.reset(); d.metrics = {};
        d.maximum = maximum; d.batch_cap = d.recover_count = 0; d.observed_member_storage = 0;
        d.needs_resume = d.needs_finish = false; d.complete = d.farm_counted = false; d.mode = args.construction;
        d.capacity_limit = scratch_pool ? temporary_output_capacity(args, d.requested, maximum, room) : room;
        d.metrics.capacity_bytes = room;
        if (d.successor)
        {
            bool const live = adopt_successor(d, *d.successor);
            d.successor.reset(); update_pool_memory();
            if (live) return true;
        }
        if (scratch_pool)
        {
            d.metrics.capacity_bytes = std::min(d.capacity_limit, scratch_pool->available_for_output(d.scratch_owner, d.path.parent_path()));
            require(scratch_pool->try_reserve_output(d.scratch_owner, d.path.parent_path(), d.metrics.capacity_bytes),
                "cannot reserve the next farming output");
            update_pool_memory();
        }
        return true;
    };
    auto prepare_next = [&](Destination& d, unsigned available)
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        // Pool prefetch keeps owning members in bounded RAM. It does not run
        // the synchronous GPU successor-staging controller on this thread.
        if (backend->mode == "pool") return false;
#endif
        if (stream_cohort.calibrating()) return false;
        if (d.successor || !prefetched.members.empty() || !scratch_pool || !d.engine->external || available < 2 ||
            (args.construction != "auto" && args.construction != "external") ||
            d.metrics.members != d.maximum || !d.metrics.stop_reason.empty() ||
            !group_prefetch_enabled(args, *backend)) return false;
        if (std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
            { return other->pending && other->pending->lease.active(); })) return false;
        size_t target = d.owner;
        uint64_t owner = 0, room = 0;
        uint16_t maximum = 0, previous_reservation = 0;
        std::optional<Identity> identity;
        fs::path path;
        if (farm)
        {
            auto& lane = farm->lanes[d.owner];
            uint64_t const members = sum(lane.completed_members, d.metrics.members);
            if (lane.ordinal + 1 >= args.max_groups || members >= args.max_total_members) return false;
            maximum = uint16_t(std::min<uint64_t>(farm->file_maximum(d.owner, lane.ordinal + 1),
                args.max_total_members - members));
            if (!maximum) return false;
            uint64_t const quota = std::min(farm->state.directories[d.owner].capacity_bytes, d.farm_capacity);
            if (lane.completed_charge >= quota || d.metrics.capacity_bytes >= quota - lane.completed_charge) return false;
            room = quota - lane.completed_charge - d.metrics.capacity_bytes;
            owner = d.scratch_owner == d.owner ? paths.size() + d.owner : d.owner;
            path = group_path(d.path.parent_path(), lane.ordinal + 1);
            identity = farm->identity(*keys, d.owner, lane.ordinal + 1);
            previous_reservation = lane.next_reserved_members;
            lane.next_reserved_members = std::max(lane.next_reserved_members, maximum);
            auto const before = farm->metadata;
            try { farm->certify(sum(sum(fixed, retained()), backend->retained())); }
            catch (group_grow::FarmingResourceError const& error)
            {
                lane.next_reserved_members = previous_reservation;
                if (!error.memory_budget) throw;
                return false;
            }
            metadata = sum(metadata - before, farm->metadata); fixed = sum(fixed - before, farm->metadata);
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, farm->planning_peak);
            if (!farm->feasible()) { lane.next_reserved_members = previous_reservation; return false; }
            size_t const needed = size_t(lane.ordinal + 2) - lane.outputs.size();
            size_t const rows = needed > d.prefunded_rows ? needed - d.prefunded_rows : 0;
            uint64_t const extra = product(rows, sizeof(Output) * 4);
            if (sum(sum(sum(fixed, extra), retained()), backend->retained()) >= args.max_ram)
            { lane.next_reserved_members = previous_reservation; return false; }
            lane.outputs.reserve(size_t(lane.ordinal + 2)); d.prefunded_rows += rows;
            metadata = sum(metadata, extra); fixed = sum(fixed, extra); farm->metadata = sum(farm->metadata, extra);
        }
        else
        {
            auto const found = std::find_if(destinations.begin(), destinations.end(), [&](auto const& other)
            {
                return other.get() != &d && other->writer == d.writer && !other->complete &&
                    !other->guard && !other->engine && !other->pending && !other->staging_successor;
            });
            if (found == destinations.end()) return false;
            auto& future = **found; target = future.owner; owner = future.scratch_owner;
            path = future.path; identity = future.requested; maximum = future.maximum;
            room = future.metrics.capacity_bytes;
            scratch_pool->cancel(owner); future.output_reserved = false;
            update_pool_memory();
        }
        auto next = std::make_unique<GroupSuccessor>(path, *identity, maximum, room);
        bool const admitted = prepare_successor(*next, args, *backend, *scratch_pool, owner, fixed, retained(),
            d.engine->staging_workers(available), target_gates[d.writer], nullptr, &histories,
            d.engine->demand_history, d.packing_released_bytes, cpu_scheduler);
        if (!admitted)
        {
            if (farm) farm->lanes[d.owner].next_reserved_members = previous_reservation;
            else
            {
                auto& future = *destinations[target];
                require(scratch_pool->try_reserve_output(owner, future.path.parent_path(), room),
                    "cannot restore an unused successor destination reservation");
                future.output_reserved = true;
            }
            update_pool_memory(); return false;
        }
        d.successor = std::move(next); d.successor_target = target; d.successor_staged = false;
        metadata = sum(metadata, d.successor->metadata_bytes);
        fixed = sum(fixed, sum(d.successor->metadata_bytes, d.successor->engine_bytes));
        update_pool_memory(); ++active;
        require(fixed == d.successor->fixed, "successor workspace accounting changed during admission");
        total.engine_reserved_bytes = std::max(total.engine_reserved_bytes, fixed);
        total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, d.successor->metrics.peak_modeled_bytes);
        if (farm)
        {
            auto& lane = farm->lanes[d.owner]; lane.successor_ordinal = lane.ordinal + 1;
            if (lane.outputs.size() == lane.ordinal)
            {
                auto output = snapshot(d.path, lane.ordinal);
                output_auxiliary(d.path, output, args.max_ram - metadata);
                output.members = output.pending_members = uint16_t(d.metrics.members); output.pending = true;
                require(d.prefunded_rows, "successor lost its prefunded output metadata");
                --d.prefunded_rows; lane.outputs.push_back(output);
            }
            auto output = snapshot(d.successor->path, lane.ordinal + 1);
            output_auxiliary(d.successor->path, output, args.max_ram - metadata); output.pending = true;
            d.successor->guard = output;
            require(d.prefunded_rows, "successor lost its prefunded output metadata");
            --d.prefunded_rows; lane.outputs.push_back(output);
        }
        else
        {
            auto& future = *destinations[target]; future.staging_successor = true;
            future.output_reserved = true;
        }
        return true;
    };
    auto submit = [&](Destination& d)
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (d.pending->generating.valid()) return blocked(ControllerWait::consumer);
#endif
        check_farm();
        for (auto const& other : destinations) check_guard(*other);
        auto* task = d.pending.get();
        if (task->work == Work::finish && d.engine->external && !d.engine->terminal_packing)
            task->work = Work::prepare_finish;
        unsigned count = workers;
        // Each bounded codec task acquires CPU admission from the shared pool.
        // File consumers retain their RAM grant while waiting for I/O; they
        // never reserve CPU workers across write or durability waits.
        size_t runnable = 1;
        for (auto const& other : destinations)
            if (other.get() != &d && other->engine && other->pending && !other->ready.valid() &&
                other->writer != d.writer) ++runnable;
        count = std::max(1u, unsigned(count / runnable));
        if (task->work == Work::prepare_finish || d.engine->staged || (!d.engine->is_staging() &&
            (task->work == Work::finish || task->work == Work::recover_finish))) count = 1;
        uint64_t const workspace = d.engine_bytes - d.stream_input_bytes;
        auto fits_workers = [&](unsigned wanted)
        {
            auto cfg = d.engine->config; cfg.workers = wanted;
            return d.engine->memory_for(cfg, d.batch_cap, args.member_storage == "compact", false) <= workspace;
        };
        while (count > 1 && !fits_workers(count)) --count;
        if (!fits_workers(count)) return blocked(ControllerWait::engine_memory);
        if (scratch_pool)
        {
            task->io = scratch_pool->try_io(d.scratch_owner);
            if (!task->io) return blocked(ControllerWait::storage_lease);
        }
        if (task->work == Work::finish && prepare_next(d, count))
            d.engine->admit_staging_workers(count, d.successor->workers);
        if (task->work != Work::prepare_finish && (d.engine->is_staging() ||
            (task->work != Work::finish && task->work != Work::recover_finish)))
        {
            auto options = args; options.workers = count;
            auto source = construction_history(&histories, options, d.path, initial_identity(d.requested), count, false,
                d.scratch ? &*d.scratch : nullptr);
            d.engine->set_workers(count, d.batch_cap, args.member_storage == "compact", workspace);
            if (source) d.engine->attach_history(std::move(source));
        }
        d.workers = count;
        d.ready = writers[d.writer]->submit([&, task]
        {
            constexpr std::array names{"append", "recover", "prepare_pack", "finish", "recover_finish", "input_fallback"};
            ControllerTiming timing(names[size_t(task->work)], d.owner);
            switch (task->work)
            {
                case Work::append:
                    if (task->select_merged) { d.engine->select_merged(!args.quiet); task->select_merged = false; }
                    task->appended = task->contribution_input ? d.engine->append(*task->contribution_input, task->prepared_members) :
                        task->prepared_input ? d.engine->append(*task->prepared_input, task->prepared_members) :
                        d.engine->append(task->members, task->hash.get()); break;
                case Work::recover: task->finished = d.engine->resume(task->members, task->hash.get()); break;
                case Work::prepare_finish: d.engine->prepare_finish(); break;
                case Work::finish: task->finished = d.engine->finish(); break;
                case Work::recover_finish: task->finished = d.engine->finish_staging(args.construction == "merged"); break;
                case Work::fallback_input: task->finished = d.engine->fallback_input(); break;
            }
        });
        return true;
    };
    auto accept = [](Destination& d, group_grow::Result const& result)
    {
        d.metrics.members = result.members; d.metrics.final_bytes = result.final_bytes;
        d.metrics.peak_file_bytes = std::max(d.metrics.peak_file_bytes, result.peak_file_bytes);
        d.metrics.workspace_bytes = std::max(d.metrics.workspace_bytes, result.workspace_bytes);
    };
    auto prefetch = [&](Destination& d)
    {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (pool_prefetch.ready.valid()) return false;
#endif
        if (stream_cohort.calibrating()) return false;
        if (d.successor || d.staging_successor) return false;
        bool const rollover = farm && d.engine && d.ready.valid() && d.pending &&
            d.pending->work == Work::finish && d.metrics.members == d.maximum &&
            d.metrics.stop_reason.empty();
        if (!prefetched.members.empty() || (!rollover && (d.guard || d.engine || d.pending)) ||
            !group_prefetch_enabled(args, *backend)) return false;
        auto identity = d.requested;
        uint16_t maximum = d.maximum;
        if (rollover)
        {
            auto const& lane = farm->lanes[d.owner];
            uint64_t const members = sum(lane.completed_members, d.metrics.members);
            if (lane.ordinal + 1 >= args.max_groups || members >= args.max_total_members) return false;
            maximum = uint16_t(std::min<uint64_t>(farm->file_maximum(d.owner, lane.ordinal + 1),
                args.max_total_members - members));
            if (!maximum) return false;
            identity = farm->identity(*keys, d.owner, lane.ordinal + 1);
        }
        auto const packing = std::find_if(destinations.begin(), destinations.end(), [&](auto const& other)
        {
            return other->writer == d.writer && other->ready.valid() && other->pending &&
                other->pending->work == Work::finish;
        });
        if (packing == destinations.end()) return false;
        if (std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
            { return other->pending && other->pending->lease.active(); })) return false;
        auto const done = [&]
        { return (*packing)->ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready; };
        if (done()) return false;
        check_farm();
        for (auto const& other : destinations) check_guard(*other);
        uint64_t const held = retained();
        auto options = args; options.workers = (*packing)->workers;
        auto cfg = configuration(options, d.metrics.capacity_bytes, fixed, cpu_scheduler, backend->remote_encoder());
        auto future_memory = [&](uint16_t count)
        { return prefetch_memory(args, identity, cfg, count, devices[d.owner].nonrotating); };
        Producer producer{identity, nullptr, *backend, args, args.max_ram,
            sum(fixed, future_memory(1)), fixed};
        if (!producer.fits(0, held)) return false;
        uint64_t const generation = std::max(producer.next_peak(0), backend->retained());
        uint64_t const occupied = sum(sum(fixed, held), generation);
        if (occupied >= args.max_ram) return false;
        uint16_t cap = admitted_batch_cap(uint16_t(std::min<uint64_t>(args.batch, maximum)), 1,
            args.max_ram - occupied, member_storage_bound(args, identity), future_memory);
        if (!cap) return false;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        uint64_t pool_receive = 0;
        if (backend->mode == "pool")
        {
            while (cap && sum(sum(sum(sum(fixed, held), future_memory(cap)), backend->retained()),
                producer.pool_receive_bytes(0, cap)) > args.max_ram) --cap;
            if (!cap) return false;
            pool_receive = producer.pool_receive_bytes(0, cap);
        }
#endif
        producer.fixed = sum(fixed, future_memory(cap));
        uint16_t previous_reservation = 0;
        if (farm)
        {
            auto& lane = farm->lanes[d.owner];
            auto& reservation = rollover ? lane.next_reserved_members : lane.reserved_members;
            previous_reservation = reservation; reservation = std::max(reservation, maximum);
            // Speculative work uses an existing global certificate. A larger
            // frozen extent is assessed during ordinary activation, with its
            // complete caller/reader memory charge, before any generation.
            if (!farm->covered()) { reservation = previous_reservation; return false; }
        }
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (backend->mode == "pool")
        {
            auto& pending = pool_prefetch;
            pending.batch.owner = d.owner; pending.batch.identity = identity; pending.batch.capacity = cap;
            pending.batch.metadata_bytes = batch_metadata(cap); pending.batch.members.reserve(cap);
            pending.receiving_bytes = pool_receive; pending.cancel.store(false);
            uint64_t const reserved = producer.fixed;
            std::string const progress = file_io::path_text(d.path) + ": prefetching pool member ";
            fixed = sum(fixed, pending.batch.metadata_bytes);
            pending.ready = std::async(std::launch::async, [&, cap, held, reserved, progress]
            {
                Producer producer{*pending.batch.identity, nullptr, *backend, args, args.max_ram, reserved, reserved};
                producer.defer_pool_history = true;
                auto const generated = producer.fill_pool(pending.batch.members, 0, cap, held,
                    progress, {}, &pending.cancel);
                wake.notify_one();
                return Task::Generated{generated, producer.peak, producer.generation_peak,
                    producer.pool_elapsed_ns, producer.fill_stop};
            });
            ++total.drive_overlap_batches; ++total.generation_overlap_batches;
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes,
                sum(sum(fixed, retained()), backend->retained()));
            return true;
        }
#endif
        auto const generated = prefetch_group(prefetched, producer, identity, d.owner, cap, held,
            file_io::path_text(d.path) + ": prefetching " + backend->mode + " member ", done);
        if (!generated)
        {
            if (farm)
            {
                auto& lane = farm->lanes[d.owner];
                (rollover ? lane.next_reserved_members : lane.reserved_members) = previous_reservation;
            }
            return false;
        }
        prefetched.metadata_bytes = batch_metadata(cap);
        (*packing)->prefetch_wait_reason = fill_wait_reason(producer.fill_stop);
        fixed = sum(fixed, prefetched.metadata_bytes);
        d.metrics.generated_members = sum(d.metrics.generated_members, generated);
        ++total.drive_overlap_batches; ++total.generation_overlap_batches;
        total.generation_overlap_members = sum(total.generation_overlap_members, producer.overlap_members);
        total.gpu_generation_peak_bytes = std::max(total.gpu_generation_peak_bytes, producer.generation_peak);
        total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, producer.peak);
        (void)retained();
        return true;
    };
    size_t next = 0, complete = size_t(std::count_if(destinations.begin(), destinations.end(),
        [](auto const& d) { return d->complete; }));
    if (farm) for (size_t i = 0; i < farm->lanes.size(); ++i)
    {
        auto const& lane = farm->lanes[i];
        for (auto const& output : lane.outputs) if (output.ordinal < lane.ordinal)
            report_output(args, group_path(file_io::native_path(farm->state.directories[i].directory), output.ordinal),
                uint16_t(output.members), output.bytes, true);
    }
    for (auto const& d : destinations) if (d->complete && d->metrics.members)
        report_output(args, d->path, uint16_t(d->metrics.members), d->metrics.final_bytes, true);
    while (complete < destinations.size())
    {
        check_cancel(args);
        bool progress = false;
        wait_reasons = 0;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (pool_prefetch.ready.valid() &&
            pool_prefetch.ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            auto const generated = pool_prefetch.ready.get();
            auto& d = *destinations[pool_prefetch.batch.owner];
            d.metrics.generated_members = sum(d.metrics.generated_members, generated.members);
            backend->generated_members = sum(backend->generated_members, generated.members);
            backend->generation_ns = sum(backend->generation_ns, generated.elapsed);
            backend->generation_history.add(generated.members, generated.elapsed);
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, generated.peak);
            total.record_generation(backend->mode, generated.generation_peak);
            require(!prefetched.identity && prefetched.members.empty(), "pool prefetch replaced an owning batch");
            prefetched = std::move(pool_prefetch.batch); pool_prefetch.batch.clear();
            pool_prefetch.receiving_bytes = 0; progress = true;
            if (!generated.members) { fixed -= prefetched.metadata_bytes; prefetched.clear(); }
        }
#endif
        for (auto& destination : destinations)
        {
            auto& d = *destination;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            if (d.pending && d.pending->generating.valid())
            {
                auto& task = *d.pending;
                if (task.generating.wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
                auto const generated = task.generating.get();
                task.receiving_bytes = 0; task.io.reset(); task.fill_stop = generated.stop;
                d.metrics.generated_members = sum(d.metrics.generated_members, generated.members);
                if (task.contribution_input)
                {
                    task.prepared_members = uint16_t(generated.members);
                    d.metrics.stream_input_written_bytes = sum(d.metrics.stream_input_written_bytes,
                        task.contribution_input->written_bytes());
                }
                backend->generated_members = sum(backend->generated_members, generated.members);
                backend->generation_ns = sum(backend->generation_ns, generated.elapsed);
                backend->generation_history.add(generated.members, generated.elapsed);
                total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, generated.peak);
                total.record_generation(backend->mode, generated.generation_peak);
                progress = true;
            }
#endif
            if (!d.ready.valid()) continue;
            if (d.successor && d.successor->engine && !d.successor_staged)
            {
                auto& successor = *d.successor;
                auto const before = successor.metrics.generated_members;
                successor.fixed = fixed;
                successor.metrics.engine_reserved_bytes = std::max(successor.metrics.engine_reserved_bytes, fixed);
                d.successor_wait_reason = stage_successor(successor, args, *backend, retained(), [&]
                {
                    return std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
                    {
                        return other->ready.valid() && other->ready.wait_for(std::chrono::seconds(0)) ==
                            std::future_status::ready;
                    });
                });
                farm_peak();
                total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, successor.metrics.peak_modeled_bytes);
                total.queued_peak_bytes = std::max(total.queued_peak_bytes,
                    sum(retained(), successor.metrics.queued_peak_bytes));
                progress = successor.metrics.generated_members != before || progress;
                bool const done = d.ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                if (done || successor.metrics.members == successor.maximum || successor.deferred)
                {
                    successor.io.reset(); successor.workers = 0;
                    d.successor_staged = true;
                    total.drive_overlap_batches = sum(total.drive_overlap_batches, successor.metrics.batches);
                    total.generation_overlap_batches = sum(total.generation_overlap_batches,
                        successor.metrics.generation_overlap_batches);
                    total.generation_overlap_members = sum(total.generation_overlap_members,
                        successor.metrics.generation_overlap_members);
                    total.gpu_generation_peak_bytes = std::max(total.gpu_generation_peak_bytes,
                        successor.metrics.gpu_generation_peak_bytes);
                    if (farm)
                    {
                        auto& lane = farm->lanes[d.owner];
                        auto output = snapshot(successor.path, lane.ordinal + 1);
                        output_auxiliary(successor.path, output, args.max_ram - metadata);
                        output.members = output.pending_members = uint16_t(successor.metrics.members); output.pending = true;
                        successor.guard = output; lane.outputs[lane.ordinal + 1] = output;
                    }
                    else
                    {
                        auto& target = *destinations[d.successor_target];
                        (void)adopt_successor(target, successor);
                        d.successor.reset(); update_pool_memory();
                    }
                }
            }
            if (d.ready.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            { if (farm) progress = prefetch(d) || progress; continue; }
            d.ready.get(); progress = true; auto& task = *d.pending; task.io.reset();
            if (d.engine->external) d.engine->external->run_shapes(d.execution_runs);
            else d.execution_runs.clear();
            d.successor_wait_reason = d.prefetch_wait_reason = 0;
            d.workers = 0;
            if (task.work == Work::prepare_finish)
            {
                // The durable transition and all its staging allocations have
                // joined. Return only this owner's released grant, then let
                // ordinary scheduling admit final packing and other producers.
                uint64_t const bytes = d.engine->memory_bytes(d.batch_cap, args.member_storage == "compact");
                require(bytes <= d.engine_bytes - d.stream_input_bytes, "final packing exceeds the destination workspace lease");
                uint64_t const released = d.engine_bytes - bytes;
                d.packing_released_bytes = released;
                d.metrics.packing_engine_bytes = std::max(d.metrics.packing_engine_bytes, bytes);
                d.metrics.packing_released_bytes = std::max(d.metrics.packing_released_bytes, released);
                fixed -= released; d.lease_bytes -= released; d.engine_bytes = bytes;
                d.stream_input_candidate = false; d.stream_input_batch = 0; d.stream_input_bytes = 0;
                task.work = Work::finish; continue;
            }
            if (scratch_pool && d.scratch && !d.engine->external && d.engine->scratch_cleanup_complete)
            {
                scratch_pool->release_scratch_after_cleanup(d.scratch_owner); d.scratch.reset(); update_pool_memory();
                uint64_t const capacity = std::min(d.capacity_limit,
                    scratch_pool->available_for_output(d.scratch_owner, d.path.parent_path()));
                if (capacity > d.metrics.capacity_bytes && d.engine->incremental)
                {
                    require(scratch_pool->try_reserve_output(d.scratch_owner, d.path.parent_path(), capacity),
                        "cannot expand destination after scratch cleanup");
                    d.engine->resize_capacity(capacity); d.metrics.capacity_bytes = capacity;
                    if (task.work == Work::append && !task.appended.accepted) continue;
                }
            }
            if (task.work == Work::append)
            {
                auto const& result = task.appended;
                if (task.prepared_input)
                {
                    uint64_t const read = task.prepared_input->read_bytes();
                    d.metrics.stream_input_read_bytes = sum(d.metrics.stream_input_read_bytes, read - task.prepared_read_accounted);
                    task.prepared_read_accounted = read;
                }
                d.metrics.attempted_final_bytes = result.attempted_final_bytes;
                d.metrics.attempted_peak_bytes = result.attempted_peak_bytes;
                if (!d.engine->is_staging() && !result.fresh_single_pass) ++d.metrics.exact_plan_batches;
                if (!result.accepted)
                {
                    std::cerr << d.path << ": candidate batch rejected: " << result.reason << '\n';
                    if ((task.prepared_input || task.contribution_input) && task.prepared_members > 1)
                    {
                        --task.prepared_members; task.fill_stop = Producer::FillStop::requested;
                        continue;
                    }
                    if (task.prepared_input || task.contribution_input)
                    {
                        // The original source remains owned until append joins.
                        // Regenerate the rejected singleton after CPU fallback,
                        // preserving ordinary exact capacity/window admission.
                        task.prepared_input.reset(); task.contribution_input.reset(); task.contribution_bytes = 0;
                        task.prepared_members = 0;
                        task.work = Work::fallback_input; d.stream_input_candidate = false;
                        continue;
                    }
                    if (task.members.size() > 1)
                    {
                        task.members.pop_back(); task.lease.reset();
                        task.fill_stop = Producer::FillStop::requested;
                        if (task.hash) task.hash->truncate(task.members.size());
                        continue; // Resubmit after the drive's queue has room.
                    }
                    d.metrics.stop_reason = "capacity";
                }
                else
                {
                    accept(d, result.result); ++d.metrics.batches;
                    if (task.prepared_input || task.contribution_input) ++d.metrics.stream_input_batches;
                    size_t const committed = task.prepared_input || task.contribution_input ? task.prepared_members : task.members.size();
                    d.metrics.max_batch_members = std::max<uint64_t>(d.metrics.max_batch_members, committed);
                    if (!task.prepared_input && !task.contribution_input)
                        d.observed_member_storage = (resident(task.members) + committed - 1) / committed;
                    if (result.fresh_single_pass) ++d.metrics.fresh_single_pass_batches;
                }
            }
            else
            {
                accept(d, task.finished);
                if (task.work == Work::recover) { d.needs_resume = false; ++d.metrics.batches; }
                if (task.work == Work::recover_finish) d.needs_finish = false;
                if (task.work == Work::fallback_input)
                {
                    fixed -= d.stream_input_bytes; d.lease_bytes -= d.stream_input_bytes;
                    d.engine_bytes -= d.stream_input_bytes; d.stream_input_bytes = 0;
                }
                if (task.work == Work::finish)
                {
                    if (d.metrics.stop_reason.empty()) d.metrics.stop_reason = d.maximum == 65535 ? "format-limit" : "member-limit";
                    if (!args.quiet) std::cerr << d.path << ": finished " << d.metrics.members << " members, " << d.metrics.final_bytes << " bytes\n";
                    release(d);
                    report_output(args, d.path, uint16_t(d.metrics.members), d.metrics.final_bytes);
                    stream_cohort.complete(d.owner, uint16_t(d.metrics.members), d.metrics.final_bytes,
                        execution_now(), active, d.metrics.stream_input_batches != 0);
                    if (!advance(d)) { d.complete = true; ++complete; }
                }
            }
            d.pending.reset();
        }
        std::vector<group_grow::ExecutionReady> generation_ready;
        generation_ready.reserve(destinations.size());
        for (size_t step = 0; step < destinations.size(); ++step)
        {
            size_t const index = (next + step) % destinations.size(); auto& d = *destinations[index];
            if (d.complete || d.ready.valid() || d.pending || d.needs_resume || d.needs_finish ||
                (d.engine && d.metrics.members == d.maximum)) continue;
            if (!d.engine && active >= total.execution_active) continue;
            group_grow::ExecutionReady candidate{index, d.generation_deferrals};
            if (d.engine && !d.engine->staged)
            {
                d.engine->refresh_history();
                auto const& observed = d.engine->history;
                auto const forecast = d.scratch ? group_grow::construction_external_forecast(d.requested.params.get_k(), d.engine->config,
                    uint16_t(d.metrics.members), std::max(uint16_t(1), d.batch_cap), d.maximum,
                    group_grow::construction_shape(observed.shapes), group_grow::construction_shape(observed.execution_contributions),
                    d.scratch->fan_in, d.scratch->packing_inputs ? d.scratch->packing_inputs : d.scratch->fan_in, d.execution_runs) :
                    group_grow::construction_forecast(d.requested.params.get_k(), d.engine->config, uint16_t(d.metrics.members),
                        d.metrics.final_bytes, std::max(uint16_t(1), d.batch_cap), d.maximum,
                        group_grow::construction_shape(observed.shapes));
                auto const work = group_grow::construction_execution_work(forecast, observed,
                    backend->generation_history.member_ns() * (forecast.members > d.metrics.members ? forecast.members - d.metrics.members : 0),
                    d.engine->is_staging(), bool(d.scratch), args.io_overlap);
                candidate.known = work.known && !d.stream_input_bytes && (!d.scratch || d.scratch->scratch.size() == 1);
                candidate.remaining = work.cpu / std::max(1u, d.engine->config.workers) + work.serial +
                    work.scratch_read + work.scratch_write + work.destination;
            }
            generation_ready.push_back(candidate);
        }
        size_t const generation_first = !prefetched.members.empty() ? prefetched.owner :
            generation_ready.empty() ? next : group_grow::execution_generation_order(generation_ready);
        for (size_t step = 0; step < destinations.size(); ++step)
        {
            size_t const index = (generation_first + step) % destinations.size(); auto& d = *destinations[index];
            if (d.complete || d.ready.valid()) continue;
            if (!activate(d)) { progress = prefetch(d) || progress; continue; }
            if (writers[d.writer] && !writers[d.writer]->has_room())
            { blocked(ControllerWait::writer_queue); continue; }
            if (d.pending) { progress = submit(d) || progress; continue; }
            if (d.needs_finish || (!d.needs_resume && (d.metrics.members == d.maximum || !d.metrics.stop_reason.empty())))
            {
                d.pending = std::make_unique<Task>(); d.pending->work = d.needs_finish ? Work::recover_finish : Work::finish;
                progress = submit(d) || progress; continue;
            }
            uint16_t const first = d.needs_resume ? d.saved.old_members : uint16_t(d.metrics.members);
            size_t wanted = d.needs_resume ? d.recover_count : size_t(std::min({args.batch,
                uint64_t(d.batch_cap), d.maximum - d.metrics.members}));
            // A rejected batch can retain a borrowed output while waiting
            // for resubmission, after its previous future has been consumed.
            if (wanted && std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
                { return other->pending && other->pending->lease.active(); }))
            { blocked(ControllerWait::member_lease); continue; }
            if (wanted && (!args.generation_overlap || (backend->mode == "gpu" && !backend->generation_overlap)) &&
                std::any_of(destinations.begin(), destinations.end(),
                [](auto const& other) { return other->ready.valid(); }))
            { blocked(ControllerWait::generation_order); continue; }
            Producer producer{d.requested, nullptr, *backend, args, args.max_ram, fixed, fixed};
            uint64_t const held = retained();
            bool const has_prefetched = !prefetched.members.empty() && prefetched.owner == d.owner;
            uint64_t const incoming_retained = has_prefetched ? prefetched.resident_bytes() : 0;
            uint64_t const incoming_metadata = has_prefetched ? prefetched.metadata_bytes : 0;
            bool select_merged = false;
            if (!d.needs_resume)
                d.engine->refresh_history();
            if (!d.needs_resume)
                if (auto selected = automatic_selection(*d.engine, producer, uint16_t(d.maximum), d.batch_cap,
                        d.engine_bytes, held - incoming_retained, d.observed_member_storage, false,
                        incoming_retained, incoming_metadata))
                {
                    d.lease_bytes = sum(d.lease_bytes - d.engine_bytes, selected->engine_bytes);
                    fixed = producer.fixed = selected->fixed; d.engine_bytes = selected->engine_bytes;
                    d.engine->retune(selected->config); d.engine->selected_batch = selected->batch;
                    wanted = std::min(wanted, size_t(selected->batch));
                    total.engine_reserved_bytes = std::max(total.engine_reserved_bytes, fixed);
                    select_merged = selected->merged;
                }
            bool const streamed = d.stream_input_candidate && d.stream_input_bytes && d.engine->external &&
                !d.needs_resume && !has_prefetched && !select_merged && args.member_storage == "compact";
            if (streamed) wanted = std::min(wanted, size_t(d.stream_input_batch));
            bool borrowed = !streamed && wanted && producer.can_borrow(held);
            if (wanted && !has_prefetched && !(streamed ? producer.fits_input(first, held) : producer.fits(first, held, borrowed)))
            {
                if (borrowed || !producer.can_borrow(held) || !producer.fits(first, held, true))
                { blocked(ControllerWait::producer_memory); continue; }
                borrowed = true;
            }
            if (!streamed && !d.needs_resume && wanted && !has_prefetched)
            {
                uint64_t const generation = producer.next_peak(first, borrowed);
                uint64_t const room = args.max_ram - fixed - std::max(generation, backend->retained());
                uint64_t const raw = fragment_bound(d.requested.params);
                uint64_t const member = args.member_storage == "compact" ?
                    group_grow::Member::compact_bytes(d.requested.params.get_k(), size_t(raw / sizeof(uint64_t))) : raw;
                // Reserve generation headroom before assigning a share of the
                // remaining RAM to each possible running/waiting output batch.
                size_t const slots = std::max(size_t(1), active);
                wanted = std::min(wanted, size_t(std::max<uint64_t>(1, room / slots / member)));
            }
            d.pending = std::make_unique<Task>(); auto& task = *d.pending;
            task.work = d.needs_resume ? Work::recover : Work::append;
            task.select_merged = select_merged;
            if (has_prefetched)
            {
                require(first == 0 && prefetched.identity, "next destination batch lost its initial index");
                same_identity(d.requested, *prefetched.identity);
                fixed -= prefetched.metadata_bytes;
                prefetched.take(task.members, task.hash, wanted, d.batch_cap);
                producer.fixed = fixed;
            }
            uint64_t const other_held = retained() - resident(task.members);
            bool const writing = std::any_of(writers.begin(), writers.end(), [](auto const& writer) { return writer && writer->busy(); });
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            if (backend->mode == "pool" && d.engine->external && !d.needs_resume && !has_prefetched && !select_merged && wanted)
            {
                uint16_t count = uint16_t(std::min(uint64_t(wanted), uint64_t(65535)));
                for (; count; --count)
                {
                    uint64_t const bytes = producer.contribution_bytes(d.engine->config, count);
                    uint64_t const planned = sum(sum(sum(fixed, other_held), backend->retained()), bytes);
                    uint64_t const available = available_host_bytes();
                    if (planned > args.max_ram || (available && bytes > available)) continue;
                    auto io = scratch_pool ? scratch_pool->try_io(d.scratch_owner) :
                        std::optional<group_grow::ScratchPool::IoLease>{};
                    if (scratch_pool && !io) break;
                    task.contribution_input = d.engine->external->prepare_contributions(count,
                        fragment_bound(d.requested.params) / sizeof(uint64_t));
                    if (!task.contribution_input) continue;
                    task.contribution_bytes = bytes;
                    // Provisioning owns its short storage lease. Each later
                    // bounded receive operation reacquires it; network waits
                    // must leave the physical drive available to peers.
                    io.reset();
                    auto* const pending = &task;
                    uint64_t const reserved = fixed;
                    task.generating = std::async(std::launch::async, [&, pending, first, other_held, bytes, reserved]
                    {
                        Producer producer{d.requested, nullptr, *backend, args, args.max_ram, reserved, reserved};
                        producer.defer_pool_history = true;
                        std::function<std::optional<group_grow::ScratchPool::IoLease>()> acquire_io;
                        if (scratch_pool) acquire_io = [&] { return scratch_pool->try_io(d.scratch_owner); };
                        auto const generated = producer.fill_contributions(*pending->contribution_input, first,
                            other_held, bytes, file_io::path_text(d.path) + ": generating compressed pool member ",
                            &pending->cancel_generation, acquire_io);
                        wake.notify_one();
                        return Task::Generated{generated, producer.peak, producer.generation_peak,
                            producer.pool_elapsed_ns, producer.fill_stop};
                    });
                    total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, planned);
                    if (writing) ++total.drive_overlap_batches;
                    break;
                }
                if (task.contribution_input)
                { next = (index + 1) % destinations.size(); progress = true; continue; }
            }
            if (backend->mode == "pool" && wanted && task.members.empty())
            {
                // Reserve all results and transport state before starting the
                // receiver. The controller reads only this fixed reservation
                // while its producer owns and mutates the actual member vector.
                auto fits_receive = [&](size_t count)
                { return sum(sum(sum(fixed, other_held), backend->retained()), producer.pool_receive_bytes(first, count)) <= args.max_ram; };
                while (wanted && !fits_receive(wanted) && !d.needs_resume) --wanted;
                if (!wanted || !fits_receive(wanted))
                { d.pending.reset(); blocked(ControllerWait::producer_memory); continue; }
                task.members.reserve(wanted);
                task.receiving_bytes = producer.pool_receive_bytes(first, wanted);
                uint64_t const reserved = fixed;
                auto* const pending = &task;
                task.generating = std::async(std::launch::async, [&, pending, first, wanted, other_held, reserved]
                {
                    Producer producer{d.requested, nullptr, *backend, args, args.max_ram, reserved, reserved};
                    producer.defer_pool_history = true;
                    auto const generated = producer.fill_pool(pending->members, first, wanted, other_held,
                        file_io::path_text(d.path) + ": generating pool member ", {}, &pending->cancel_generation);
                    require(generated && (!d.needs_resume || pending->members.size() == wanted),
                        "saved destination batch exceeds pool generation admission");
                    producer.admit_engine(pending->members);
                    wake.notify_one();
                    return Task::Generated{generated, producer.peak, producer.generation_peak,
                        producer.pool_elapsed_ns, producer.fill_stop};
                });
                total.peak_modeled_bytes = std::max(total.peak_modeled_bytes,
                    sum(sum(fixed, retained()), backend->retained()));
                if (writing) ++total.drive_overlap_batches;
                next = (index + 1) % destinations.size(); progress = true; continue;
            }
#endif
            if (streamed && wanted)
            {
                // An idle engine owns its provisional reservations before any
                // member generation. Other destinations may append/pack while
                // this controller thread feeds and seals the member-major input.
                auto io = scratch_pool ? scratch_pool->try_io(d.scratch_owner) :
                    std::optional<group_grow::ScratchPool::IoLease>{};
                if (scratch_pool && !io) { d.pending.reset(); blocked(ControllerWait::storage_lease); continue; }
                task.prepared_input = d.engine->external->prepare_input(uint16_t(wanted),
                    fragment_bound(d.requested.params) / sizeof(uint64_t));
                if (task.prepared_input)
                {
                    auto const generated = producer.fill_input(*task.prepared_input, first, other_held,
                        file_io::path_text(d.path) + ": preparing bounded " + backend->mode + " member ", [&]
                        {
                            return std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
                            { return other->ready.valid() && other->ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
                        });
                    task.prepared_members = uint16_t(generated);
                    d.metrics.generated_members += generated;
                    d.metrics.stream_input_written_bytes = sum(d.metrics.stream_input_written_bytes, task.prepared_input->written_bytes());
                    task.fill_stop = producer.fill_stop;
                    total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, producer.peak);
                    total.generation_overlap_members = sum(total.generation_overlap_members, producer.overlap_members);
                    total.record_generation(backend->mode, producer.generation_peak);
                    if (writing) ++total.drive_overlap_batches;
                    for (auto const& candidate : generation_ready)
                        if (candidate.index != index) ++destinations[candidate.index]->generation_deferrals;
                    d.generation_deferrals = 0;
                    io.reset(); (void)submit(d); next = (index + 1) % destinations.size(); progress = true; break;
                }
                // Conservative preflight can decline a real fitting batch.
                // Keep the ordinary path and its exact scratch/capacity retry.
                ++d.metrics.stream_input_preflight_rejections;
                d.stream_input_candidate = false;
                fixed -= d.stream_input_bytes; d.lease_bytes -= d.stream_input_bytes;
                d.engine_bytes -= d.stream_input_bytes; d.stream_input_bytes = 0; producer.fixed = fixed;
                if (!producer.fits(first, held))
                { d.pending.reset(); blocked(ControllerWait::producer_memory); continue; }
            }
            if (args.member_storage == "compact" && args.pool_broker.empty() && !d.engine->is_staging() && !task.select_merged &&
                !task.hash && task.members.empty())
                task.hash = std::make_unique<group_grow::MemberBatchHash>(d.batch_cap);
            task.members.reserve(wanted);
            require(task.members.capacity() <= d.batch_cap, "destination batch exceeds admitted metadata capacity");
            d.metrics.generated_members += producer.fill(task.members, first, wanted, other_held, task.hash.get(), &task.lease,
                file_io::path_text(d.path) + ": generating " + backend->mode + " member ", [&]
                {
                    // A completed drive can release buffers and start its next
                    // phase between members. Recovery keeps the complete saved
                    // batch required by its existing transaction protocol.
                    return !d.needs_resume && !task.members.empty() &&
                        std::any_of(destinations.begin(), destinations.end(), [](auto const& other)
                        {
                            return other->ready.valid() && other->ready.wait_for(std::chrono::seconds(0)) ==
                                std::future_status::ready;
                        });
                });
            task.fill_stop = wanted ? producer.fill_stop : Producer::FillStop::requested;
            require(!wanted || !task.members.empty(), "destination producer returned an empty batch");
            require(!d.needs_resume || task.members.size() == wanted, "saved destination batch exceeds --max-ram");
            producer.admit_engine(task.members);
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, producer.peak);
            total.peak_modeled_bytes = std::max(total.peak_modeled_bytes, sum(sum(fixed, retained()), backend->retained()));
            require(total.peak_modeled_bytes <= args.max_ram, "queued destination buffers exceed --max-ram");
            total.generation_overlap_members = sum(total.generation_overlap_members, producer.overlap_members);
            total.record_generation(backend->mode, producer.generation_peak);
            if (writing && wanted) ++total.drive_overlap_batches;
            for (auto const& candidate : generation_ready)
                if (candidate.index != index) ++destinations[candidate.index]->generation_deferrals;
            d.generation_deferrals = 0;
            (void)submit(d); next = (index + 1) % destinations.size(); progress = true; break;
        }
        if (!progress)
        {
            bool pending = std::any_of(destinations.begin(), destinations.end(), [](auto const& d)
            {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
                if (d->pending && d->pending->generating.valid()) return true;
#endif
                return d->ready.valid();
            });
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            pending = pending || pool_prefetch.ready.valid();
#endif
            if (!pending && stream_cohort.calibrating()) { stream_cohort.cancel(); continue; }
            if (!pending && !prefetched.members.empty())
            {
                // Actual scratch grants or live host headroom can change after
                // prefetch. Drop speculative work and retry ordinary admission.
                fixed -= prefetched.metadata_bytes; prefetched.clear(); continue;
            }
            require(pending, farming_resource_error.empty() ?
                "member producer and destination workspaces exceed --max-ram or available host memory" : farming_resource_error);
            bool completed = false;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
            completed = pool_prefetch.ready.valid() &&
                pool_prefetch.ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
#endif
            for (auto const& d : destinations)
            {
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
                if (d->pending && d->pending->generating.valid() &&
                    d->pending->generating.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                { completed = true; break; }
#endif
                if (d->ready.valid())
                {
                    // A callback may have yielded for a different drive that
                    // completed after the earlier scan. Service it immediately;
                    // its old generation stop cannot explain an idle wait.
                    if (d->ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                    { completed = true; break; }
                    wait_reasons |= d->successor_wait_reason | d->prefetch_wait_reason;
                }
                if (d->pending) wait_reasons |= fill_wait_reason(d->pending->fill_stop);
            }
            if (completed) continue;
            auto const begin = std::chrono::steady_clock::now();
            wait_reasons |= wait_bit(ControllerWait::consumer);
            ControllerTiming timing("wait", 0, wait_reasons);
            std::unique_lock lock(wake_mutex); wake.wait_for(lock, std::chrono::milliseconds(50));
            uint64_t const elapsed = group_grow::elapsed_ns(begin);
            total.queue_wait_ns = sum(total.queue_wait_ns, elapsed);
            total.controller_waits.add(wait_reasons, elapsed);
        }
    }
    for (auto const& destination : destinations) accumulate(destination->metrics, destination->mode);
    if (total.stop_reason.empty()) total.stop_reason = farm ? farm->stop_reason(args) : args.max_members == 65535 ? "format-limit" : "member-limit";
    farm_peak();
    uint64_t output_count = 0, accounted = farm ? farm->manifest_charge : 0, last_members = 0;
    if (farm) for (auto const& lane : farm->lanes)
    {
        output_count = sum(output_count, lane.outputs.size());
        for (auto const& output : lane.outputs) accounted = sum(accounted, output.charge());
        if (!lane.outputs.empty()) last_members = lane.outputs.back().members;
    }
    if (farm) farm->assessment_metrics(total);
    total.stream_input_trial = stream_cohort.status();
    print_metrics(args, total, *backend, output_count, last_members, accounted); return total.members ? 0 : 2;
}
#ifndef GROW_TEST_CONSTRUCTION
int run(Arguments const& supplied)
{
    Arguments args = supplied;
    require(std::endian::native == std::endian::little, "upstream format requires little endian");
    check_cancel(args);
    if (args.directory_farm && args.plan_only && args.target.empty() && args.manifest.empty() && args.relocations.empty())
        return print_destination_candidates(args);
    std::optional<FarmingJob> farming_job;
    if (args.directory_farm) farming_job = farming_arguments(args);
    bool const key_options = args.farmer || args.pool_pk || args.pool_ph || args.pool_address || args.seed;
    require(!key_options || (!args.group && !args.memo && args.fixtures.empty()),
            "key-derived identities cannot be combined with --group-id, --memo or fixtures");
    std::optional<Keys> keys;
    if (!args.plan_only && (key_options || args.group_members || args.directory_farm || !args.destinations.empty())) keys.emplace(args);
    if (args.directory_farm)
    {
        auto farm = farming_run(args, keys ? &*keys : nullptr, *farming_job);
        print_farming_plan(*farm, args);
        require(farm->feasible(), "farming profile or requested extent is infeasible: " + farm->infeasibility_reason());
        bool const missing = std::find(farm->available.begin(), farm->available.end(), uint8_t(0)) != farm->available.end();
        if (args.plan_only) return missing ? 2 : 0;
        if (!args.command.empty() && !args.quiet)
            std::cerr << '[' << args.command << "] saved job: " << file_io::path_text(farm->manifest)
                << " (resume with fill -r and any member directory)\n";
        for (size_t i = 0; i < farm->available.size(); ++i) if (!farm->available[i])
            std::cerr << "[fill] pending unavailable destination: " << farm->state.directories[i].directory << '\n';
        farm->select_available();
        int const result = multiple_destinations(args, &*keys, farm.get());
        return !result && missing ? 4 : result;
    }
    if (!args.destinations.empty()) return multiple_destinations(args, &*keys);
    if (args.group_members) return rollover(args, *keys);
    require(fs::is_directory(file_io::native_path(args.target.parent_path())), "destination directory must exist");
    bool const exists = fs::exists(file_io::native_path(args.target));
    require(!exists || args.resume, "destination exists; use --resume to continue it");
    require(exists || !args.resume, "--resume destination does not exist");
    std::optional<Fixtures> fixture;
    if (!args.fixtures.empty()) fixture.emplace(fixtures(args.fixtures, args.max_ram));
    std::optional<Identity> initial;
    if (fixture) initial = fixture->identity;
    else if (keys) initial = keys->derive(0);
    else if (!exists || (args.group && args.memo))
    {
        require(args.group && args.memo, "new destination requires --group-id and --memo");
        initial = Identity{PlotGroupParams(PlotGroupId(*args.group), uint8_t(args.k.value_or(28)),
                           uint8_t(args.strength.value_or(2)), uint8_t(args.meta.value_or(0))), hex(*args.memo)};
    }
    if (initial) validate(*initial);
    Output auxiliary;
    auto recovery = output_auxiliary(args.target, auxiliary, args.max_ram - (fixture ? fixture->bytes : 0));
    uint64_t const filesystem_limit = filesystem_capacity(args.target, args.headroom, auxiliary.auxiliary_allocated);
    uint64_t const file_limit = group_grow::maximum_file_bytes(file_io::path_text(args.target.parent_path()));
    require(!args.capacity || args.capacity <= file_limit,
            "requested single-file capacity exceeds the filesystem or process file-size limit (" +
            std::to_string(file_limit) + " bytes); use a smaller capacity or bounded group rollover");
    uint64_t const capacity = std::min(args.capacity ? std::min(args.capacity, filesystem_limit) : filesystem_limit, file_limit);
    uint64_t const fixture_members = fixture ? fixture->paths.size() : 65535;
    uint16_t const maximum = uint16_t(std::min(args.max_members, fixture_members));
    std::unique_ptr<Backend> backend;
    auto const metrics = run_group(args, args.target, fixture ? &*fixture : nullptr, initial,
                                   maximum, capacity, 0, backend, nullptr, std::move(recovery), auxiliary);
    report_output(args, args.target, uint16_t(metrics.members), metrics.final_bytes);
    print_metrics(args, metrics, *backend);
    return metrics.members ? 0 : 2;
}
#endif
} // namespace

#ifndef GROW_TEST_CONSTRUCTION
int pos2gpu::run_group(GroupOptions const& options)
{
    auto args = options;
#ifdef XCHPLOT2_GROUP_COMMANDS
    if (args.automatic_compute && !args.plan_only) args.compute_session = std::make_shared<ComputeSession>(args);
#endif
    try { return run(args); }
    catch (Interrupted const& error)
    { std::cerr << error.what() << '\n'; return 4; }
}
#endif

#ifdef XCHPLOT2_GROUP_COMMANDS
pos2gpu::BatchResult pos2gpu::run_group_batch(std::vector<BatchEntry> const& entries,
    BatchOptions const& options, GroupOptions args)
{
    BatchResult result;
    if (entries.empty()) return result;
    std::vector<NamedGroup> named; named.reserve(entries.size());
    std::map<fs::path, BatchEntry const*> outputs;
    for (auto const& entry : entries)
    {
        validate_batch_entry(entry);
        require(entry.grouped && !entry.raw, "grouped and single-member/raw jobs must use separate manifests");
        fs::path const path = fs::absolute(file_io::native_path(entry.out_dir) / file_io::native_path(entry.out_name)).lexically_normal();
        require(outputs.emplace(path, &entry).second, "duplicate grouped output path");
        if (named.empty()) args.target = path;
        else args.destinations.push_back(path);
        Identity identity{PlotGroupParams(PlotGroupId(entry.group_id), uint8_t(entry.k),
            uint8_t(entry.strength), uint8_t(entry.meta_group)), entry.memo};
        validate(identity);
        named.push_back({std::move(identity), uint16_t(entry.group_members)});
    }
    auto const start = std::chrono::steady_clock::now();
    args.on_output = [&](fs::path const& path, uint16_t members, uint64_t bytes, bool existing)
    {
        auto const& entry = *outputs.at(path);
        if (members != entry.group_members)
        {
            ++result.plots_failed;
            std::cerr << path << ": capacity stopped at " << members << " of " << entry.group_members << " requested members\n";
            return;
        }
        if (existing) ++result.plots_skipped;
        else { ++result.plots_written; result.bytes_written = sum(result.bytes_written, bytes); }
        if (options.on_plot_ready) options.on_plot_ready(entry);
    };
    if (args.automatic_compute) args.compute_session = std::make_shared<ComputeSession>(args);
    try { multiple_destinations(args, nullptr, nullptr, named); }
    catch (Interrupted const& error)
    { std::cerr << error.what() << '\n'; }

    result.total_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
#endif
