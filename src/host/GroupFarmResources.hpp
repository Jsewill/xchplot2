// Conditional mixed-farm resource envelopes. Foreign services remain outside
// our admission gate; declarations are reservations, not hardware qualification.
#pragma once
#include "host/GroupHarvesterProfile.hpp"
#include <set>
#ifdef _WIN32
#include <psapi.h>
#include <winternl.h>
#endif
#ifdef __linux__
#include <dirent.h>
#include <time.h>
#endif

namespace group_grow::farm
{
using Clock = std::chrono::steady_clock;
inline constexpr uint64_t resource_schema = 1;
inline constexpr size_t max_resources = 128, max_workloads = 256, max_stages = 4096;
inline constexpr uint64_t profile_byte_limit = 4 * 1024 * 1024;
inline constexpr uint64_t compute_generate_cpu = uint64_t(1) << 1, compute_generate_gpu = uint64_t(1) << 2,
    compute_encode = uint64_t(1) << 3;
enum class ResourceKind { cpu, host_memory, gpu_compute, gpu_memory, storage, controller, network, solver };
enum class CostKind { unknown, reservation, serialized };
struct Resource
{
    std::string id, host;
    std::vector<std::string> aliases;
    ResourceKind kind = ResourceKind::cpu;
    // Bytes for host/GPU memory; physical parallel lanes otherwise. CPU lanes
    // are an explicitly offered logical-core budget. GPU defaults to one lane.
    uint64_t capacity = 0;
    bool capacity_known = false, conditional_shared = false, telemetry_required = false;
};
struct Use
{
    std::string resource;
    // Resident memory and opaque service reservations survive plot retirement.
    // They may be removed only in a refreshed, reviewed profile after release.
    uint64_t resident_units = 0, reserved_units = 0, transient_units = 0;
    // An indivisible request's service-wall bound under required_units lanes,
    // including scheduler overhead. No ideal parallel speedup is assumed.
    uint64_t required_units = 0, quality_us = 0, proof_us = 0, blocking_us = 0;
};
struct Workload
{
    std::string id, format, decoder, decoder_version, backend, pool_difficulty;
    std::string source, captured_at;
    // Explicit, unscanned service scope. Its complete maximum inventory and
    // queued/full-proof envelope remains reserved throughout retirement.
    std::string external_service_node_id, external_service_host;
    uint64_t external_service_members = 0;
    CostKind cost = CostKind::unknown;
    bool bounded = false, scope_complete = false;
    uint64_t plot_count = 0, quality_burst = 0, proof_burst = 0;
    uint64_t future_quality_burst = 0, future_proof_burst = 0;
    uint64_t filter_bits = 0, future_filter_bits = 0, period_us = 0, deadline_us = 0;
    // Opaque reservations declare their complete response bound. Serialized
    // workloads add this non-resource tail to the conservative resource drains.
    uint64_t response_tail_us = 0;
    std::vector<Use> uses;
};
struct GatePolicy
{
    uint64_t max_age_ms = 5000, recovery_hold_ms = 1000, recovery_samples = 3;
    uint64_t max_cpu_busy_permille = 950;
};
struct Remaining { std::string workload; uint64_t count = 0; };
struct Demand
{
    std::string resource;
    uint64_t resident_units = 0, transient_units = 0, reserved_units = 0, required_units = 0;
    // Construction/verification demands cover the entire file operation,
    // including every member/window and the bounded verification sample policy.
    // Other arrays cover their declared cohort/supervision burst. These are
    // conditional service/whole carried-in blocking bounds, not batch averages
    // or the separate operation timeout.
    uint64_t work_us = 0, blocking_us = 0, period_us = 0, deadline_us = 0;
    bool bounded = false;
};
struct NativeCohort
{
    uint64_t max_members = 0;
    std::vector<Demand> new_farming, proof_reserve;
};
struct ComputeWorkerEnvelope
{
    // The reviewed local profile is authority for TLS identity to physical
    // host/device mapping, just as for remote Chia service host bindings.
    std::string tls_identity, host, gpu_identity;
    uint64_t cpu_threads = 0, slots = 0, host_bytes = 0, gpu_bytes = 0;
    uint64_t standing_host_bytes = 0, standing_gpu_bytes = 0, allowed_kinds = 0;
    // Reserve the whole globally bounded executor, including other owners and
    // idle residency. Current lease/queue occupancy never releases this scope.
    std::vector<Demand> reservation;
};
struct NativeEnvelope
{
    bool bounded = false, scope_complete = false, covers_future_filter = false, phase_totals_include_controller = false;
    uint64_t k = 0, strength = 0;
    std::string host, reader, backend, engine;
    // Service roles without an override are declared colocated on host.
    // These are physical resource hosts, never names inferred from peer IPs.
    std::map<std::string, std::string> service_hosts;
    std::string gpu_compute_resource, gpu_memory_resource;
    uint64_t max_ram_bytes = 0, max_cpu_threads = 0, max_gpu_bytes = 0, max_batch_members = 0;
    // Reviewed context/runtime/cache residency, additional to the construction
    // allocation grant above. It persists until this GPU session ends; the
    // free-memory allocation probe and its safety margin do not bound it.
    uint64_t standing_gpu_bytes = 0;
    // Construction/verification demands bound one whole file up to this
    // membership. max_batch_members only caps an individual execution batch.
    uint64_t max_group_members = 0;
    uint64_t max_construction_us = 0, max_verification_us = 0;
    uint64_t verification_cpu_threads = 0;
    // Minimum external-pool difficulty covered by the full-solver demand in
    // every cohort. Zero means solo/OG; a descriptive workload label is not a
    // quantitative authorization for external-pool arrivals.
    uint64_t min_pool_difficulty = 0, pool_protocol_version = 0;
    std::string compute_owner_identity, compute_broker_identity, compute_broker_host;
    std::vector<ComputeWorkerEnvelope> compute_workers;
    std::vector<Demand> compute_services;
    std::vector<std::string> output_resources, scratch_resources, source_resources;
    std::vector<Demand> construction, verification, supervision;
    std::vector<NativeCohort> cohorts;
};
struct NativeRequest
{
    uint64_t k = 0, strength = 0;
    std::string host, reader, backend, engine;
    uint64_t ram_bytes = 0, cpu_threads = 0, gpu_bytes = 0, batch_members = 0, group_members = 0;
    uint64_t standing_gpu_bytes = 0;
    uint64_t cpu_affinity_lanes = 0;
    uint64_t construction_us = 0, verification_us = 0;
    uint64_t verification_ram_bytes = 0, verification_cpu_threads = 0, verification_gpu_bytes = 0;
    bool gpu_binding_verified = false;
    std::string gpu_identity;
    uint64_t controller_ram_bytes = 0, adapter_ram_bytes = 0, adapter_cpu_threads = 0, adapter_timeout_us = 0;
    uint64_t generation_peak_bytes = 0, compute_session_ram_bytes = 0, compute_codec_ram_bytes = 0;
    bool compute_enabled = false;
    std::vector<std::string> output_resources, scratch_resources, source_resources;
};
struct Profile
{
    std::string source, captured_at, producer;
    std::vector<Resource> resources;
    std::vector<Workload> workloads;
    GatePolicy gate;
    NativeEnvelope native;
};
// Preparation is guidance beside the policy, never admission authority. Keep
// observations separate so an OS snapshot or reader sample cannot become an
// operator-declared capacity, queue bound or full-proof reservation.
struct ResourcePreparationFact
{
    std::string path, value, basis, source, scope;
};
struct ResourcePreparationNeed
{
    std::string group, path, action;
};
struct ResourcePreparation
{
    std::vector<ResourcePreparationFact> facts;
    std::vector<ResourcePreparationNeed> missing;
};
inline constexpr size_t max_preparation_entries = 4096;
struct Stage
{
    std::string name;
    bool future_filter = false, scope_complete = false;
    std::vector<Remaining> remaining;
    // Already-assessed native v2 quality demand is independent of its separately
    // bounded full-proof solver reserve. Reader calibration does not supply it.
    std::vector<Demand> new_farming, proof_reserve, construction, verification, supervision, background;
};
struct Margin
{
    std::string resource;
    ResourceKind kind = ResourceKind::cpu;
    uint64_t capacity = 0, resident_units = 0, transient_units = 0, reserved_units = 0;
    uint64_t required_units = 0, request_units = 0, occupied_units = 0, remaining_units = 0, serialized_work_us = 0, blocking_us = 0;
    uint64_t drain_us = 0, period_us = 0, deadline_us = 0;
    std::map<std::string, std::string> worst_stages;
};
struct Assessment
{
    bool feasible = true, conditional_shared_gpu = false, conditional_shared_service = false, per_dimension_worst = false;
    std::vector<std::string> reasons;
    std::vector<Margin> margins;
};
namespace resource_detail
{
inline void require(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
inline uint64_t add(uint64_t a, uint64_t b)
{ require(b <= UINT64_MAX - a, "mixed farm resource sum overflow"); return a + b; }
inline uint64_t multiply(uint64_t a, uint64_t b)
{ require(!a || b <= UINT64_MAX / a, "mixed farm resource product overflow"); return a * b; }
inline bool memory(ResourceKind kind)
{ return kind == ResourceKind::host_memory || kind == ResourceKind::gpu_memory; }
inline char const* name(ResourceKind kind)
{
    switch (kind)
    {
    case ResourceKind::cpu: return "cpu";
    case ResourceKind::host_memory: return "host_memory";
    case ResourceKind::gpu_compute: return "gpu_compute";
    case ResourceKind::gpu_memory: return "gpu_memory";
    case ResourceKind::storage: return "storage";
    case ResourceKind::controller: return "controller";
    case ResourceKind::network: return "network";
    case ResourceKind::solver: return "solver";
    }
    throw std::runtime_error("unsupported mixed farm resource kind");
}
inline ResourceKind kind(std::string const& value)
{
    for (auto item : {ResourceKind::cpu, ResourceKind::host_memory, ResourceKind::gpu_compute,
                      ResourceKind::gpu_memory, ResourceKind::storage, ResourceKind::controller,
                      ResourceKind::network, ResourceKind::solver})
        if (value == name(item)) return item;
    throw std::runtime_error("unsupported mixed farm resource kind");
}
inline void text(std::string const& value, bool nonempty = false)
{ require(value.size() <= 1024 && (!nonempty || !value.empty()), "invalid mixed farm resource text"); }
inline uint64_t minimum(uint64_t a, uint64_t b) { return !a ? b : !b ? a : std::min(a, b); }
struct Graph
{
    std::vector<Resource> resources;
    std::map<std::string, size_t> names;
    explicit Graph(Profile const& profile)
    {
        require(!profile.resources.empty() && profile.resources.size() <= max_resources, "invalid mixed farm resource count");
        std::vector<size_t> parents(profile.resources.size());
        for (size_t i = 0; i < parents.size(); ++i) parents[i] = i;
        auto root = [&](size_t i) { while (parents[i] != i) i = parents[i]; return i; };
        std::map<std::string, size_t> seen;
        std::map<std::pair<ResourceKind, std::string>, size_t> hosts;
        std::set<std::string> ids;
        for (size_t i = 0; i < profile.resources.size(); ++i)
        {
            auto const& r = profile.resources[i]; text(r.id, true); text(r.host, true);
            (void)name(r.kind);
            require(ids.insert(r.id).second && r.aliases.size() <= 32, "duplicate mixed farm resource ID or too many aliases");
            require(r.capacity || !r.capacity_known, "known mixed farm resource has zero capacity");
            // The public schema does not describe disjoint CPU sets or NUMA
            // memory partitions. Different labels on one host cannot multiply
            // its offered CPU/RAM budget.
            if (r.kind == ResourceKind::cpu || r.kind == ResourceKind::host_memory)
            {
                auto [at, fresh] = hosts.emplace(std::pair{r.kind, r.host}, i);
                if (!fresh) parents[root(i)] = root(at->second);
            }
            auto link = [&](std::string const& key)
            {
                text(key, true); auto [at, fresh] = seen.emplace(key, i);
                if (!fresh)
                {
                    auto const& other = profile.resources[at->second];
                    require(other.kind == r.kind && other.host == r.host, "mixed farm alias crosses physical host or resource kind");
                    parents[root(i)] = root(at->second);
                }
            };
            link(r.id); for (auto const& alias : r.aliases) link(alias);
        }
        std::map<size_t, size_t> canonical;
        for (size_t i = 0; i < profile.resources.size(); ++i)
        {
            auto [at, fresh] = canonical.emplace(root(i), resources.size());
            if (fresh) { resources.push_back(profile.resources[i]); resources.back().aliases.clear(); }
            else
            {
                auto& r = resources[at->second]; auto const& other = profile.resources[i];
                r.capacity = std::min(r.capacity, other.capacity); r.capacity_known = r.capacity_known && other.capacity_known;
                r.conditional_shared = r.conditional_shared && other.conditional_shared;
                r.telemetry_required = r.telemetry_required || other.telemetry_required;
            }
            names.emplace(profile.resources[i].id, at->second);
            for (auto const& alias : profile.resources[i].aliases) names.emplace(alias, at->second);
        }
    }
    size_t find(std::string const& id) const
    { auto at = names.find(id); require(at != names.end(), "mixed farm demand names an unknown physical resource"); return at->second; }
};
inline void validate(Profile const& profile)
{
    text(profile.source, true); text(profile.captured_at); text(profile.producer, true);
    Graph graph(profile);
    require(profile.workloads.size() <= max_workloads, "too many mixed farm workloads");
    require(profile.gate.max_age_ms && profile.gate.max_age_ms <= 3600000 &&
            profile.gate.recovery_hold_ms <= 3600000 && profile.gate.recovery_samples &&
            profile.gate.recovery_samples <= 64 && profile.gate.max_cpu_busy_permille <= 1000,
            "invalid mixed farm telemetry policy");
    std::set<std::string> ids, external_ids;
    for (auto const& w : profile.workloads)
    {
        text(w.id, true); require(ids.insert(w.id).second, "duplicate mixed farm workload ID");
        for (auto const* s : {&w.format, &w.decoder, &w.decoder_version, &w.backend, &w.pool_difficulty, &w.source, &w.captured_at}) text(*s);
        require(w.filter_bits <= 255 && w.future_filter_bits <= 255 && w.uses.size() <= max_resources,
                "invalid mixed farm workload filter or resource count");
        require(w.cost == CostKind::unknown || w.cost == CostKind::reservation || w.cost == CostKind::serialized,
                "unsupported mixed farm cost kind");
        bool const external = !w.external_service_node_id.empty();
        require(external == !w.external_service_host.empty() && external == bool(w.external_service_members),
            "external service workload requires its peer node identity, physical host and maximum member count");
        if (external)
        {
            require(external_ids.insert(w.external_service_node_id).second, "external service node identity has duplicate workload scopes");
            require(w.plot_count && w.external_service_members >= w.plot_count && w.external_service_node_id.size() == 64 &&
                std::all_of(w.external_service_node_id.begin(), w.external_service_node_id.end(),
                    [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
                "invalid external service identity or maximum population");
            text(w.external_service_host, true);
        }
        std::set<size_t> used;
        for (auto const& u : w.uses)
        {
            size_t const at = graph.find(u.resource);
            require(used.insert(at).second, "mixed farm workload repeats a physical resource alias");
            if (memory(graph.resources[at].kind))
                require(!u.reserved_units && !u.required_units && !u.quality_us && !u.proof_us && !u.blocking_us,
                        "memory resource has service-time units");
            else require(!u.resident_units && !u.transient_units, "service resource has memory-byte units");
            if (w.cost == CostKind::reservation)
                require(!u.required_units && !u.quality_us && !u.proof_us && !u.blocking_us,
                        "opaque reservation also charges modeled work");
            else if (w.cost == CostKind::serialized)
                require(!u.reserved_units, "modeled service also reserves opaque lanes");
        }
    }
    auto const& n = profile.native;
    require((!n.min_pool_difficulty && !n.pool_protocol_version) ||
        (n.min_pool_difficulty && (n.pool_protocol_version == 1 || n.pool_protocol_version == 2)),
        "native pool demand requires both a reviewed difficulty and supported pooling protocol");
    text(n.host); text(n.reader); text(n.backend); text(n.engine); text(n.gpu_compute_resource); text(n.gpu_memory_resource);
    require(n.service_hosts.size() <= 4, "too many native service host bindings");
    for (auto const& [role, host] : n.service_hosts)
    {
        require(role == "harvester" || role == "farmer" || role == "full_node" || role == "solver", "unknown native service host role");
        text(host, true);
    }
    require(n.max_group_members <= UINT16_MAX, "native whole-file membership bound exceeds the plot format");
    require(n.cohorts.size() <= 256 && n.output_resources.size() <= max_resources && n.scratch_resources.size() <= max_resources && n.source_resources.size() <= max_resources &&
            n.construction.size() <= max_resources * 4 && n.verification.size() <= max_resources * 4 &&
            n.supervision.size() <= max_resources * 4, "native resource envelope is too large");
    uint64_t prior = 0;
    auto demands = [&](std::vector<Demand> const& values)
    {
        for (auto const& d : values)
        {
            text(d.resource, true); auto const kind = graph.resources[graph.find(d.resource)].kind;
            if (memory(kind)) require(!d.reserved_units && !d.required_units && !d.work_us && !d.blocking_us, "native memory demand has service units");
            else require(!d.resident_units && !d.transient_units, "native service demand has memory units");
            require(!(d.reserved_units && (d.required_units || d.work_us)), "native demand double charges reservation and modeled service");
        }
    };
    for (auto const* mappings : {&n.output_resources, &n.scratch_resources, &n.source_resources}) for (auto const& id : *mappings) { text(id, true); (void)graph.find(id); }
    demands(n.construction); demands(n.verification); demands(n.supervision);
    demands(n.compute_services);
    require(n.compute_workers.size() <= max_workloads && n.compute_services.size() <= max_resources * 4,
        "native compute scope exceeds its bounded declaration size");
    text(n.compute_owner_identity); text(n.compute_broker_identity); text(n.compute_broker_host);
    std::set<std::string> compute_ids;
    for (auto const& worker : n.compute_workers)
    {
        text(worker.tls_identity, true); text(worker.host, true); text(worker.gpu_identity);
        require(compute_ids.insert(worker.tls_identity).second && worker.tls_identity.size() == 64 &&
            std::all_of(worker.tls_identity.begin(), worker.tls_identity.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
            "invalid or duplicate reviewed compute worker TLS identity");
        require(worker.cpu_threads && worker.cpu_threads <= 1024 && worker.slots && worker.slots <= 1024 && worker.host_bytes &&
            worker.standing_host_bytes && worker.standing_host_bytes < worker.host_bytes && worker.allowed_kinds &&
            !(worker.allowed_kinds & ~(compute_generate_cpu | compute_generate_gpu | compute_encode)) &&
            !worker.reservation.empty() && worker.reservation.size() <= max_resources * 4,
            "reviewed compute worker omits bounded whole-service capacity or task capabilities");
        require((worker.gpu_bytes && worker.standing_gpu_bytes && !worker.gpu_identity.empty()) ||
            (!worker.gpu_bytes && !worker.standing_gpu_bytes && worker.gpu_identity.empty()), "reviewed compute GPU requires allocation and standing physical residency");
        require(worker.gpu_identity.empty() || worker.gpu_identity.starts_with("host:" + worker.host + '/'),
            "reviewed compute physical GPU identity differs from its resource host");
        require(!(worker.allowed_kinds & compute_generate_gpu) || worker.gpu_bytes, "reviewed GPU task capability has no physical GPU service budget");
        demands(worker.reservation);
    }
    for (auto const& c : n.cohorts)
    {
        require(c.max_members > prior && c.new_farming.size() <= max_resources * 4 && c.proof_reserve.size() <= max_resources * 4,
                "native resource cohorts must be bounded and increasing");
        prior = c.max_members;
        demands(c.new_farming); demands(c.proof_reserve);
    }
}
inline void fail(Assessment& result, std::string reason)
{ result.feasible = false; if (result.reasons.size() < 1024) result.reasons.push_back(std::move(reason)); }
} // namespace resource_detail

// Reuse only while this borrowed profile is immutable and alive. Native
// resource rebinding or a refreshed declaration requires new preparation.
class AssessmentPreparation
{
    Profile const& profile_;
    resource_detail::Graph graph_;
    static resource_detail::Graph prepare(Profile const& profile)
    { resource_detail::validate(profile); return resource_detail::Graph(profile); }
public:
    explicit AssessmentPreparation(Profile const& profile) : profile_(profile), graph_(prepare(profile)) {}
    Profile const& profile() const { return profile_; }
    resource_detail::Graph const& graph() const { return graph_; }
    uint64_t memory_bytes() const
    {
        using namespace resource_detail;
        uint64_t bytes = add(sizeof(*this), multiply(graph_.resources.capacity(), sizeof(Resource)));
        for (auto const& r : graph_.resources)
            bytes = add(bytes, add(add(r.id.capacity(), r.host.capacity()), add(2, multiply(r.aliases.capacity(), sizeof(std::string)))));
        for (auto const& [name, index] : graph_.names)
            bytes = add(bytes, add(sizeof(decltype(graph_.names)::value_type) + 64, add(name.capacity(), 1)));
        return bytes;
    }
};

// Inventory coverage includes outside plots, not just replacement candidates.
// A blank/unknown assignment is unsafe even if those files will be retained.
inline void validate_inventory_coverage(Profile const& profile, std::vector<Remaining> const& counts)
{
    resource_detail::validate(profile);
    std::map<std::string, uint64_t> actual;
    for (auto const& item : counts)
    {
        resource_detail::text(item.workload, true);
        actual[item.workload] = resource_detail::add(actual[item.workload], item.count);
    }
    resource_detail::require(actual.size() <= max_workloads, "too many inventory workload assignments");
    for (auto const& w : profile.workloads)
    {
        auto at = actual.find(w.id);
        if (!w.external_service_node_id.empty())
        {
            resource_detail::require(at == actual.end(), "unscanned external service workload was assigned scanned retirement inventory");
            continue;
        }
        resource_detail::require((at == actual.end() ? 0 : at->second) == w.plot_count,
                                 "mixed farm profile does not cover the complete active inventory");
        if (at != actual.end()) actual.erase(at);
    }
    resource_detail::require(actual.empty(), "inventory workload has no mixed farm profile");
}

inline void bind_native_stage(AssessmentPreparation const& preparation, NativeRequest const& request, uint64_t ready_members,
                              Stage& stage, bool building, bool verifying)
{
    using namespace resource_detail;
    auto const& profile = preparation.profile(); auto const& graph = preparation.graph(); auto const& n = profile.native;
    require(n.bounded && n.scope_complete, "missing complete bounded native resource envelope");
    require(n.covers_future_filter && n.phase_totals_include_controller && request.controller_ram_bytes,
            "native envelope must explicitly cover future filters and phase-total controller residency");
    require(!n.output_resources.empty(), "native resource envelope omits physical output mapping");
    require(!request.host.empty() && request.host == n.host && request.k == n.k && request.strength == n.strength && request.reader == n.reader && request.backend == n.backend && request.engine == n.engine,
            "native resource envelope differs from selected plot/reader/backend/engine");
    require(request.ram_bytes && request.ram_bytes <= n.max_ram_bytes && request.cpu_threads && request.cpu_threads <= n.max_cpu_threads &&
            request.gpu_bytes <= n.max_gpu_bytes && request.batch_members && request.batch_members <= n.max_batch_members &&
            request.construction_us && request.construction_us <= n.max_construction_us && request.verification_us && request.verification_us <= n.max_verification_us,
            "selected native construction or verification exceeds reviewed resource envelope");
    require(request.group_members && request.group_members <= n.max_group_members,
            "selected whole-file membership exceeds reviewed construction and verification phase bounds");
    require(request.cpu_affinity_lanes && request.cpu_threads <= request.cpu_affinity_lanes,
            "native request has no valid inherited CPU affinity scope");
    require(request.standing_gpu_bytes == (request.backend == "gpu" && !request.compute_enabled ? n.standing_gpu_bytes : 0) &&
            (request.backend != "gpu" || request.compute_enabled || request.standing_gpu_bytes),
            "GPU migration requires a reviewed standing native GPU residency bound");
    auto mappings = [&](std::vector<std::string> const& actual, std::vector<std::string> const& saved)
    {
        require(actual.size() <= max_resources && saved.size() <= max_resources, "too many native physical mappings");
        std::set<size_t> expected, observed;
        for (auto const& id : saved) expected.insert(graph.find(id));
        for (auto const& id : actual) observed.insert(graph.find(id));
        require(expected == observed, "native physical output or scratch mapping differs from resource envelope");
    };
    mappings(request.output_resources, n.output_resources); mappings(request.scratch_resources, n.scratch_resources);
    mappings(request.source_resources, n.source_resources);
    std::optional<size_t> selected_gpu, selected_vram;
    if (request.gpu_bytes || request.verification_gpu_bytes || request.standing_gpu_bytes)
    {
        require(request.gpu_binding_verified && !request.gpu_identity.empty(), "selected GPU has no verified physical identity");
        selected_gpu = graph.find(n.gpu_compute_resource); selected_vram = graph.find(n.gpu_memory_resource);
        require(graph.resources[*selected_gpu].kind == ResourceKind::gpu_compute && graph.resources[*selected_vram].kind == ResourceKind::gpu_memory &&
                graph.resources[*selected_gpu].host == request.host && graph.resources[*selected_vram].host == request.host &&
                graph.find(request.gpu_identity + "/compute") == *selected_gpu && graph.find(request.gpu_identity + "/memory") == *selected_vram,
                "selected GPU differs from reviewed physical compute or memory resource");
    }
    stage.new_farming.clear(); stage.proof_reserve.clear(); stage.construction.clear(); stage.verification.clear(); stage.supervision.clear();
    stage.background = n.compute_services;
    for (auto const& worker : n.compute_workers)
        stage.background.insert(stage.background.end(), worker.reservation.begin(), worker.reservation.end());
    require(stage.background.size() <= max_resources * 4, "native compute background demands exceed their bounded stage size");
    // Target farming/solver services must already be live before replacement.
    // Reserve the first complete cohort even with no ready members, covering
    // standing residency and queues without guessing a smaller idle footprint.
    auto const at = std::find_if(n.cohorts.begin(), n.cohorts.end(), [&](NativeCohort const& c) { return c.max_members >= ready_members; });
    require(at != n.cohorts.end(), "native farming count exceeds reviewed cohort envelope");
    require(!at->new_farming.empty() && !at->proof_reserve.empty(), "native cohort omits qualities or full-proof solver reserve");
    stage.new_farming = at->new_farming; stage.proof_reserve = at->proof_reserve;
    if (building) { require(!n.construction.empty(), "missing native construction demands"); stage.construction = n.construction; }
    if (verifying) { require(!n.verification.empty(), "missing native verification demands"); stage.verification = n.verification; }
    if (!building && !verifying)
    {
        require(!n.supervision.empty() && request.adapter_ram_bytes && request.adapter_cpu_threads && request.adapter_timeout_us,
                "missing complete bounded controller/RPC phase envelope");
        stage.supervision = n.supervision;
    }
    auto coverage = [&](std::vector<Demand> const& values, uint64_t cpu_required, uint64_t ram_required,
                        uint64_t gpu_required, bool scratch, bool local, bool sources = false)
    {
        std::map<size_t, uint64_t> units, resident;
        std::map<std::string, std::set<ResourceKind>> hosts;
        std::set<ResourceKind> kinds; std::set<size_t> used;
        for (auto const& d : values)
        {
            require(d.bounded, "unbounded native resource demand");
            size_t const i = graph.find(d.resource); auto const kind = graph.resources[i].kind;
            uint64_t const amount = memory(kind) ? add(d.resident_units, d.transient_units) : add(d.required_units, d.reserved_units);
            require(amount > 0 && (memory(kind) || d.reserved_units || d.work_us), "zero or irrelevant native resource demand");
            used.insert(i);
            hosts[graph.resources[i].host].insert(kind);
            if (!local || graph.resources[i].host == request.host) kinds.insert(kind);
            units[i] = add(units[i], amount);
            resident[i] = add(resident[i], d.resident_units);
        }
        require(kinds.contains(ResourceKind::cpu) && kinds.contains(ResourceKind::host_memory), "native demand omits CPU or host memory");
        for (auto const& [host, resources] : hosts)
            if (resources.contains(ResourceKind::cpu) || resources.contains(ResourceKind::host_memory) ||
                resources.contains(ResourceKind::gpu_compute) || resources.contains(ResourceKind::gpu_memory) || resources.contains(ResourceKind::solver))
                require(resources.contains(ResourceKind::cpu) && resources.contains(ResourceKind::host_memory),
                    "native service demand omits CPU or host memory on its physical execution host");
        uint64_t cpu = 0, ram = 0, vram = 0;
        for (auto const& [i, amount] : units)
        {
            auto const kind = graph.resources[i].kind;
            if (local && graph.resources[i].host != request.host) continue;
            if (kind == ResourceKind::cpu) cpu = add(cpu, amount);
            if (kind == ResourceKind::host_memory) ram = add(ram, amount);
            if (kind == ResourceKind::gpu_memory) vram = add(vram, amount);
        }
        require(cpu >= cpu_required && ram >= ram_required, "native demand understates selected CPU or RAM footprint");
        if (gpu_required)
            require(selected_gpu && selected_vram && used.contains(*selected_gpu) && units[*selected_vram] >= gpu_required &&
                    kinds.contains(ResourceKind::gpu_compute) && vram >= gpu_required, "native demand omits selected GPU compute or VRAM");
        // Validate the already declared phase totals. Idle retained context
        // memory needs no invented GPU-compute work, and is not added twice.
        if (local && request.standing_gpu_bytes)
            require(selected_vram && resident[*selected_vram] >= request.standing_gpu_bytes &&
                    units[*selected_vram] >= add(gpu_required, request.standing_gpu_bytes),
                    "native demand omits standing selected GPU residency or allocation footprint");
        for (auto const& id : request.output_resources) require(used.contains(graph.find(id)), "native demand omits a physical output resource");
        if (scratch) for (auto const& id : request.scratch_resources) require(used.contains(graph.find(id)), "native demand omits a physical scratch resource");
        if (sources) for (auto const& id : request.source_resources) require(used.contains(graph.find(id)), "supervision demand omits a physical source resource");
    };
    // Worker/child and controller may run concurrently, but their inherited
    // affinity union cannot use more physical CPU lanes than that whole set.
    // GPU discovery can create retained driver/runtime workers before the
    // scoped generation cap. host_threads does not cap those existing threads,
    // so every GPU-session phase retains the whole inherited CPU-set envelope.
    auto concurrent_cpu = [&](uint64_t child)
    { return request.backend == "gpu" && !request.compute_enabled ? request.cpu_affinity_lanes : std::min(request.cpu_affinity_lanes, add(child, 1)); };
    if (building) coverage(stage.construction, concurrent_cpu(request.cpu_threads), add(request.ram_bytes, request.controller_ram_bytes), request.gpu_bytes, true, true);
    if (verifying)
    {
        uint64_t const cpu = request.verification_cpu_threads ? request.verification_cpu_threads : request.cpu_threads;
        uint64_t const ram = request.verification_ram_bytes ? request.verification_ram_bytes : request.ram_bytes;
        require(cpu <= n.max_cpu_threads && ram <= n.max_ram_bytes && request.verification_gpu_bytes <= n.max_gpu_bytes,
                "selected verifier exceeds reviewed native capacity envelope");
        require(!n.verification_cpu_threads || cpu == n.verification_cpu_threads, "verifier CPU cap differs from reviewed affinity envelope");
        coverage(stage.verification, concurrent_cpu(cpu), add(ram, request.controller_ram_bytes), request.verification_gpu_bytes, false, true);
    }
    if (!stage.supervision.empty()) coverage(stage.supervision, concurrent_cpu(request.adapter_cpu_threads),
        add(request.controller_ram_bytes, request.adapter_ram_bytes), 0, false, true, true);
    // These declarations cover their complete cohort. Native reader mean
    // work statistics and a memory-only 'proof reserve' cannot supply them.
    coverage(stage.new_farming, 1, 1, 0, false, false);
    coverage(stage.proof_reserve, 1, 1, 0, false, false);
    if (request.compute_enabled)
    {
        require(!n.compute_workers.empty() && !n.compute_owner_identity.empty() && !n.compute_broker_identity.empty() && !n.compute_broker_host.empty(),
            "selected compute requires reviewed owner, broker and physical worker scopes");
        uint64_t const generation = request.backend == "cpu" ? compute_generate_cpu : compute_generate_gpu;
        require(std::any_of(n.compute_workers.begin(), n.compute_workers.end(), [&](auto const& worker) { return bool(worker.allowed_kinds & generation); }),
            "reviewed compute workers cannot generate the selected construction backend");
        require(!request.compute_codec_ram_bytes || std::any_of(n.compute_workers.begin(), n.compute_workers.end(),
            [](auto const& worker) { return bool(worker.allowed_kinds & compute_encode); }), "reviewed compute workers have no permitted codec capability");
        uint64_t const service_ram = add(request.compute_session_ram_bytes, request.compute_codec_ram_bytes);
        uint64_t cpu = 0, ram = 0, broker_cpu = 0, broker_ram = 0; bool network = n.compute_broker_host == request.host;
        for (auto const& demand : n.compute_services)
        {
            auto const& resource = graph.resources[graph.find(demand.resource)];
            if (resource.host == n.compute_broker_host)
            {
                if (resource.kind == ResourceKind::cpu) broker_cpu = add(broker_cpu, add(demand.reserved_units, demand.required_units));
                if (resource.kind == ResourceKind::host_memory) broker_ram = add(broker_ram, add(demand.resident_units, demand.transient_units));
            }
            if (resource.kind == ResourceKind::network && (demand.reserved_units || demand.required_units)) network = true;
            if (resource.host != request.host) continue;
            if (resource.kind == ResourceKind::cpu) cpu = add(cpu, add(demand.reserved_units, demand.required_units));
            if (resource.kind == ResourceKind::host_memory) ram = add(ram, add(demand.resident_units, demand.transient_units));
        }
        require(!service_ram || (cpu && ram >= service_ram), "reviewed compute services omit the local broker/codec memory and CPU partitions");
        require(broker_cpu && broker_ram && network, "reviewed compute services omit the physical broker host or its remote network scope");
    }
    for (auto const& worker : n.compute_workers)
    {
        uint64_t cpu = 0, ram = 0, resident_ram = 0, vram = 0, resident_vram = 0;
        bool gpu = false, network = worker.host == request.host;
        std::optional<size_t> selected;
        if (worker.gpu_bytes) selected = graph.find(worker.gpu_identity + "/memory");
        for (auto const& demand : worker.reservation)
        {
            require(demand.bounded && !demand.required_units && !demand.work_us,
                "whole compute worker requires opaque globally bounded service reservations");
            auto const i = graph.find(demand.resource); auto const& resource = graph.resources[i];
            if (resource.kind == ResourceKind::network && demand.reserved_units) network = true;
            if (resource.host != worker.host) continue;
            if (resource.kind == ResourceKind::cpu) cpu = add(cpu, demand.reserved_units);
            if (resource.kind == ResourceKind::host_memory)
            { ram = add(ram, add(demand.resident_units, demand.transient_units)); resident_ram = add(resident_ram, demand.resident_units); }
            if (worker.gpu_bytes && resource.kind == ResourceKind::gpu_compute && i == graph.find(worker.gpu_identity + "/compute") && demand.reserved_units) gpu = true;
            if (selected && i == *selected)
            { vram = add(vram, add(demand.resident_units, demand.transient_units)); resident_vram = add(resident_vram, demand.resident_units); }
        }
        uint64_t const cpu_bound = worker.gpu_bytes && worker.host == request.host ?
            std::max(worker.cpu_threads, request.cpu_affinity_lanes) : worker.cpu_threads;
        require(cpu >= cpu_bound && ram >= worker.host_bytes && resident_ram >= worker.standing_host_bytes && network,
            "whole compute worker reservation omits physical CPU, total/standing host memory or remote network capacity");
        if (worker.gpu_bytes)
            require(selected && graph.resources[*selected].kind == ResourceKind::gpu_memory && graph.resources[*selected].host == worker.host && gpu &&
                vram >= add(worker.gpu_bytes, worker.standing_gpu_bytes) && resident_vram >= worker.standing_gpu_bytes,
                "whole compute worker reservation omits physical GPU allocation or standing residency");
    }
    stage.scope_complete = true;
}
inline void bind_native_stage(Profile const& profile, NativeRequest const& request, uint64_t ready_members,
                              Stage& stage, bool building, bool verifying)
{ bind_native_stage(AssessmentPreparation(profile), request, ready_members, stage, building, verifying); }

inline void require_target_pool_difficulty(Profile const& profile, uint64_t current, uint64_t protocol = 0)
{
    resource_detail::require(((!current && !profile.native.min_pool_difficulty) || (current && profile.native.min_pool_difficulty && current >= profile.native.min_pool_difficulty)) &&
        profile.native.pool_protocol_version == protocol,
        "current target pool difficulty is below the reviewed full-solver floor or its protocol differs");
}

inline Assessment assess_stage(AssessmentPreparation const& preparation, Stage const& stage)
{
    using namespace resource_detail;
    auto const& profile = preparation.profile(); auto const& graph = preparation.graph(); text(stage.name, true);
    require(stage.remaining.size() <= max_workloads, "too many stage workload partitions");
    Assessment result;
    if (!stage.scope_complete) fail(result, "incomplete-stage-resource-scope:" + stage.name);
    std::vector<bool> external_gpu(graph.resources.size()), own_gpu(graph.resources.size());
    std::vector<std::vector<size_t>> paths;
    std::vector<std::pair<uint64_t, uint64_t>> responses;
    for (auto const& r : graph.resources)
    {
        Margin margin; margin.resource = r.id; margin.kind = r.kind; margin.capacity = r.capacity;
        result.margins.push_back(std::move(margin));
        if (!r.capacity_known) fail(result, "unknown-capacity:" + r.id);
    }
    std::map<std::string, uint64_t> remaining;
    for (auto const& item : stage.remaining)
        require(remaining.emplace(item.workload, item.count).second, "duplicate stage workload partition");
    for (auto const& w : profile.workloads)
    {
        auto at = remaining.find(w.id);
        // Omitting an outside workload retains it; it never silently vanishes.
        uint64_t const count = at == remaining.end() ? w.plot_count : at->second;
        require(w.external_service_node_id.empty() || count == w.plot_count,
            "migration cannot reduce an unscanned external service reservation");
        require(count <= w.plot_count, "stage increases an existing workload inventory");
        if (at != remaining.end()) remaining.erase(at);
        bool const active = count != 0;
        bool const pending = std::any_of(w.uses.begin(), w.uses.end(), [](Use const& u) { return u.blocking_us != 0; });
        if (w.cost == CostKind::unknown || !w.bounded || !w.scope_complete || w.uses.empty())
            fail(result, "unknown-cost:" + w.id);
        if (w.format.empty() || w.decoder.empty() || w.decoder_version.empty() || w.backend.empty() ||
            w.pool_difficulty.empty() || w.source.empty()) fail(result, "missing-cost-scope:" + w.id);
        uint64_t const quality = !active ? 0 : stage.future_filter ? w.future_quality_burst : w.quality_burst;
        uint64_t const proofs = !active ? 0 : stage.future_filter ? w.future_proof_burst : w.proof_burst;
        if (active && (!w.period_us || !w.deadline_us || !quality || !proofs)) fail(result, "missing-burst-or-deadline:" + w.id);
        std::vector<size_t> path;
        for (auto const& u : w.uses)
        {
            size_t const i = graph.find(u.resource); auto& m = result.margins[i];
            m.resident_units = add(m.resident_units, u.resident_units);
            m.reserved_units = add(m.reserved_units, u.reserved_units);
            // A removed file does not prove pending requests or transient
            // decoder allocations have drained. Keep that declared headroom
            // and complete carried-in queue bound until the profile is refreshed.
            m.transient_units = add(m.transient_units, u.transient_units);
            // Independently owned queues may each contribute carried-in work.
            // Each declaration bounds its whole queue, not one average request.
            m.blocking_us = add(m.blocking_us, u.blocking_us);
            if (active || u.blocking_us)
            {
                m.required_units = add(m.required_units, u.required_units);
                m.request_units = std::max(m.request_units, u.required_units);
                m.period_us = minimum(m.period_us, w.period_us); m.deadline_us = minimum(m.deadline_us, w.deadline_us);
                if (!memory(m.kind) && w.cost == CostKind::serialized) path.push_back(i);
            }
            if (active)
            {
                m.serialized_work_us = add(m.serialized_work_us, add(multiply(quality, u.quality_us), multiply(proofs, u.proof_us)));
                if (!memory(m.kind) && w.cost == CostKind::serialized)
                {
                    if (!u.required_units || (!u.quality_us && !u.proof_us)) fail(result, "missing-service-bound:" + w.id + ":" + m.resource);
                }
            }
            if (m.kind == ResourceKind::gpu_compute && (active || u.reserved_units || u.blocking_us)) external_gpu[i] = true;
        }
        if (active || pending)
        {
            if (w.cost == CostKind::reservation && !w.response_tail_us) fail(result, "missing-reserved-response-bound:" + w.id);
            paths.push_back(std::move(path)); responses.push_back({w.response_tail_us, w.deadline_us});
        }
    }
    require(remaining.empty(), "stage names an unknown mixed farm workload");
    bool proof_service = false;
    for (auto const& d : stage.proof_reserve)
        proof_service = proof_service || (!memory(graph.resources[graph.find(d.resource)].kind) && d.bounded && (d.work_us || d.reserved_units));
    if (!stage.new_farming.empty() && !proof_service) fail(result, "missing-native-full-proof-reserve");
    auto demands = [&](std::vector<Demand> const& values, char const* label, bool own)
    {
        require(values.size() <= max_resources * 4, "too many native stage resource demands");
        std::vector<size_t> path; uint64_t deadline = 0;
        std::vector<uint64_t> phase_cpu(graph.resources.size());
        for (auto const& d : values)
        {
            size_t const i = graph.find(d.resource); auto& m = result.margins[i];
            if (!d.bounded) fail(result, std::string("unknown-") + label + "-cost:" + m.resource);
            if (memory(m.kind)) require(!d.reserved_units && !d.required_units && !d.work_us && !d.blocking_us,
                                        "native memory demand has service units");
            else require(!d.resident_units && !d.transient_units, "native service demand has byte units");
            require(!(d.reserved_units && (d.work_us || d.required_units)), "native demand double charges opaque and modeled service");
            m.resident_units = add(m.resident_units, d.resident_units); m.transient_units = add(m.transient_units, d.transient_units);
            m.reserved_units = add(m.reserved_units, d.reserved_units); m.required_units = add(m.required_units, d.required_units);
            if (own && m.kind == ResourceKind::cpu)
            {
                phase_cpu[i] = add(phase_cpu[i], d.required_units);
                m.request_units = std::max(m.request_units, phase_cpu[i]);
            }
            else m.request_units = std::max(m.request_units, d.required_units);
            m.serialized_work_us = add(m.serialized_work_us, d.work_us); m.blocking_us = add(m.blocking_us, d.blocking_us);
            m.period_us = minimum(m.period_us, d.period_us); m.deadline_us = minimum(m.deadline_us, d.deadline_us);
            if (d.work_us)
            {
                if (!d.required_units || !d.period_us || !d.deadline_us) fail(result, std::string("missing-") + label + "-service-bound:" + m.resource);
                path.push_back(i); deadline = minimum(deadline, d.deadline_us);
            }
            if (m.kind == ResourceKind::gpu_compute && (d.work_us || d.reserved_units || d.required_units))
            { if (own) own_gpu[i] = true; else external_gpu[i] = true; }
        }
        if (!path.empty()) { paths.push_back(std::move(path)); responses.push_back({0, deadline}); }
    };
    // The harvester/solver serving our completed native plots is still an
    // outside service, not construction work controlled by this executor.
    demands(stage.new_farming, "native-farming", false); demands(stage.proof_reserve, "native-full-proof", false);
    demands(stage.construction, "construction", true); demands(stage.verification, "verification", true);
    demands(stage.supervision, "supervision", true);
    demands(stage.background, "compute-services", false);
    for (size_t i = 0; i < result.margins.size(); ++i)
    {
        auto& m = result.margins[i]; auto const& r = graph.resources[i];
        uint64_t required = m.required_units;
        if (!memory(m.kind) && r.conditional_shared)
        {
            // Explicitly qualified shared service uses the whole serialized
            // finite-batch bound. It needs each actual indivisible request's
            // shape, not their simultaneous sum. Opaque reserved lanes remain
            // protected separately; external queues are not under our control.
            required = m.request_units;
            result.conditional_shared_service = result.conditional_shared_service || m.required_units > m.request_units;
        }
        uint64_t const occupied = memory(m.kind) ? add(m.resident_units, m.transient_units) : add(m.reserved_units, required);
        m.occupied_units = occupied;
        if (occupied > m.capacity) fail(result, "capacity-exceeded:" + m.resource);
        if (m.request_units > m.capacity) fail(result, "request-shape-exceeds-capacity:" + m.resource);
        m.remaining_units = occupied <= m.capacity ? m.capacity - occupied : 0;
        // Deliberately serialize the whole finite burst. Parallel lanes only
        // establish a request can run with its declared shape; never divide
        // work by core/GPU count. The envelope assumes bounded service and
        // carried-in blocking, not control over a foreign scheduler's queue.
        m.drain_us = add(m.serialized_work_us, m.blocking_us);
        if (m.period_us && m.drain_us > m.period_us) fail(result, "burst-does-not-drain:" + m.resource);
        if (external_gpu[i] && own_gpu[i])
        {
            if (!r.conditional_shared) fail(result, "shared-gpu-not-qualified:" + m.resource);
            else result.conditional_shared_gpu = true;
        }
        for (auto const* field : {"resident_units", "transient_units", "reserved_units", "required_units", "request_units",
                                 "occupied_units", "remaining_units", "serialized_work_us", "blocking_us", "drain_us", "period_us", "deadline_us"})
            m.worst_stages.emplace(field, stage.name);
    }
    for (size_t i = 0; i < paths.size(); ++i)
    {
        uint64_t response = responses[i].first;
        std::set<size_t> unique(paths[i].begin(), paths[i].end());
        for (auto resource : unique) response = add(response, result.margins[resource].drain_us);
        if (responses[i].second && response > responses[i].second) fail(result, "response-deadline-exceeded:" + stage.name);
    }
    return result;
}
inline Assessment assess_stage(Profile const& profile, Stage const& stage)
{ return assess_stage(AssessmentPreparation(profile), stage); }
inline void merge_stage_assessment(Assessment& result, Assessment next, std::string const& stage_name)
{
    resource_detail::text(stage_name, true);
    resource_detail::require(result.margins.empty() || result.margins.size() == next.margins.size(), "migration resource aggregation scope changed");
    result.per_dimension_worst = true; result.feasible = result.feasible && next.feasible;
    result.conditional_shared_gpu = result.conditional_shared_gpu || next.conditional_shared_gpu;
    result.conditional_shared_service = result.conditional_shared_service || next.conditional_shared_service;
    for (auto const& reason : next.reasons)
        if (result.reasons.size() < 1024) result.reasons.push_back(stage_name + ":" + reason);
    // Keep the least remaining capacity observed at any intermediate stage.
    if (result.margins.empty()) result.margins = std::move(next.margins);
    else for (size_t i = 0; i < result.margins.size(); ++i)
    {
        auto& peak = result.margins[i]; auto const& current = next.margins[i];
        resource_detail::require(peak.resource == current.resource && peak.kind == current.kind && peak.capacity == current.capacity,
                                 "migration resource aggregation physical identity changed");
        auto maximum = [&](uint64_t& value, uint64_t other, char const* field)
        { if (other > value) { value = other; peak.worst_stages[field] = stage_name; } };
        auto minimum_bound = [&](uint64_t& value, uint64_t other, char const* field, bool zero_unset = false)
        { if ((zero_unset && !value && other) || (other < value && (!zero_unset || other))) { value = other; peak.worst_stages[field] = stage_name; } };
        minimum_bound(peak.remaining_units, current.remaining_units, "remaining_units");
        maximum(peak.resident_units, current.resident_units, "resident_units");
        maximum(peak.transient_units, current.transient_units, "transient_units");
        maximum(peak.reserved_units, current.reserved_units, "reserved_units");
        maximum(peak.required_units, current.required_units, "required_units");
        maximum(peak.request_units, current.request_units, "request_units");
        maximum(peak.occupied_units, current.occupied_units, "occupied_units");
        maximum(peak.serialized_work_us, current.serialized_work_us, "serialized_work_us");
        maximum(peak.blocking_us, current.blocking_us, "blocking_us");
        maximum(peak.drain_us, current.drain_us, "drain_us");
        minimum_bound(peak.period_us, current.period_us, "period_us", true);
        minimum_bound(peak.deadline_us, current.deadline_us, "deadline_us", true);
    }
}
inline Assessment assess_transition(Profile const& profile, std::vector<Stage> const& stages)
{
    resource_detail::require(!stages.empty() && stages.size() <= max_stages, "invalid migration stage count");
    Assessment result;
    for (auto const& stage : stages) merge_stage_assessment(result, assess_stage(profile, stage), stage.name);
    if (!stages.back().future_filter) resource_detail::fail(result, "missing-final-future-filter-assessment");
    return result;
}

inline uint64_t profile_memory(Profile const& profile)
{
    using namespace resource_detail;
    uint64_t bytes = sizeof(profile);
    auto string = [&](std::string const& s) { bytes = add(bytes, add(s.capacity(), 1)); };
    string(profile.source); string(profile.captured_at); string(profile.producer);
    bytes = add(bytes, multiply(profile.resources.capacity(), sizeof(Resource)));
    bytes = add(bytes, multiply(profile.workloads.capacity(), sizeof(Workload)));
    for (auto const& r : profile.resources)
    {
        string(r.id); string(r.host); bytes = add(bytes, multiply(r.aliases.capacity(), sizeof(std::string)));
        for (auto const& a : r.aliases) string(a);
    }
    for (auto const& w : profile.workloads)
    {
        for (auto const* s : {&w.id, &w.format, &w.decoder, &w.decoder_version, &w.backend, &w.pool_difficulty, &w.source, &w.captured_at,
            &w.external_service_node_id, &w.external_service_host}) string(*s);
        bytes = add(bytes, multiply(w.uses.capacity(), sizeof(Use)));
        for (auto const& u : w.uses) string(u.resource);
    }
    auto const& native = profile.native;
    string(native.host); string(native.reader); string(native.backend); string(native.engine);
    for (auto const& [role, host] : native.service_hosts) { string(role); string(host); bytes = add(bytes, 128); }
    string(native.gpu_compute_resource); string(native.gpu_memory_resource);
    auto demands = [&](std::vector<Demand> const& values)
    { bytes = add(bytes, multiply(values.capacity(), sizeof(Demand))); for (auto const& d : values) string(d.resource); };
    demands(native.construction); demands(native.verification); demands(native.supervision); demands(native.compute_services);
    string(native.compute_owner_identity); string(native.compute_broker_identity); string(native.compute_broker_host);
    bytes = add(bytes, multiply(native.compute_workers.capacity(), sizeof(ComputeWorkerEnvelope)));
    for (auto const& worker : native.compute_workers)
    { string(worker.tls_identity); string(worker.host); string(worker.gpu_identity); demands(worker.reservation); }
    for (auto const* mappings : {&native.output_resources, &native.scratch_resources, &native.source_resources})
    { bytes = add(bytes, multiply(mappings->capacity(), sizeof(std::string))); for (auto const& id : *mappings) string(id); }
    bytes = add(bytes, multiply(native.cohorts.capacity(), sizeof(NativeCohort)));
    for (auto const& c : native.cohorts) { demands(c.new_farming); demands(c.proof_reserve); }
    return bytes;
}
inline uint64_t resource_preparation_memory(ResourcePreparation const& value)
{
    using namespace resource_detail;
    uint64_t bytes = add(sizeof(value), add(multiply(value.facts.capacity(), sizeof(ResourcePreparationFact)),
        multiply(value.missing.capacity(), sizeof(ResourcePreparationNeed))));
    auto string = [&](std::string const& text) { bytes = add(bytes, add(text.capacity(), 1)); };
    for (auto const& fact : value.facts)
        for (auto const* field : {&fact.path, &fact.value, &fact.basis, &fact.source, &fact.scope}) string(*field);
    for (auto const& need : value.missing)
        for (auto const* field : {&need.group, &need.path, &need.action}) string(*field);
    return bytes;
}
inline void validate_resource_preparation(ResourcePreparation const& value)
{
    using namespace resource_detail;
    require(value.facts.size() <= max_preparation_entries && value.missing.size() <= max_preparation_entries,
        "resource preparation report exceeds its entry bound");
    auto field_text = [&](std::string const& value)
    { require(!value.empty() && value.size() <= 4096, "invalid resource preparation text"); };
    for (auto const& fact : value.facts)
    {
        for (auto const* field : {&fact.path, &fact.value, &fact.source, &fact.scope}) field_text(*field);
        require(fact.basis == "observed" || fact.basis == "declared" || fact.basis == "unknown",
            "unsupported resource preparation provenance");
    }
    for (auto const& need : value.missing)
        for (auto const* field : {&need.group, &need.path, &need.action}) field_text(*field);
}
// A checklist of absent declarations, not a second feasibility model. The
// normal review still binds the concrete execution and assesses every stage.
inline ResourcePreparation prepare_resource_declarations(Profile const& profile)
{
    using namespace resource_detail;
    validate(profile); Graph graph(profile); ResourcePreparation result;
    auto need = [&](bool absent, std::string group, std::string path, std::string action)
    { if (absent) { require(result.missing.size() < max_preparation_entries, "too many missing resource declarations");
        result.missing.push_back({std::move(group), std::move(path), std::move(action)}); } };
    for (size_t i = 0; i < profile.resources.size(); ++i)
    {
        auto const& resource = profile.resources[i];
        need(!resource.capacity_known, "physical-capacity", "farm_resources.resources[" + std::to_string(i) + "]",
            "Declare capacity and capacity_known for " + resource.id + "; bytes for memory, offered parallel lanes otherwise. Observed availability is not a grant.");
    }
    for (size_t i = 0; i < profile.workloads.size(); ++i)
    {
        auto const& workload = profile.workloads[i]; auto const path = "farm_resources.workloads[" + std::to_string(i) + "]";
        need(workload.cost == CostKind::unknown || !workload.bounded || !workload.scope_complete,
            "existing-services", path, "For " + workload.id + ", declare cost (reservation or serialized), bounded and scope_complete only after covering all services, queues, resident/transient memory and full proofs.");
        need(workload.uses.empty(), "existing-services", path + ".uses", "Map this workload's physical CPU, memory, storage, solver and any network/GPU resources; include shared services.");
        auto unknown = [](std::string const& value) { return value.empty() || value == "unknown"; };
        need(unknown(workload.format) || workload.format.ends_with("/unknown") || unknown(workload.decoder) ||
            unknown(workload.decoder_version) || unknown(workload.backend) || unknown(workload.pool_difficulty) ||
            unknown(workload.source) || workload.source == "inventory-template", "existing-services", path,
            "Supply format, decoder, decoder_version, backend, pool_difficulty and source for the actual cost envelope. Unknown/template labels and header-container versions do not establish the running decoder's scope.");
        need(workload.plot_count && (!workload.quality_burst || !workload.proof_burst || !workload.future_quality_burst ||
            !workload.future_proof_burst || !workload.period_us || !workload.deadline_us), "arrival-bounds", path,
            "Declare quality_burst, proof_burst, future_quality_burst, future_proof_burst, period_us and deadline_us for the complete active population, including future filters.");
        need(workload.cost == CostKind::reservation && !workload.response_tail_us, "existing-services", path + ".response_tail_us",
            "Declare the complete reserved service response bound; opaque reservations remain charged after source retirement.");
        bool missing_service = false;
        for (auto const& use : workload.uses)
        {
            auto const kind = graph.resources[graph.find(use.resource)].kind;
            missing_service = missing_service || (workload.cost == CostKind::serialized && !memory(kind) &&
                (!use.required_units || (!use.quality_us && !use.proof_us)));
        }
        need(missing_service, "service-bounds", path + ".uses",
            "Each serialized service entry needs required_units and quality_us/proof_us as whole-request bounds; blocking_us covers the complete carried-in queue, not sampled latency. Memory entries use bytes.");
    }
    auto const& native = profile.native; std::string const path = "farm_resources.native";
    need(!native.bounded || !native.scope_complete || !native.covers_future_filter || !native.phase_totals_include_controller,
        "replacement-scope", path, "Review bounded, scope_complete, covers_future_filter and phase_totals_include_controller for the entire replacement farm, controller and foreign services.");
    need(!native.k || !native.strength || native.host.empty() || native.reader.empty() || native.backend.empty() || native.engine.empty(),
        "replacement-scope", path, "Set k, strength, host, reader, backend and engine to the intended replacement execution; observed reader provenance does not select construction settings.");
    need(!native.max_ram_bytes || !native.max_cpu_threads || !native.max_batch_members || !native.max_group_members ||
        !native.max_construction_us || !native.max_verification_us, "whole-file-bounds", path,
        "Declare max_ram_bytes, max_cpu_threads, max_batch_members, max_group_members, max_construction_us and max_verification_us. Bounds cover a whole output file and its proof verification, not one batch or a timeout sample.");
    need(native.output_resources.empty(), "physical-bindings", path + ".output_resources",
        "Map every intended output filesystem to its actual storage resource; reuse drive aliases for physically shared storage.");
    auto demands = [&](std::vector<Demand> const& values, std::string const& field)
    {
        need(values.empty(), "native-demands", field, "Declare complete phase-total CPU/RAM, output I/O and relevant scratch/source/GPU demands, including controller residency; service time and whole carried-in blocking must be bounded.");
        bool missing_bound = false, cpu = false, ram = false;
        std::map<std::string, std::set<ResourceKind>> execution_hosts;
        for (auto const& demand : values)
        {
            auto const& resource = graph.resources[graph.find(demand.resource)]; auto const kind = resource.kind;
            cpu = cpu || kind == ResourceKind::cpu; ram = ram || kind == ResourceKind::host_memory;
            if (kind == ResourceKind::cpu || kind == ResourceKind::host_memory || kind == ResourceKind::gpu_compute ||
                kind == ResourceKind::gpu_memory || kind == ResourceKind::solver) execution_hosts[resource.host].insert(kind);
            bool const absent = !demand.bounded || (memory(kind) ? !demand.resident_units && !demand.transient_units :
                (!demand.reserved_units && (!demand.required_units || !demand.work_us || !demand.period_us || !demand.deadline_us)));
            missing_bound = missing_bound || absent;
        }
        need(missing_bound, "native-demands", field,
            "Each entry needs bounded and a positive memory footprint or service reservation/work envelope; work needs required_units, work_us, period_us and deadline_us. Samples do not establish worst-case service or queue bounds.");
        bool missing_host = !values.empty() && (!cpu || !ram);
        for (auto const& [host, kinds] : execution_hosts)
        { (void)host; missing_host = missing_host || !kinds.contains(ResourceKind::cpu) || !kinds.contains(ResourceKind::host_memory); }
        need(missing_host, "physical-bindings", field, "Include CPU and host-memory demands on every physical execution host used by this phase; solver/GPU work also requires its host CPU and RAM scope.");
    };
    demands(native.construction, path + ".construction"); demands(native.verification, path + ".verification");
    demands(native.supervision, path + ".supervision");
    need(native.cohorts.empty(), "farming-and-proofs", path + ".cohorts",
        "Declare increasing max_members cohorts, each with complete new_farming and separate proof_reserve demands for the harvester/farmer/full node/solver. Quality calibration cannot supply a full-proof reserve.");
    for (size_t i = 0; i < native.cohorts.size(); ++i)
    {
        auto const cohort = path + ".cohorts[" + std::to_string(i) + "]";
        demands(native.cohorts[i].new_farming, cohort + ".new_farming"); demands(native.cohorts[i].proof_reserve, cohort + ".proof_reserve");
        need(!native.cohorts[i].proof_reserve.empty() && std::none_of(native.cohorts[i].proof_reserve.begin(), native.cohorts[i].proof_reserve.end(), [&](auto const& demand)
            { return !memory(graph.resources[graph.find(demand.resource)].kind) && (demand.work_us || demand.reserved_units); }),
            "farming-and-proofs", cohort + ".proof_reserve", "A memory allowance alone does not reserve full-proof service. Declare complete solver work or an opaque service reservation on its physical execution resources.");
    }
    if (native.backend == "gpu" && (native.gpu_compute_resource.empty() || native.gpu_memory_resource.empty() ||
        !native.max_gpu_bytes || !native.standing_gpu_bytes))
        result.facts.push_back({path, "GPU execution scope depends on the selected local or broker execution path", "unknown", "resource-preparation-checklist",
            "Local GPU execution requires gpu_compute_resource, gpu_memory_resource, max_gpu_bytes and standing_gpu_bytes. Broker generation instead requires complete reviewed compute_workers and compute_services. Worker records alone do not select either path; the concrete request still must pass admission."});
    validate_resource_preparation(result); return result;
}
inline void write_resource_preparation_json(std::ostream& out, ResourcePreparation const& value)
{
    validate_resource_preparation(value);
    auto quote = [&](std::string const& text) { pos2gpu::write_json_string(out, text); };
    out << "{\"assessment_required\":true,\"facts\":["; bool comma = false;
    for (auto const& fact : value.facts)
    {
        if (comma) out << ','; comma = true; out << "{\"path\":"; quote(fact.path); out << ",\"value\":"; quote(fact.value);
        out << ",\"basis\":"; quote(fact.basis); out << ",\"source\":"; quote(fact.source); out << ",\"scope\":"; quote(fact.scope); out << '}';
    }
    out << "],\"missing\":["; comma = false;
    for (auto const& need : value.missing)
    {
        if (comma) out << ','; comma = true; out << "{\"group\":"; quote(need.group); out << ",\"path\":"; quote(need.path);
        out << ",\"action\":"; quote(need.action); out << '}';
    }
    out << "]}";
}
inline ResourcePreparation parse_resource_preparation(profile_detail::Json const& object)
{
    using namespace profile_detail;
    auto required = [&](Json const& value, char const* key) -> Json const&
    { auto const* at = field(value, key); resource_detail::require(at, "missing resource preparation field"); return *at; };
    resource_detail::require(boolean(required(object, "assessment_required")), "resource preparation cannot grant admission");
    auto const& facts = required(object, "facts"); auto const& missing = required(object, "missing");
    resource_detail::require(facts.kind == Json::array && missing.kind == Json::array &&
        facts.items.size() <= max_preparation_entries && missing.items.size() <= max_preparation_entries,
        "resource preparation report exceeds its entry bound");
    ResourcePreparation result;
    for (auto const& fact : facts.items) result.facts.push_back({text(required(fact, "path")), text(required(fact, "value")),
        text(required(fact, "basis")), text(required(fact, "source")), text(required(fact, "scope"))});
    for (auto const& need : missing.items) result.missing.push_back({text(required(need, "group")), text(required(need, "path")), text(required(need, "action"))});
    validate_resource_preparation(result); return result;
}
namespace resource_detail
{
inline std::vector<Demand> parse_demands(profile_detail::Json const& array)
{
    using namespace profile_detail;
    require(array.kind == Json::array && array.items.size() <= max_resources * 4, "invalid native demand array");
    std::vector<Demand> result;
    for (auto const& item : array.items)
    {
        auto number = [&](char const* name) { auto const* value = field(item, name); return value ? integer(*value) : uint64_t(0); };
        auto const* id = field(item, "resource"); require(id, "native demand is missing a physical resource");
        Demand d; d.resource = profile_detail::text(*id);
        d.resident_units = number("resident_units"); d.transient_units = number("transient_units"); d.reserved_units = number("reserved_units");
        d.required_units = number("required_units"); d.work_us = number("work_us"); d.blocking_us = number("blocking_us");
        d.period_us = number("period_us"); d.deadline_us = number("deadline_us");
        if (auto const* bounded = field(item, "bounded")) d.bounded = boolean(*bounded);
        result.push_back(std::move(d));
    }
    return result;
}
inline void write_demands(std::ostream& out, std::vector<Demand> const& demands)
{
    out << '['; bool comma = false;
    for (auto const& d : demands)
    {
        if (comma) out << ','; comma = true; out << "{\"resource\":"; pos2gpu::write_json_string(out, d.resource);
        out << ",\"resident_units\":" << d.resident_units << ",\"transient_units\":" << d.transient_units
            << ",\"reserved_units\":" << d.reserved_units << ",\"required_units\":" << d.required_units
            << ",\"work_us\":" << d.work_us << ",\"blocking_us\":" << d.blocking_us
            << ",\"period_us\":" << d.period_us << ",\"deadline_us\":" << d.deadline_us
            << ",\"bounded\":" << (d.bounded ? "true" : "false") << '}';
    }
    out << ']';
}
inline NativeEnvelope parse_native(profile_detail::Json const& object)
{
    using namespace profile_detail;
    NativeEnvelope n;
    auto number = [&](char const* name) { auto const* value = field(object, name); return value ? integer(*value) : uint64_t(0); };
    auto string = [&](char const* name) { auto const* value = field(object, name); return value ? profile_detail::text(*value) : std::string{}; };
    if (auto const* value = field(object, "bounded")) n.bounded = boolean(*value);
    if (auto const* value = field(object, "scope_complete")) n.scope_complete = boolean(*value);
    if (auto const* value = field(object, "covers_future_filter")) n.covers_future_filter = boolean(*value);
    if (auto const* value = field(object, "phase_totals_include_controller")) n.phase_totals_include_controller = boolean(*value);
    n.host = string("host"); n.k = number("k"); n.strength = number("strength"); n.reader = string("reader"); n.backend = string("backend"); n.engine = string("engine");
    if (auto const* hosts = field(object, "service_hosts"))
    {
        require(hosts->kind == Json::object && hosts->fields.size() <= 4, "invalid native service host bindings");
        for (auto const& [role, host] : hosts->fields) n.service_hosts.emplace(role, profile_detail::text(host));
    }
    n.gpu_compute_resource = string("gpu_compute_resource"); n.gpu_memory_resource = string("gpu_memory_resource");
    n.max_ram_bytes = number("max_ram_bytes"); n.max_cpu_threads = number("max_cpu_threads"); n.max_gpu_bytes = number("max_gpu_bytes");
    n.standing_gpu_bytes = number("standing_gpu_bytes");
    n.max_batch_members = number("max_batch_members"); n.max_group_members = number("max_group_members");
    n.max_construction_us = number("max_construction_us"); n.max_verification_us = number("max_verification_us");
    n.verification_cpu_threads = number("verification_cpu_threads");
    n.min_pool_difficulty = number("min_pool_difficulty");
    n.pool_protocol_version = number("pool_protocol_version");
    for (auto const& [name, target] : {std::pair{"output_resources", &n.output_resources}, {"scratch_resources", &n.scratch_resources}, {"source_resources", &n.source_resources}})
        if (auto const* values = field(object, name))
        {
            require(values->kind == Json::array && values->items.size() <= max_resources, "invalid native physical mapping array");
            for (auto const& value : values->items) target->push_back(profile_detail::text(value));
        }
    if (auto const* values = field(object, "construction")) n.construction = parse_demands(*values);
    if (auto const* values = field(object, "verification")) n.verification = parse_demands(*values);
    if (auto const* values = field(object, "supervision")) n.supervision = parse_demands(*values);
    n.compute_owner_identity = string("compute_owner_identity"); n.compute_broker_identity = string("compute_broker_identity");
    n.compute_broker_host = string("compute_broker_host");
    if (auto const* values = field(object, "compute_services")) n.compute_services = parse_demands(*values);
    if (auto const* workers = field(object, "compute_workers"))
    {
        require(workers->kind == Json::array && workers->items.size() <= max_workloads, "invalid reviewed compute worker array");
        for (auto const& object : workers->items)
        {
            require(object.kind == Json::object, "invalid reviewed compute worker declaration");
            auto number = [&](char const* name) { auto const* value = field(object, name); return value ? integer(*value) : uint64_t(0); };
            auto string = [&](char const* name) { auto const* value = field(object, name); return value ? profile_detail::text(*value) : std::string{}; };
            ComputeWorkerEnvelope worker; worker.tls_identity = string("tls_identity"); worker.host = string("host"); worker.gpu_identity = string("gpu_identity");
            worker.cpu_threads = number("cpu_threads"); worker.slots = number("slots"); worker.host_bytes = number("host_bytes"); worker.gpu_bytes = number("gpu_bytes");
            worker.standing_host_bytes = number("standing_host_bytes"); worker.standing_gpu_bytes = number("standing_gpu_bytes");
            if (auto const* kinds = field(object, "allowed_kinds"))
            {
                require(kinds->kind == Json::array && kinds->items.size() <= 3, "invalid reviewed compute task kind array");
                for (auto const& value : kinds->items)
                {
                    auto const kind = profile_detail::text(value); uint64_t bit = 0;
                    if (kind == "generate-cpu") bit = compute_generate_cpu;
                    else if (kind == "generate-gpu") bit = compute_generate_gpu;
                    else if (kind == "encode") bit = compute_encode;
                    else throw std::runtime_error("unknown reviewed compute task kind");
                    require(!(worker.allowed_kinds & bit), "repeated reviewed compute task kind"); worker.allowed_kinds |= bit;
                }
            }
            if (auto const* values = field(object, "reservation")) worker.reservation = parse_demands(*values);
            n.compute_workers.push_back(std::move(worker));
        }
    }
    if (auto const* cohorts = field(object, "cohorts"))
    {
        require(cohorts->kind == Json::array && cohorts->items.size() <= 256, "invalid native resource cohort array");
        for (auto const& object : cohorts->items)
        {
            NativeCohort c;
            if (auto const* value = field(object, "max_members")) c.max_members = integer(*value);
            if (auto const* value = field(object, "new_farming")) c.new_farming = parse_demands(*value);
            if (auto const* value = field(object, "proof_reserve")) c.proof_reserve = parse_demands(*value);
            n.cohorts.push_back(std::move(c));
        }
    }
    return n;
}
inline void write_native(std::ostream& out, NativeEnvelope const& n)
{
    auto quote = [&](std::string const& s) { pos2gpu::write_json_string(out, s); };
    out << "{\"bounded\":" << (n.bounded ? "true" : "false") << ",\"scope_complete\":" << (n.scope_complete ? "true" : "false")
        << ",\"covers_future_filter\":" << (n.covers_future_filter ? "true" : "false")
        << ",\"phase_totals_include_controller\":" << (n.phase_totals_include_controller ? "true" : "false")
        << ",\"k\":" << n.k << ",\"strength\":" << n.strength << ",\"host\":"; quote(n.host); out << ",\"reader\":"; quote(n.reader);
    out << ",\"backend\":"; quote(n.backend); out << ",\"engine\":"; quote(n.engine);
    if (!n.service_hosts.empty())
    {
        out << ",\"service_hosts\":{"; bool comma = false;
        for (auto const& [role, host] : n.service_hosts) { if (comma) out << ','; comma = true; quote(role); out << ':'; quote(host); }
        out << '}';
    }
    out << ",\"gpu_compute_resource\":"; quote(n.gpu_compute_resource); out << ",\"gpu_memory_resource\":"; quote(n.gpu_memory_resource);
    out << ",\"max_ram_bytes\":" << n.max_ram_bytes << ",\"max_cpu_threads\":" << n.max_cpu_threads << ",\"max_gpu_bytes\":" << n.max_gpu_bytes
        << ",\"standing_gpu_bytes\":" << n.standing_gpu_bytes
        << ",\"max_batch_members\":" << n.max_batch_members << ",\"max_group_members\":" << n.max_group_members
        << ",\"max_construction_us\":" << n.max_construction_us << ",\"max_verification_us\":" << n.max_verification_us
        << ",\"verification_cpu_threads\":" << n.verification_cpu_threads;
    if (n.min_pool_difficulty) out << ",\"min_pool_difficulty\":" << n.min_pool_difficulty << ",\"pool_protocol_version\":" << n.pool_protocol_version;
    for (auto const& [name, values] : {std::pair{"output_resources", &n.output_resources}, {"scratch_resources", &n.scratch_resources}, {"source_resources", &n.source_resources}})
    {
        out << ",\"" << name << "\":["; bool comma = false;
        for (auto const& id : *values) { if (comma) out << ','; comma = true; quote(id); } out << ']';
    }
    out << ",\"construction\":"; write_demands(out, n.construction); out << ",\"verification\":"; write_demands(out, n.verification);
    out << ",\"supervision\":"; write_demands(out, n.supervision);
    if (!n.compute_workers.empty() || !n.compute_services.empty())
    {
        out << ",\"compute_owner_identity\":"; quote(n.compute_owner_identity);
        out << ",\"compute_broker_identity\":"; quote(n.compute_broker_identity);
        out << ",\"compute_broker_host\":"; quote(n.compute_broker_host);
        out << ",\"compute_services\":"; write_demands(out, n.compute_services);
        out << ",\"compute_workers\":["; bool comma = false;
        for (auto const& worker : n.compute_workers)
        {
            if (comma) out << ','; comma = true; out << "{\"tls_identity\":"; quote(worker.tls_identity);
            out << ",\"host\":"; quote(worker.host); out << ",\"gpu_identity\":"; quote(worker.gpu_identity);
            out << ",\"cpu_threads\":" << worker.cpu_threads << ",\"slots\":" << worker.slots << ",\"host_bytes\":" << worker.host_bytes
                << ",\"gpu_bytes\":" << worker.gpu_bytes << ",\"standing_host_bytes\":" << worker.standing_host_bytes
                << ",\"standing_gpu_bytes\":" << worker.standing_gpu_bytes << ",\"allowed_kinds\":[";
            bool kind_comma = false;
            for (auto const& [name, bit] : {std::pair{"generate-cpu", compute_generate_cpu}, {"generate-gpu", compute_generate_gpu}, {"encode", compute_encode}})
                if (worker.allowed_kinds & bit) { if (kind_comma) out << ','; kind_comma = true; quote(name); }
            out << "],\"reservation\":";
            write_demands(out, worker.reservation); out << '}';
        }
        out << ']';
    }
    out << ",\"cohorts\":["; bool comma = false;
    for (auto const& c : n.cohorts)
    {
        if (comma) out << ','; comma = true; out << "{\"max_members\":" << c.max_members << ",\"new_farming\":"; write_demands(out, c.new_farming);
        out << ",\"proof_reserve\":"; write_demands(out, c.proof_reserve); out << '}';
    }
    out << "]}";
}
} // namespace resource_detail
inline Profile parse_profile(profile_detail::Json const& supplied)
{
    using namespace profile_detail;
    auto const* nested = field(supplied, "farm_resources"); auto const& object = nested ? *nested : supplied;
    auto required = [&](Json const& value, char const* name) -> Json const&
    { auto const* result = field(value, name); resource_detail::require(result, "missing mixed farm profile field"); return *result; };
    resource_detail::require(integer(required(object, "schema")) == resource_schema, "unsupported mixed farm resource schema");
    auto string = [&](Json const& value, char const* name) { auto const* item = field(value, name); return item ? text(*item) : std::string{}; };
    auto number = [&](Json const& value, char const* name) { auto const* item = field(value, name); return item ? integer(*item) : uint64_t(0); };
    auto flag = [&](Json const& value, char const* name) { auto const* item = field(value, name); return item && boolean(*item); };
    auto array = [&](Json const& value, char const* name, size_t bound) -> std::vector<Json> const&
    {
        auto const& result = required(value, name);
        resource_detail::require(result.kind == Json::array && result.items.size() <= bound, "invalid mixed farm profile array");
        return result.items;
    };
    Profile profile; profile.source = string(object, "source"); profile.captured_at = string(object, "captured_at"); profile.producer = string(object, "producer");
    if (auto const* policy = field(object, "gate"))
    {
        if (auto const* item = field(*policy, "max_age_ms")) profile.gate.max_age_ms = integer(*item);
        if (auto const* item = field(*policy, "recovery_hold_ms")) profile.gate.recovery_hold_ms = integer(*item);
        if (auto const* item = field(*policy, "recovery_samples")) profile.gate.recovery_samples = integer(*item);
        if (auto const* item = field(*policy, "max_cpu_busy_permille")) profile.gate.max_cpu_busy_permille = integer(*item);
    }
    if (auto const* native = field(object, "native")) profile.native = resource_detail::parse_native(*native);
    for (auto const& item : array(object, "resources", max_resources))
    {
        Resource r; r.id = string(item, "id"); r.host = string(item, "host"); r.kind = resource_detail::kind(string(item, "kind"));
        r.capacity = number(item, "capacity"); r.capacity_known = flag(item, "capacity_known");
        r.conditional_shared = flag(item, "conditional_shared"); r.telemetry_required = flag(item, "telemetry_required");
        for (auto const& alias : array(item, "aliases", 32)) r.aliases.push_back(text(alias));
        profile.resources.push_back(std::move(r));
    }
    for (auto const& item : array(object, "workloads", max_workloads))
    {
        Workload w;
        w.id = string(item, "id"); w.format = string(item, "format"); w.decoder = string(item, "decoder");
        w.decoder_version = string(item, "decoder_version"); w.backend = string(item, "backend"); w.pool_difficulty = string(item, "pool_difficulty");
        w.source = string(item, "source"); w.captured_at = string(item, "captured_at");
        w.external_service_node_id = string(item, "external_service_node_id"); w.external_service_host = string(item, "external_service_host");
        w.external_service_members = number(item, "external_service_members");
        auto const cost = string(item, "cost");
        resource_detail::require(cost == "unknown" || cost == "reservation" || cost == "serialized", "unsupported mixed farm workload cost kind");
        w.cost = cost == "reservation" ? CostKind::reservation : cost == "serialized" ? CostKind::serialized : CostKind::unknown;
        w.bounded = flag(item, "bounded"); w.scope_complete = flag(item, "scope_complete");
        w.plot_count = number(item, "plot_count"); w.quality_burst = number(item, "quality_burst"); w.proof_burst = number(item, "proof_burst");
        w.future_quality_burst = number(item, "future_quality_burst"); w.future_proof_burst = number(item, "future_proof_burst");
        w.filter_bits = number(item, "filter_bits"); w.future_filter_bits = number(item, "future_filter_bits");
        w.period_us = number(item, "period_us"); w.deadline_us = number(item, "deadline_us"); w.response_tail_us = number(item, "response_tail_us");
        for (auto const& entry : array(item, "uses", max_resources))
        {
            Use u; u.resource = string(entry, "resource");
            u.resident_units = number(entry, "resident_units"); u.transient_units = number(entry, "transient_units"); u.reserved_units = number(entry, "reserved_units");
            u.required_units = number(entry, "required_units"); u.quality_us = number(entry, "quality_us"); u.proof_us = number(entry, "proof_us");
            u.blocking_us = number(entry, "blocking_us"); w.uses.push_back(std::move(u));
        }
        profile.workloads.push_back(std::move(w));
    }
    resource_detail::validate(profile); return profile;
}
inline Profile read_profile(std::istream& input, uint64_t max_memory_bytes = profile_detail::parsing_memory_bytes)
{
    resource_detail::require(max_memory_bytes >= 4096, "mixed farm profile parsing grant is too small");
    auto const text = profile_detail::read_text(input, std::min(profile_byte_limit, max_memory_bytes / 8));
    auto const object = profile_detail::JsonReader(text, max_memory_bytes / 2).read();
    auto result = parse_profile(object);
    resource_detail::require(profile_memory(result) <= max_memory_bytes / 4, "mixed farm profile exceeds its retained memory grant");
    return result;
}
// A scan can emit this editable skeleton beside its inventory. Discovery fills
// identities/counts, never grants capacity or invents a decoder/proof cost.
inline Profile make_profile_template(std::vector<Resource> resources, std::vector<Remaining> const& counts,
                                     std::string const& captured_at, std::string const& producer)
{
    Profile result; result.source = "inventory-template"; result.captured_at = captured_at; result.producer = producer;
    result.resources = std::move(resources);
    for (auto& r : result.resources) { r.capacity_known = false; r.conditional_shared = false; }
    resource_detail::Graph graph(result);
    std::map<std::string, uint64_t> grouped;
    for (auto const& item : counts)
    {
        resource_detail::text(item.workload, true);
        grouped[item.workload] = resource_detail::add(grouped[item.workload], item.count);
    }
    resource_detail::require(grouped.size() <= max_workloads, "too many generated workload classes");
    for (auto const& [id, count] : grouped)
    {
        Workload w; w.id = id; w.plot_count = count; w.source = "inventory-template"; w.captured_at = captured_at;
        w.format = w.decoder = w.decoder_version = w.backend = w.pool_difficulty = "unknown";
        for (auto const& r : graph.resources) { Use use; use.resource = r.id; w.uses.push_back(std::move(use)); }
        result.workloads.push_back(std::move(w));
    }
    resource_detail::validate(result); return result;
}

struct Victim
{
    std::string id, workload, value_unit;
    uint64_t count = 1, physical_bytes = 0, eligible_capacity = 0;
    std::optional<uint64_t> forecast_value, replacement_us;
    std::string eligible_capacity_unit;
    bool eligible_capacity_known = false;
};
struct RankedVictim
{
    Victim victim;
    uint64_t released_units = 0, released_work_us = 0;
};
// Select one named bottleneck for comparison; do not add CPU time, network
// time, bytes, or vendor-specific reward units into an invented universal score.
// Unknown monetary/production inputs remain unknown and do not receive a price.
inline std::vector<RankedVictim> rank_victims(Profile const& profile, Stage const& stage,
                                             std::vector<Victim> const& victims, std::string const& bottleneck = {})
{
    using namespace resource_detail;
    require(victims.size() <= 100000, "too many migration ranking candidates");
    Graph graph(profile); auto const baseline = assess_stage(profile, stage);
    std::optional<size_t> selected;
    if (!bottleneck.empty()) selected = graph.find(bottleneck);
    std::map<std::string, uint64_t> remaining;
    std::set<std::string> external;
    for (auto const& w : profile.workloads)
    {
        remaining[w.id] = w.plot_count;
        if (!w.external_service_node_id.empty()) external.insert(w.id);
    }
    for (auto const& item : stage.remaining) remaining[item.workload] = item.count;
    std::set<std::string> ids; std::vector<RankedVictim> result;
    std::map<std::pair<std::string, uint64_t>, std::pair<uint64_t, uint64_t>> relief_cache;
    for (auto const& v : victims)
    {
        text(v.id, true); text(v.workload, true); text(v.value_unit); text(v.eligible_capacity_unit);
        require(ids.insert(v.id).second && v.count && v.physical_bytes, "invalid migration ranking candidate");
        auto const at = remaining.find(v.workload);
        require(at != remaining.end() && v.count <= at->second, "ranking candidate exceeds remaining workload");
        require(!external.contains(v.workload), "unscanned external service workload cannot be a retirement candidate");
        if (v.forecast_value) require(!v.value_unit.empty(), "forecast value is missing its explicit unit");
        if (v.eligible_capacity_known) require(!v.eligible_capacity_unit.empty(), "eligible capacity is missing its explicit unit");
        if (v.replacement_us) require(*v.replacement_us, "replacement duration must be positive when known");
        RankedVictim ranked; ranked.victim = v;
        if (selected)
        {
            auto const key = std::pair{v.workload, v.count}; auto cached = relief_cache.find(key);
            if (cached == relief_cache.end())
            {
                Stage after = stage;
                auto item = std::find_if(after.remaining.begin(), after.remaining.end(), [&](Remaining const& x) { return x.workload == v.workload; });
                if (item == after.remaining.end()) after.remaining.push_back({v.workload, at->second - v.count});
                else item->count -= v.count;
                auto const reduced = assess_stage(profile, after);
                auto const& a = baseline.margins[*selected]; auto const& b = reduced.margins[*selected];
                cached = relief_cache.emplace(key, std::pair{a.occupied_units > b.occupied_units ? a.occupied_units - b.occupied_units : 0,
                                                           a.serialized_work_us > b.serialized_work_us ? a.serialized_work_us - b.serialized_work_us : 0}).first;
            }
            ranked.released_units = cached->second.first; ranked.released_work_us = cached->second.second;
        }
        result.push_back(std::move(ranked));
    }
    bool const comparable_value = !victims.empty() && std::all_of(victims.begin(), victims.end(), [&](Victim const& v)
    { return v.forecast_value && v.value_unit == victims.front().value_unit; });
    bool const comparable_duration = std::all_of(victims.begin(), victims.end(), [](Victim const& v) { return v.replacement_us.has_value(); });
    bool const comparable_capacity = !victims.empty() && std::all_of(victims.begin(), victims.end(), [&](Victim const& v)
    { return v.eligible_capacity_known && v.eligible_capacity_unit == victims.front().eligible_capacity_unit; });
    std::stable_sort(result.begin(), result.end(), [&](RankedVictim const& a, RankedVictim const& b)
    {
        if (a.released_units != b.released_units) return a.released_units > b.released_units;
        if (a.released_work_us != b.released_work_us) return a.released_work_us > b.released_work_us;
        auto const& x = a.victim; auto const& y = b.victim;
        if (comparable_value)
        {
            long double const left = static_cast<long double>(*x.forecast_value) / x.physical_bytes;
            long double const right = static_cast<long double>(*y.forecast_value) / y.physical_bytes;
            if (left != right) return left < right;
        }
        if (comparable_duration && x.replacement_us != y.replacement_us) return *x.replacement_us < *y.replacement_us;
        if (x.physical_bytes != y.physical_bytes) return x.physical_bytes > y.physical_bytes;
        if (comparable_capacity && x.eligible_capacity != y.eligible_capacity) return x.eligible_capacity < y.eligible_capacity;
        return x.id < y.id;
    });
    return result;
}
inline void write_profile_json(std::ostream& out, Profile const& profile)
{
    resource_detail::validate(profile);
    auto quote = [&](std::string const& s) { pos2gpu::write_json_string(out, s); };
    out << "{\"schema\":" << resource_schema << ",\"source\":"; quote(profile.source);
    out << ",\"captured_at\":"; quote(profile.captured_at); out << ",\"producer\":"; quote(profile.producer);
    out << ",\"gate\":{\"max_age_ms\":" << profile.gate.max_age_ms << ",\"recovery_hold_ms\":" << profile.gate.recovery_hold_ms
        << ",\"recovery_samples\":" << profile.gate.recovery_samples << ",\"max_cpu_busy_permille\":" << profile.gate.max_cpu_busy_permille << "},\"resources\":[";
    bool comma = false;
    for (auto const& r : profile.resources)
    {
        if (comma) out << ','; comma = true; out << "{\"id\":"; quote(r.id); out << ",\"host\":"; quote(r.host);
        out << ",\"kind\":"; quote(resource_detail::name(r.kind));
        out << ",\"capacity\":" << r.capacity << ",\"capacity_known\":" << (r.capacity_known ? "true" : "false")
            << ",\"conditional_shared\":" << (r.conditional_shared ? "true" : "false")
            << ",\"telemetry_required\":" << (r.telemetry_required ? "true" : "false") << ",\"aliases\":[";
        bool alias_comma = false; for (auto const& alias : r.aliases) { if (alias_comma) out << ','; alias_comma = true; quote(alias); }
        out << "]}";
    }
    out << "],\"workloads\":["; comma = false;
    for (auto const& w : profile.workloads)
    {
        if (comma) out << ','; comma = true; out << "{\"id\":"; quote(w.id); out << ",\"format\":"; quote(w.format);
        out << ",\"decoder\":"; quote(w.decoder); out << ",\"decoder_version\":"; quote(w.decoder_version);
        out << ",\"backend\":"; quote(w.backend); out << ",\"pool_difficulty\":"; quote(w.pool_difficulty);
        out << ",\"source\":"; quote(w.source); out << ",\"captured_at\":"; quote(w.captured_at);
        if (!w.external_service_node_id.empty())
        {
            out << ",\"external_service_node_id\":"; quote(w.external_service_node_id);
            out << ",\"external_service_host\":"; quote(w.external_service_host);
            out << ",\"external_service_members\":" << w.external_service_members;
        }
        out << ",\"cost\":"; quote(w.cost == CostKind::reservation ? "reservation" : w.cost == CostKind::serialized ? "serialized" : "unknown");
        out << ",\"bounded\":" << (w.bounded ? "true" : "false") << ",\"scope_complete\":" << (w.scope_complete ? "true" : "false")
            << ",\"plot_count\":" << w.plot_count << ",\"quality_burst\":" << w.quality_burst << ",\"proof_burst\":" << w.proof_burst
            << ",\"future_quality_burst\":" << w.future_quality_burst << ",\"future_proof_burst\":" << w.future_proof_burst
            << ",\"filter_bits\":" << w.filter_bits << ",\"future_filter_bits\":" << w.future_filter_bits
            << ",\"period_us\":" << w.period_us << ",\"deadline_us\":" << w.deadline_us << ",\"response_tail_us\":" << w.response_tail_us << ",\"uses\":[";
        bool use_comma = false;
        for (auto const& u : w.uses)
        {
            if (use_comma) out << ','; use_comma = true; out << "{\"resource\":"; quote(u.resource);
            out << ",\"resident_units\":" << u.resident_units << ",\"transient_units\":" << u.transient_units
                << ",\"reserved_units\":" << u.reserved_units << ",\"required_units\":" << u.required_units
                << ",\"quality_us\":" << u.quality_us << ",\"proof_us\":" << u.proof_us << ",\"blocking_us\":" << u.blocking_us << '}';
        }
        out << "]}";
    }
    out << "],\"native\":"; resource_detail::write_native(out, profile.native); out << '}';
}
inline void write_assessment_json(std::ostream& out, Assessment const& result)
{
    auto quote = [&](std::string const& s) { pos2gpu::write_json_string(out, s); };
    out << "{\"feasible\":" << (result.feasible ? "true" : "false")
        << ",\"assessment\":\"conditional-declared-envelopes\",\"foreign_queues_controlled\":false,\"gpu_preemption_guaranteed\":false"
        << ",\"conditional_shared_gpu\":" << (result.conditional_shared_gpu ? "true" : "false")
        << ",\"conditional_shared_service\":" << (result.conditional_shared_service ? "true" : "false")
        << ",\"per_dimension_worst_across_stages\":" << (result.per_dimension_worst ? "true" : "false") << ",\"reasons\":[";
    bool comma = false;
    for (auto const& reason : result.reasons) { if (comma) out << ','; comma = true; quote(reason); }
    out << "],\"resources\":["; comma = false;
    for (auto const& m : result.margins)
    {
        if (comma) out << ','; comma = true; out << "{\"resource\":"; quote(m.resource);
        out << ",\"kind\":"; quote(resource_detail::name(m.kind));
        out << ",\"capacity\":" << m.capacity << ",\"resident_units\":" << m.resident_units << ",\"transient_units\":" << m.transient_units
            << ",\"reserved_units\":" << m.reserved_units << ",\"required_units\":" << m.required_units << ",\"request_units\":" << m.request_units
            << ",\"occupied_units\":" << m.occupied_units << ",\"remaining_units\":" << m.remaining_units
            << ",\"serialized_work_us\":" << m.serialized_work_us << ",\"blocking_us\":" << m.blocking_us << ",\"drain_us\":" << m.drain_us
            << ",\"period_us\":" << m.period_us << ",\"deadline_us\":" << m.deadline_us << ",\"worst_stages\":{";
        bool stage_comma = false;
        for (auto const& [field, stage] : m.worst_stages) { if (stage_comma) out << ','; stage_comma = true; quote(field); out << ':'; quote(stage); }
        out << "}}";
    }
    out << "]}";
}

// Bounded reuse for one immutable, borrowed profile. Only feasible normalized
// envelopes are cached; positive cohort counts have identical retained burst/
// residency costs. Stage names are restored for every caller's exact report.
class AssessmentCache
{
    AssessmentPreparation preparation_;
    std::vector<std::pair<std::string, Assessment>> cache_;
    static uint64_t assessment_memory(Assessment const& value)
    {
        using namespace resource_detail;
        uint64_t bytes = add(multiply(value.reasons.capacity(), sizeof(std::string)), multiply(value.margins.capacity(), sizeof(Margin)));
        auto string = [&](std::string const& s) { bytes = add(bytes, add(s.capacity(), 1)); };
        for (auto const& reason : value.reasons) string(reason);
        for (auto const& margin : value.margins)
        {
            string(margin.resource);
            for (auto const& [field, stage] : margin.worst_stages)
            { bytes = add(bytes, sizeof(decltype(margin.worst_stages)::value_type) + 64); string(field); string(stage); }
        }
        return bytes;
    }
    std::string key(Stage const& stage) const
    {
        using namespace resource_detail;
        text(stage.name, true); require(stage.remaining.size() <= max_workloads, "too many streaming workload partitions");
        std::map<std::string, uint64_t> remaining;
        for (auto const& r : stage.remaining) require(remaining.emplace(r.workload, r.count).second, "duplicate streaming workload partition");
        std::ostringstream out; out << stage.future_filter << stage.scope_complete << ':';
        for (auto const& w : preparation_.profile().workloads)
        {
            auto at = remaining.find(w.id); auto const count = at == remaining.end() ? w.plot_count : at->second;
            require(count <= w.plot_count, "streaming stage increases existing workload count");
            require(w.external_service_node_id.empty() || count == w.plot_count,
                "streaming migration cannot reduce an unscanned external service reservation");
            out << (count != 0); if (at != remaining.end()) remaining.erase(at);
        }
        require(remaining.empty(), "streaming stage names an unknown workload");
        uint64_t bytes = 0;
        for (auto const* values : {&stage.new_farming, &stage.proof_reserve, &stage.construction, &stage.verification, &stage.supervision, &stage.background})
        {
            require(values->size() <= max_resources * 4, "too many streaming native demands");
            for (auto const& d : *values) { bytes = resource_detail::add(bytes, resource_detail::add(d.resource.size(), 256)); if (bytes > 65536) return {}; }
            resource_detail::write_demands(out, *values);
        }
        auto result = out.str(); return result.size() <= 65536 ? result : std::string{};
    }
public:
    // One allowance covers preparation, retained keys/results, normalized
    // stage copies, current assessment and provenance rewriting. Retention
    // uses at most half, leaving the other half for the bounded working set.
    static constexpr uint64_t memory_bytes = 16 * 1024 * 1024;
    explicit AssessmentCache(Profile const& profile) : preparation_(profile)
    {
        cache_.reserve(8);
        resource_detail::require(retained_memory_bytes() <= memory_bytes / 2, "mixed farm assessment preparation exceeds its memory allowance");
    }
    AssessmentPreparation const& preparation() const { return preparation_; }
    uint64_t retained_memory_bytes() const
    {
        using namespace resource_detail;
        uint64_t bytes = add(preparation_.memory_bytes(), add(sizeof(cache_), multiply(cache_.capacity(), sizeof(decltype(cache_)::value_type))));
        for (auto const& [identity, value] : cache_)
            bytes = add(bytes, add(add(identity.capacity(), 1), assessment_memory(value)));
        return bytes;
    }
    Assessment assess(Stage const& stage)
    {
        auto const identity = key(stage); Assessment current;
        auto at = std::find_if(cache_.begin(), cache_.end(), [&](auto const& entry) { return !identity.empty() && entry.first == identity; });
        if (at == cache_.end())
        {
            Stage normalized = stage; normalized.name = "resource-state"; current = assess_stage(preparation_, normalized);
            if (!identity.empty() && current.feasible)
            {
                if (cache_.size() == 8) cache_.erase(cache_.begin());
                uint64_t const bytes = resource_detail::add(resource_detail::add(identity.size(), 1), assessment_memory(current));
                if (bytes <= memory_bytes / 2 - retained_memory_bytes())
                {
                    cache_.emplace_back(identity, current);
                    if (retained_memory_bytes() > memory_bytes / 2) cache_.pop_back();
                }
            }
        }
        else current = at->second;
        for (auto& margin : current.margins) for (auto& [field, provenance] : margin.worst_stages) provenance = stage.name;
        for (auto& reason : current.reasons)
            if (reason.starts_with("response-deadline-exceeded:") || reason.starts_with("incomplete-stage-resource-scope:"))
                reason = reason.substr(0, reason.find(':') + 1) + stage.name;
        return current;
    }
};
// Bounded streaming aggregation shares its cache with related feasibility
// checks, rather than retaining another graph and result set for those stages.
class AssessmentAccumulator
{
    AssessmentCache cache_;
    Assessment result_;
    bool present_ = false, final_future_ = false;
public:
    static constexpr uint64_t memory_bytes = AssessmentCache::memory_bytes;
    explicit AssessmentAccumulator(Profile const& profile) : cache_(profile) {}
    AssessmentCache& cache() { return cache_; }
    void add(Stage const& stage)
    {
        auto current = cache_.assess(stage);
        merge_stage_assessment(result_, std::move(current), stage.name); present_ = true; final_future_ = stage.future_filter;
    }
    Assessment finish() const
    {
        resource_detail::require(present_, "empty streaming migration assessment");
        auto result = result_;
        if (!final_future_) resource_detail::fail(result, "missing-final-future-filter-assessment");
        return result;
    }
};

struct ResourceSample
{
    std::string resource;
    bool available_known = false, cpu_busy_known = false, queue_tail_known = false, missed_deadline = false;
    uint64_t available_units = 0, cpu_busy_permille = 0, queue_tail_us = 0;
};
// Runtime-only protocol, deliberately not accepted by the public JSON parser.
// epoch is a local collector/route-selection generation. producer_tick_ms must
// advance with sequence; bounded_source_age_ms must come from an actual bounded
// read of current counters or an authenticated source with a bounded age, not
// from receipt time, file mtime, or an imported wall-clock timestamp.
struct Telemetry
{
    uint64_t schema = 1, epoch = 0, sequence = 0, producer_tick_ms = 0;
    std::string producer;
    std::optional<uint64_t> bounded_source_age_ms;
    bool recovery_observation = true;
    std::vector<ResourceSample> resources;
};
// These are current observations of this process and its held proof child,
// never imported telemetry or credit for a foreign farming/compute service.
struct OwnMemoryCredit
{
    uint64_t parent_resident_bytes = 0, parent_limit_bytes = 0;
    uint64_t child_resident_bytes = 0, child_limit_bytes = 0;
};
struct Admission
{
    Assessment assessment;
    bool allowed = false, paused = false;
};
class AdmissionGate
{
    struct Seen
    {
        uint64_t epoch = 0, sequence = 0, tick = 0;
        std::string producer;
        Clock::time_point received;
        uint64_t source_age_ms = 0;
        ResourceSample sample;
    };
    Profile profile_;
    resource_detail::Graph graph_;
    std::map<size_t, Seen> samples_;
    uint64_t observations_ = 0, counted_observation_ = 0, safe_samples_ = 0;
    bool paused_ = false;
    std::optional<Clock::time_point> safe_since_, last_evaluated_;
public:
    explicit AdmissionGate(Profile profile) : profile_(std::move(profile)), graph_(profile_)
    { resource_detail::validate(profile_); }
    void pause()
    { paused_ = true; safe_samples_ = 0; safe_since_.reset(); counted_observation_ = observations_; }
    bool observe(Telemetry const& sample, Clock::time_point now = Clock::now())
    {
        using namespace resource_detail;
        if (sample.schema != 1 || !sample.epoch || !sample.sequence || sample.producer.empty() || sample.producer.size() > 1024 ||
            !sample.bounded_source_age_ms || *sample.bounded_source_age_ms > profile_.gate.max_age_ms ||
            sample.resources.empty() || sample.resources.size() > max_resources) return false;
        std::set<size_t> seen;
        for (auto const& item : sample.resources)
        {
            size_t const i = graph_.find(item.resource);
            if (!seen.insert(i).second || item.cpu_busy_permille > 1000) return false;
            auto at = samples_.find(i);
            if (at == samples_.end()) continue;
            auto const& old = at->second;
            if (now < old.received || sample.epoch < old.epoch ||
                (sample.epoch == old.epoch && (sample.producer != old.producer || sample.sequence <= old.sequence || sample.producer_tick_ms <= old.tick))) return false;
        }
        for (auto const& item : sample.resources)
        {
            size_t const i = graph_.find(item.resource);
            samples_[i] = {sample.epoch, sample.sequence, sample.producer_tick_ms, sample.producer,
                           now, *sample.bounded_source_age_ms, item};
        }
        if (sample.recovery_observation) observations_ = add(observations_, 1);
        return true;
    }
    Admission evaluate_owned(Stage const& stage, Clock::time_point now, OwnMemoryCredit const& credit)
    {
        using namespace resource_detail;
        Admission result; result.assessment = assess_stage(profile_, stage);
        if (last_evaluated_ && now < *last_evaluated_) fail(result.assessment, "monotonic-observation-clock-regressed");
        last_evaluated_ = now;
        std::vector<uint64_t> incremental(graph_.resources.size());
        // Free-memory readings already exclude currently resident foreign
        // processes. Preserve their declared transient burst, then add all
        // native/candidate allocations, which may not yet be resident.
        for (auto const& w : profile_.workloads)
        {
            for (auto const& use : w.uses)
            {
                size_t const i = graph_.find(use.resource);
                if (memory(graph_.resources[i].kind)) incremental[i] = add(incremental[i], use.transient_units);
            }
        }
        auto add_demands = [&](std::vector<Demand> const& values)
        {
            for (auto const& demand : values)
            {
                size_t const i = graph_.find(demand.resource);
                uint64_t const units = memory(graph_.resources[i].kind) ? add(demand.resident_units, demand.transient_units) : add(demand.reserved_units, demand.required_units);
                incremental[i] = add(incremental[i], units);
            }
        };
        add_demands(stage.new_farming); add_demands(stage.proof_reserve); add_demands(stage.construction); add_demands(stage.verification);
        add_demands(stage.supervision);
        add_demands(stage.background);
        if (credit.parent_limit_bytes || credit.child_limit_bytes)
        {
            std::vector<uint64_t> own(graph_.resources.size());
            for (auto const* values : {&stage.construction, &stage.verification, &stage.supervision, &stage.background})
            for (auto const& d : *values)
            {
                size_t const i = graph_.find(d.resource);
                if (graph_.resources[i].kind == ResourceKind::host_memory && graph_.resources[i].host == profile_.native.host)
                    own[i] = add(own[i], add(d.resident_units, d.transient_units));
            }
            uint64_t const limit = add(credit.parent_limit_bytes, credit.child_limit_bytes);
            uint64_t const observed = add(std::min(credit.parent_resident_bytes, credit.parent_limit_bytes),
                std::min(credit.child_resident_bytes, credit.child_limit_bytes));
            uint64_t room = 0; for (auto bytes : own) room = add(room, bytes);
            require(limit <= room, "owned process memory exceeds the declared local phase and background demand");
            for (size_t i = 0; i < own.size(); ++i)
                incremental[i] -= std::min(own[i], observed);
            // Static capacity, protected farming/proof and foreign transient
            // demand remain unchanged. Only already-resident anonymous pages
            // belonging to these owned processes can offset their own demand.
        }
        for (size_t i = 0; i < graph_.resources.size(); ++i)
        {
            auto const& resource = graph_.resources[i]; auto at = samples_.find(i);
            if (at == samples_.end())
            {
                if (resource.telemetry_required) fail(result.assessment, "missing-fresh-telemetry:" + resource.id);
                continue; // Complete declared reservation remains in force.
            }
            auto const& seen = at->second;
            bool const regressed = now < seen.received;
            auto const elapsed = regressed ? 0 : std::chrono::duration_cast<std::chrono::milliseconds>(now - seen.received).count();
            if (regressed || uint64_t(elapsed) > profile_.gate.max_age_ms - seen.source_age_ms)
            { fail(result.assessment, "stale-telemetry:" + resource.id); continue; }
            auto const& reading = seen.sample;
            if (!reading.available_known && !reading.cpu_busy_known && !reading.queue_tail_known && !reading.missed_deadline)
                fail(result.assessment, "unavailable-telemetry:" + resource.id);
            if (reading.available_known && incremental[i] > reading.available_units)
                fail(result.assessment, "observed-headroom-exceeded:" + resource.id);
            if (reading.cpu_busy_known && reading.cpu_busy_permille >= profile_.gate.max_cpu_busy_permille)
                fail(result.assessment, "observed-cpu-pressure:" + resource.id);
            if (reading.missed_deadline) fail(result.assessment, "observed-farming-deadline-miss:" + resource.id);
            auto const& margin = result.assessment.margins[i];
            if (reading.queue_tail_known)
            {
                uint64_t const response = add(margin.drain_us, reading.queue_tail_us);
                if ((margin.deadline_us && response > margin.deadline_us) || (margin.period_us && response > margin.period_us))
                    fail(result.assessment, "observed-queue-tail-exceeded:" + resource.id);
            }
        }
        if (!result.assessment.feasible)
        {
            paused_ = true; safe_samples_ = 0; safe_since_.reset(); counted_observation_ = observations_;
        }
        else if (paused_)
        {
            if (!safe_since_) safe_since_ = now;
            if (observations_ != counted_observation_)
            { safe_samples_ = std::min(profile_.gate.recovery_samples, safe_samples_ + 1); counted_observation_ = observations_; }
            auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - *safe_since_).count();
            if (safe_samples_ >= profile_.gate.recovery_samples && uint64_t(elapsed) >= profile_.gate.recovery_hold_ms) paused_ = false;
        }
        result.paused = paused_; result.allowed = result.assessment.feasible && !paused_; return result;
    }
    Admission evaluate(Stage const& stage, Clock::time_point now = Clock::now(),
                       std::optional<uint64_t> active_own_anon_bytes = {})
    {
        OwnMemoryCredit credit;
        if (active_own_anon_bytes)
        {
            resource_detail::require(!stage.construction.empty() && stage.verification.empty() && stage.supervision.empty(),
                "active own-memory credit requires one running construction phase");
            credit.parent_resident_bytes = *active_own_anon_bytes;
            for (auto const& d : stage.construction)
            {
                auto const& r = graph_.resources[graph_.find(d.resource)];
                if (r.kind == ResourceKind::host_memory && r.host == profile_.native.host)
                    credit.parent_limit_bytes = resource_detail::add(credit.parent_limit_bytes, resource_detail::add(d.resident_units, d.transient_units));
            }
        }
        return evaluate_owned(stage, now, credit);
    }
};

// Borrowed only during run_worker's observation callback. The supervisor owns
// and has not reaped/closed this process; generation distinguishes retry epochs.
struct SupervisedProcess
{
    uint64_t generation = 0;
#ifdef _WIN32
    HANDLE handle = nullptr;
    bool exited() const { return handle && ::WaitForSingleObject(handle, 0) == WAIT_OBJECT_0; }
#elif defined(__linux__)
    pid_t pid = 0;
    bool exited() const
    {
        siginfo_t info{};
        return pid > 0 && !::waitid(P_PID, id_t(pid), &info, WEXITED | WNOHANG | WNOWAIT) && info.si_pid == pid;
    }
#else
    bool exited() const { return false; }
#endif
};
class ProcessObservationUnavailable : public std::runtime_error
{
public:
    ProcessObservationUnavailable() : std::runtime_error("held proof process counters or affinity are unavailable") {}
};
namespace resource_detail
{
struct CpuObservation
{
    uint64_t total = 0, idle = 0, units_per_second = 0, processors = 0;
    std::string scope;
    Clock::time_point at, before_wall, after_wall;
    std::optional<uint64_t> own_before_ns, own_after_ns;
    uint64_t own_counter_padding_ns = 0;
    bool own_scope_known = false;
};
inline uint64_t busy_permille(uint64_t busy, uint64_t total)
{ require(total && busy <= total, "invalid CPU pressure counters"); return multiply(busy, 1000) / total + (multiply(busy, 1000) % total != 0); }
inline std::optional<uint64_t> active_cpu_pressure(uint64_t total, uint64_t idle, uint64_t own_ns,
    uint64_t interior_ns, uint64_t lanes, uint64_t units, uint64_t processors)
{
    if (!total || idle > total || !lanes || !units || units > 1000000000 || !processors) return {};
    // Only the interior process-clock interval is credited. User/system tick
    // conversion can each round at a boundary; retain two ticks per CPU as
    // uncertainty rather than subtracting another process's busy time.
    uint64_t const tick_ns = 1000000000 / units + (1000000000 % units != 0);
    uint64_t const uncertainty = multiply(multiply(tick_ns, processors), 2);
    if (own_ns > add(multiply(interior_ns, processors), uncertainty)) return {};
    uint64_t credit_ns = own_ns > uncertainty ? own_ns - uncertainty : 0;
    credit_ns = std::min(credit_ns, multiply(interior_ns, lanes));
    uint64_t const ticks = add(multiply(credit_ns / 1000000000, units), multiply(credit_ns % 1000000000, units) / 1000000000);
    return busy_permille(total - idle - std::min(total - idle, ticks), total);
}
class CpuPressureHistory
{
    std::optional<CpuObservation> baseline_;
    struct Interval
    {
        uint64_t total, idle, own_ns, interior_ns, units, processors;
        Clock::time_point at;
        bool own_known;
        std::string scope;
    };
    std::optional<Interval> interval_;
    uint64_t generation_ = 0;
public:
    uint64_t generation() const { return generation_; }
    void observe(std::optional<CpuObservation> const& current, uint64_t max_age_ms)
    {
        if (!current) { baseline_.reset(); interval_.reset(); return; }
        auto reset = [&] { baseline_ = current; interval_.reset(); };
        if (!baseline_) { reset(); return; }
        auto const& old = *baseline_; auto const& value = *current;
        if (value.at < old.at || value.scope != old.scope || value.units_per_second != old.units_per_second || value.processors != old.processors ||
            value.total < old.total || value.idle < old.idle)
        { reset(); return; }
        auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(value.at - old.at).count();
        if (uint64_t(elapsed) > max_age_ms) { reset(); return; }
        uint64_t const total = value.total - old.total, idle = value.idle - old.idle;
        if (idle > total) { reset(); return; }
        uint64_t const minimum_ms = std::max<uint64_t>(1, std::min<uint64_t>(50, max_age_ms / 2));
        // Back-to-back current/future checks must not destroy a good baseline
        // or manufacture a fresh CPU observation before counters advance.
        if (!total || uint64_t(elapsed) < minimum_ms) return;
        bool const known = old.own_scope_known && value.own_scope_known && old.own_after_ns && value.own_before_ns &&
            *value.own_before_ns >= *old.own_after_ns && value.before_wall >= old.after_wall;
        uint64_t const raw_own = known ? *value.own_before_ns - *old.own_after_ns : 0;
        uint64_t const own = raw_own > old.own_counter_padding_ns ? raw_own - old.own_counter_padding_ns : 0;
        auto const interior = known ? std::chrono::duration_cast<std::chrono::nanoseconds>(value.before_wall - old.after_wall).count() : 0;
        interval_ = Interval{total, idle, own, uint64_t(interior), value.units_per_second, value.processors, value.at, known, value.scope};
        generation_ = add(generation_, 1);
        baseline_ = current;
    }
    std::optional<std::pair<uint64_t, uint64_t>> reading(Clock::time_point now, uint64_t max_age_ms,
        std::string const& scope, uint64_t active_lanes, bool own_scope_known) const
    {
        if (!interval_ || now < interval_->at || scope != interval_->scope) return {};
        auto const age = std::chrono::duration_cast<std::chrono::milliseconds>(now - interval_->at).count();
        if (uint64_t(age) > max_age_ms) return {};
        std::optional<uint64_t> busy;
        if (!active_lanes) busy = busy_permille(interval_->total - interval_->idle, interval_->total);
        else if (interval_->own_known && own_scope_known)
            busy = active_cpu_pressure(interval_->total, interval_->idle, interval_->own_ns, interval_->interior_ns,
                                       active_lanes, interval_->units, interval_->processors);
        if (!busy) return {};
        return std::pair{*busy, uint64_t(age)};
    }
};
#ifdef _WIN32
inline std::optional<uint64_t> process_cpu_ns(HANDLE process = ::GetCurrentProcess())
{
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!::GetProcessTimes(process, &created, &exited, &kernel, &user)) return {};
    auto number = [](FILETIME value) { return (uint64_t(value.dwHighDateTime) << 32) | value.dwLowDateTime; };
    return multiply(add(number(kernel), number(user)), 100);
}
inline std::optional<uint64_t> private_working_set(HANDLE process = ::GetCurrentProcess())
{
    // PROCESS_MEMORY_COUNTERS_EX2 is supported by the Windows releases with
    // the September 2023 cumulative update. Keep its documented layout here
    // because older SDKs expose only EX. An older OS that fills only EX leaves
    // the sentinel unchanged and cannot grant resident-memory credit.
    struct Counters
    {
        PROCESS_MEMORY_COUNTERS_EX base;
        SIZE_T private_working_set;
        ULONG64 shared_commit;
    } value{};
    value.base.cb = sizeof(value); value.private_working_set = SIZE_MAX;
    if (!::K32GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&value), sizeof(value)) ||
        value.private_working_set == SIZE_MAX || value.private_working_set > value.base.WorkingSetSize ||
        value.private_working_set > value.base.PrivateUsage) return {};
    return uint64_t(value.private_working_set);
}
inline bool process_affinity_within(DWORD_PTR allowed)
{
    DWORD_PTR actual = 0, system = 0;
    return ::GetActiveProcessorGroupCount() == 1 &&
        ::GetProcessAffinityMask(::GetCurrentProcess(), &actual, &system) && actual && !(actual & ~allowed);
}
inline std::optional<std::string> proof_affinity(SupervisedProcess const& process, DWORD_PTR allowed)
{
    DWORD_PTR actual = 0, system = 0;
    if (!process.handle || !process.generation || ::GetActiveProcessorGroupCount() != 1 ||
        !::GetProcessAffinityMask(process.handle, &actual, &system) || !actual || (actual & ~allowed)) return {};
    return std::to_string(actual);
}
#elif defined(__linux__)
inline std::optional<uint64_t> process_cpu_ns()
{
    timespec value{};
    if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) || value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000) return {};
    return add(multiply(uint64_t(value.tv_sec), 1000000000), uint64_t(value.tv_nsec));
}
inline std::optional<std::string> process_affinity_scope(cpu_set_t const& allowed, pid_t process)
{
    auto const path = "/proc/" + std::to_string(process) + "/task";
    DIR* directory = ::opendir(path.c_str()); if (!directory) return {};
    struct Close { DIR* value; ~Close() { ::closedir(value); } } close{directory};
    size_t count = 0; cpu_set_t scope; CPU_ZERO(&scope);
    for (;;)
    {
        errno = 0; auto const* entry = ::readdir(directory);
        if (!entry)
        {
            if (errno || !count || !CPU_COUNT(&scope)) return {};
            std::string result;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &scope)) result += std::to_string(cpu) + ',';
            return result;
        }
        std::string_view name(entry->d_name); if (name == "." || name == "..") continue;
        if (++count > 4096) return {};
        pid_t id = 0; auto const parsed = std::from_chars(name.data(), name.data() + name.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() || id <= 0) return {};
        cpu_set_t actual; CPU_ZERO(&actual);
        if (::sched_getaffinity(id, sizeof(actual), &actual)) { if (errno == ESRCH) continue; return {}; }
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &actual))
        { if (!CPU_ISSET(cpu, &allowed)) return {}; CPU_SET(cpu, &scope); }
    }
}
inline bool process_affinity_within(cpu_set_t const& allowed)
{ return process_affinity_scope(allowed, ::getpid()).has_value(); }
inline std::optional<std::string> proof_affinity(SupervisedProcess const& process, cpu_set_t const& allowed)
{ return process.pid > 0 && process.generation ? process_affinity_scope(allowed, process.pid) : std::optional<std::string>{}; }
inline std::optional<uint64_t> proof_cpu_ns(SupervisedProcess const& process)
{
    clockid_t clock; timespec value{};
    if (process.pid <= 0 || ::clock_getcpuclockid(process.pid, &clock) || ::clock_gettime(clock, &value) ||
        value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000) return {};
    return add(multiply(uint64_t(value.tv_sec), 1000000000), uint64_t(value.tv_nsec));
}
inline std::optional<uint64_t> kib_field(std::string const& path, std::string_view field)
{
    std::ifstream input(path); if (!input) return {};
    auto const text = profile_detail::read_text(input, 65536); std::istringstream lines(text); std::string line;
    while (std::getline(lines, line)) if (line.starts_with(field))
    {
        std::istringstream fields(line.substr(field.size())); uint64_t value; std::string unit, extra;
        if (fields >> value >> unit && unit == "kB" && !(fields >> extra)) return multiply(value, 1024);
        return {};
    }
    return {};
}
#endif
} // namespace resource_detail

// Current local OS counters are a pressure signal, never a capacity discovery
// or a spare-compute grant. GPU and remote-solver telemetry require their own
// authenticated adapter; absence cannot satisfy telemetry_required.
class LocalPressureSampler
{
    std::string host_;
    uint64_t epoch_, sequence_ = 0;
    bool active_ = false;
    std::optional<std::string> cpu_scope_;
    resource_detail::CpuPressureHistory cpu_history_;
    resource_detail::CpuPressureHistory raw_cpu_history_;
    uint64_t recovery_generation_ = 0;
    std::optional<uint64_t> own_anon_;
    std::optional<uint64_t> child_anon_;
public:
    explicit LocalPressureSampler(std::string host, uint64_t epoch = 1) : host_(std::move(host)), epoch_(epoch)
    { resource_detail::text(host_, true); resource_detail::require(epoch_, "local telemetry epoch must be positive"); }
    std::optional<uint64_t> own_anon_bytes() const { return own_anon_; }
    std::optional<uint64_t> child_anon_bytes() const { return child_anon_; }
    Telemetry sample(Profile const& profile, Clock::time_point now = Clock::now(), uint64_t active_cpu_lanes = 0,
        SupervisedProcess const* proof = nullptr, bool observe_parent_memory = false)
    {
        using namespace resource_detail;
        bool const active = active_cpu_lanes != 0;
        if (active != active_) { epoch_ = add(epoch_, 1); active_ = active; }
        Telemetry result; result.producer = "local-os-pressure-v2:" + host_; result.epoch = epoch_;
        result.sequence = sequence_ = add(sequence_, 1); own_anon_.reset(); child_anon_.reset();
        require(!proof || (active && proof->generation), "proof pressure requires a live supervised CPU scope");
        auto const tick = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        require(tick >= 0, "local monotonic telemetry clock is negative"); result.producer_tick_ms = uint64_t(tick);
        std::optional<uint64_t> available;
        std::optional<CpuObservation> cpu;
        std::string cpu_scope; bool own_scope = false;
#ifdef _WIN32
        auto const private_before = active || observe_parent_memory ? private_working_set() : std::optional<uint64_t>{};
        auto const child_before = proof ? private_working_set(proof->handle) : std::optional<uint64_t>{};
        MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory)) available = memory.ullAvailPhys;
        if (active || observe_parent_memory)
        {
            auto const private_after = private_working_set();
            if (private_before && private_after) own_anon_ = std::min(*private_before, *private_after);
        }
        if (proof)
        {
            auto const child_after = private_working_set(proof->handle);
            if (!child_before || !child_after) throw ProcessObservationUnavailable();
            child_anon_ = std::min(*child_before, *child_after);
        }
        DWORD_PTR allowed = 0, system = 0;
        DWORD const count = ::GetActiveProcessorCount(0);
        if (::GetActiveProcessorGroupCount() == 1 && count && count <= sizeof(DWORD_PTR) * CHAR_BIT &&
            ::GetProcessAffinityMask(::GetCurrentProcess(), &allowed, &system) && allowed)
        {
            // GetSystemTimes aggregates CPUs outside a restricted affinity.
            // Query the documented per-processor records dynamically and use
            // only the known single group's selected processors. Unsupported
            // layouts/counters remain unknown and cannot satisfy admission.
            using Query = NTSTATUS (NTAPI*)(SYSTEM_INFORMATION_CLASS, PVOID, ULONG, PULONG);
            auto const module = ::GetModuleHandleW(L"ntdll.dll");
            auto const query = module ? reinterpret_cast<Query>(::GetProcAddress(module, "NtQuerySystemInformation")) : nullptr;
            std::array<SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION, sizeof(DWORD_PTR) * CHAR_BIT> records{};
            ULONG returned = 0;
            CpuObservation value; value.at = now; value.processors = uint64_t(std::popcount(allowed));
            value.units_per_second = 10000000;
            auto const child_scope_before = proof ? proof_affinity(*proof, allowed) : std::optional<std::string>{};
            value.before_wall = Clock::now(); value.own_before_ns = process_cpu_ns();
            auto const child_cpu_before = proof ? process_cpu_ns(proof->handle) : std::optional<uint64_t>{};
            bool valid = process_affinity_within(allowed) && query &&
                query(SystemProcessorPerformanceInformation, records.data(), ULONG(sizeof(records)), &returned) >= 0 &&
                returned == count * sizeof(records.front());
            for (DWORD id = 0; id < count && valid; ++id) if (allowed & (DWORD_PTR(1) << id))
            {
                auto const& record = records[id];
                if (record.IdleTime.QuadPart < 0 || record.KernelTime.QuadPart < record.IdleTime.QuadPart || record.UserTime.QuadPart < 0)
                { valid = false; break; }
                value.total = add(value.total, add(uint64_t(record.KernelTime.QuadPart), uint64_t(record.UserTime.QuadPart)));
                value.idle = add(value.idle, uint64_t(record.IdleTime.QuadPart));
                cpu_scope += std::to_string(id) + ',';
            }
            auto const child_cpu_after = proof ? process_cpu_ns(proof->handle) : std::optional<uint64_t>{};
            value.own_after_ns = process_cpu_ns(); value.after_wall = Clock::now();
            if (proof)
            {
                auto const child_scope_after = proof_affinity(*proof, allowed);
                if (!child_scope_before || child_scope_before != child_scope_after || !child_cpu_before || !child_cpu_after)
                    throw ProcessObservationUnavailable();
                if (value.own_before_ns) *value.own_before_ns = add(*value.own_before_ns, *child_cpu_before);
                if (value.own_after_ns) *value.own_after_ns = add(*value.own_after_ns, *child_cpu_after);
                value.own_counter_padding_ns = 200;
                cpu_scope += "/proof:" + std::to_string(proof->generation) + ':' + *child_scope_before;
            }
            own_scope = process_affinity_within(allowed);
            value.own_scope_known = valid && own_scope && value.own_before_ns && value.own_after_ns &&
                *value.own_after_ns >= *value.own_before_ns;
            value.scope = cpu_scope.substr(0, cpu_scope.find("/proof:"));
            if (valid) cpu = value;
        }
#elif defined(__linux__)
        // Bracket the available-memory observation. Credit only private
        // anonymous pages observed at both sides, never file RSS/page cache.
        auto const anon_before = active || observe_parent_memory ? kib_field("/proc/self/status", "RssAnon:") : std::optional<uint64_t>{};
        auto const child_path = proof ? "/proc/" + std::to_string(proof->pid) + "/status" : std::string{};
        auto const child_before = proof ? kib_field(child_path, "RssAnon:") : std::optional<uint64_t>{};
        available = kib_field("/proc/meminfo", "MemAvailable:");
        if (active || observe_parent_memory)
        {
            auto const anon_after = kib_field("/proc/self/status", "RssAnon:");
            if (anon_before && anon_after) own_anon_ = std::min(*anon_before, *anon_after);
        }
        if (proof)
        {
            auto const child_after = kib_field(child_path, "RssAnon:");
            if (!child_before || !child_after) throw ProcessObservationUnavailable();
            child_anon_ = std::min(*child_before, *child_after);
        }
        cpu_set_t allowed, found; CPU_ZERO(&allowed); CPU_ZERO(&found);
        if (!::sched_getaffinity(0, sizeof(allowed), &allowed) && CPU_COUNT(&allowed))
        {
            CpuObservation value; value.at = now; value.processors = uint64_t(CPU_COUNT(&allowed));
            long const hz = ::sysconf(_SC_CLK_TCK); if (hz > 0) value.units_per_second = uint64_t(hz);
            for (int id = 0; id < CPU_SETSIZE; ++id) if (CPU_ISSET(id, &allowed)) cpu_scope += std::to_string(id) + ',';
            value.scope = cpu_scope;
            bool const before_scope = process_affinity_within(allowed);
            auto const child_scope_before = proof ? proof_affinity(*proof, allowed) : std::optional<std::string>{};
            value.before_wall = Clock::now(); value.own_before_ns = process_cpu_ns();
            auto const child_cpu_before = proof ? proof_cpu_ns(*proof) : std::optional<uint64_t>{};
            std::ifstream input("/proc/stat"); char buffer[4096];
            for (size_t lines = 0; lines <= CPU_SETSIZE + 1 && input.getline(buffer, sizeof(buffer)); ++lines)
            {
                std::string_view line(buffer); if (!line.starts_with("cpu")) break;
                if (line.size() < 5 || line[3] == ' ') continue;
                auto const space = line.find(' ', 3); if (space == line.npos) break;
                unsigned id = 0; auto const parsed = std::from_chars(line.data() + 3, line.data() + space, id);
                if (parsed.ec != std::errc{} || parsed.ptr != line.data() + space || id >= CPU_SETSIZE || !CPU_ISSET(int(id), &allowed)) continue;
                if (CPU_ISSET(int(id), &found)) break;
                std::istringstream fields(std::string(line.substr(space))); uint64_t values[10]{}; size_t count = 0;
                while (count < 10 && fields >> values[count]) ++count;
                if (count < 4) break;
                for (size_t i = 0; i < std::min<size_t>(8, count); ++i) value.total = add(value.total, values[i]);
                value.idle = add(value.idle, add(values[3], values[4])); CPU_SET(int(id), &found);
                if (CPU_EQUAL(&allowed, &found)) { cpu = value; break; }
            }
            own_scope = before_scope && process_affinity_within(allowed);
            auto const child_cpu_after = proof ? proof_cpu_ns(*proof) : std::optional<uint64_t>{};
            value.own_after_ns = process_cpu_ns(); value.after_wall = Clock::now();
            if (proof)
            {
                auto const child_scope_after = proof_affinity(*proof, allowed);
                if (!child_scope_before || child_scope_before != child_scope_after || !child_cpu_before || !child_cpu_after)
                    throw ProcessObservationUnavailable();
                if (value.own_before_ns) *value.own_before_ns = add(*value.own_before_ns, *child_cpu_before);
                if (value.own_after_ns) *value.own_after_ns = add(*value.own_after_ns, *child_cpu_after);
                cpu_scope += "/proof:" + std::to_string(proof->generation) + ':' + *child_scope_before;
            }
            if (cpu)
            {
                cpu->own_before_ns = value.own_before_ns;
                cpu->own_after_ns = value.own_after_ns; cpu->after_wall = value.after_wall;
                cpu->own_scope_known = own_scope && cpu->own_before_ns && cpu->own_after_ns && *cpu->own_after_ns >= *cpu->own_before_ns;
            }
        }
#endif
        if (cpu_scope_ && *cpu_scope_ != cpu_scope) epoch_ = add(epoch_, 1);
        cpu_scope_ = cpu_scope; result.epoch = epoch_;
        raw_cpu_history_.observe(cpu, profile.gate.max_age_ms);
        if (cpu) cpu->scope = cpu_scope;
        cpu_history_.observe(cpu, profile.gate.max_age_ms);
        auto const collected = Clock::now();
        auto pressure = cpu_history_.reading(collected, profile.gate.max_age_ms, cpu_scope, active_cpu_lanes, own_scope);
        // A new child's CPU history needs a fresh interval. Until then retain
        // the complete observed system pressure, giving no own-work discount.
        if (!pressure && proof && cpu && cpu->own_scope_known)
            pressure = raw_cpu_history_.reading(collected, profile.gate.max_age_ms,
                cpu_scope.substr(0, cpu_scope.find("/proof:")), 0, true);
        result.recovery_observation = pressure && raw_cpu_history_.generation() != recovery_generation_;
        if (result.recovery_observation) recovery_generation_ = raw_cpu_history_.generation();
        Graph graph(profile);
        for (auto const& resource : graph.resources)
        {
            if (resource.host != host_ || (resource.kind != ResourceKind::cpu && resource.kind != ResourceKind::host_memory)) continue;
            ResourceSample reading; reading.resource = resource.id;
            if (resource.kind == ResourceKind::host_memory && available) { reading.available_known = true; reading.available_units = *available; }
            if (resource.kind == ResourceKind::cpu && pressure) { reading.cpu_busy_known = true; reading.cpu_busy_permille = pressure->first; }
            result.resources.push_back(std::move(reading));
        }
        auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - now).count();
        if (elapsed >= 0) result.bounded_source_age_ms = std::max(uint64_t(elapsed), pressure ? pressure->second : uint64_t(0));
        return result;
    }
};
} // namespace group_grow::farm
