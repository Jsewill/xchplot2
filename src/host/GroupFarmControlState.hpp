// Complete causal snapshots keep repeated controls bounded while saved peers
// are absent. A field retains concurrent values until an explicit action has
// observed and replaced them; sequence numbers never choose a silent winner.
#pragma once
#include "host/GroupFarmState.hpp"
#include <map>
#include <set>
#include <span>

namespace group_grow
{
struct FarmingControlDot
{
    std::string actor, value;
    uint64_t sequence = 0;
    bool operator==(FarmingControlDot const&) const = default;
};
struct FarmingControlState
{
    std::string job_id, fingerprint, source_policy, native_binding;
    uint64_t source_revision = 0;
    std::vector<std::string> native_members;
    std::map<std::string, uint64_t> context;
    std::map<std::string, std::vector<FarmingControlDot>> fields;
};
struct FarmingControlNativeBinding
{
    std::string source_identity, target_identity;
};
inline std::string farming_control_native_digest(std::string const& job, std::span<std::string const> members)
{
    std::ostringstream out; out << std::quoted(job) << ' ' << members.size() << '\n';
    for (auto const& identity : members) out << std::quoted(identity) << '\n';
    return stream_state::digest(out.str());
}
inline std::vector<std::string> farming_control_native_members(FarmingState const& state)
{
    std::vector<std::string> result; result.reserve(state.directories.size());
    for (auto const& directory : state.directories) result.push_back(directory.directory_identity);
    return result;
}
inline std::string farming_control_native_digest(FarmingState const& state)
{ return farming_control_native_digest(state.job_id, farming_control_native_members(state)); }
inline FarmingState farming_control_source_shape(FarmingState const& target, std::span<FarmingControlNativeBinding const> bindings)
{
    FarmingState source; source.job_id = target.job_id; source.fingerprint = target.fingerprint; source.policy_revision = target.policy_revision;
    std::map<std::string, std::string> replacements; std::set<std::string> originals;
    for (auto const& binding : bindings)
    {
        require(!binding.source_identity.empty() && binding.source_identity.size() <= 256 &&
            !binding.target_identity.empty() && binding.target_identity.size() <= 256 &&
            originals.insert(binding.source_identity).second && replacements.emplace(binding.target_identity, binding.source_identity).second &&
            std::any_of(target.directories.begin(), target.directories.end(),
                [&](auto const& directory) { return directory.directory_identity == binding.target_identity; }),
            "invalid approved farming control native handoff");
    }
    std::set<std::string> identities;
    for (auto const& directory : target.directories)
    {
        auto const replacement = replacements.find(directory.directory_identity); FarmingDirectory row;
        row.directory_identity = replacement == replacements.end() ? directory.directory_identity : replacement->second;
        require(identities.insert(row.directory_identity).second, "farming control native handoff aliases another member");
        source.directories.push_back(std::move(row));
    }
    return source;
}
inline uint64_t farming_control_size_bound(FarmingState const& state)
{
    require(state.directories.size() <= 1024, "farming control membership exceeds its bound");
    auto quoted = [](std::string const& text)
    {
        return text.size() + 2 + size_t(std::count_if(text.begin(), text.end(), [](char c) { return c == '"' || c == '\\'; }));
    };
    uint64_t actors = 0, fields = quoted("control");
    for (auto const& directory : state.directories)
    {
        require(directory.directory_identity.size() <= 256, "farming control native identity exceeds its bound");
        actors += quoted(directory.directory_identity); fields += quoted("control:" + directory.directory_identity);
    }
    uint64_t const count = state.directories.size(), digits = std::numeric_limits<uint64_t>::digits10 + 1;
    auto const dots = actors + count * (digits + quoted("draining") + 3);
    // The fixed envelope covers identities, source policy, magic and counts;
    // all repeated rows use the actual escaped native-identity lengths.
    return 4096 + actors * 2 + count * (digits + 3) + fields + (count + 1) * (digits + 2 + dots);
}
inline uint64_t farming_control_memory_bound(FarmingState const& state)
{
    // Parsing, causal merge and canonical serialization can retain separate
    // maps, expanded string rows and geometrically grown dot vectors.
    return 64 * 1024 + farming_control_size_bound(state) * 16;
}
inline void validate_farming_controls(FarmingState const& state, FarmingControlState const& value)
{
    require(!value.native_members.empty() && value.native_members.size() <= state.directories.size() &&
        std::equal(value.native_members.begin(), value.native_members.end(), state.directories.begin(),
            [](auto const& identity, auto const& directory) { return identity == directory.directory_identity; }),
        "farming controls changed a saved native member slot");
    require(value.job_id == state.job_id && value.fingerprint == state.fingerprint &&
        value.native_binding == farming_control_native_digest(value.job_id, value.native_members) &&
        value.source_revision <= state.policy_revision && value.source_policy.size() == 64 &&
        value.source_policy.find_first_not_of("0123456789abcdef") == value.source_policy.npos,
        "farming control snapshot belongs to another job or newer policy");
    require(value.context.size() <= state.directories.size() && value.fields.size() <= state.directories.size() + 1,
        "farming control snapshot exceeds its member or field bound");
    auto known = [&](std::string const& identity)
    { return std::find(value.native_members.begin(), value.native_members.end(), identity) != value.native_members.end(); };
    for (auto const& [actor, sequence] : value.context)
        require(sequence && known(actor), "farming control has a foreign actor or invalid sequence");
    for (auto const& [field, dots] : value.fields)
    {
        require(field == "control" || (field.starts_with("control:") && known(field.substr(8))), "farming control has an unknown field");
        require(!dots.empty() && dots.size() <= state.directories.size(), "farming control values exceed their native actor bound");
        std::set<std::string> actors;
        for (auto const& dot : dots)
        {
            auto const at = value.context.find(dot.actor);
            require(at != value.context.end() && dot.sequence && dot.sequence <= at->second && actors.insert(dot.actor).second &&
                farming_control_valid(dot.value), "invalid farming control causal value");
        }
    }
}
inline std::string farming_control_state_text(FarmingControlState const& value)
{
    std::ostringstream out;
    out << "ordinary-group-farming-controls-v1\n" << std::quoted(value.job_id) << ' ' << std::quoted(value.fingerprint) << ' '
        << std::quoted(value.source_policy) << ' ' << std::quoted(value.native_binding) << ' ' << value.source_revision << '\n'
        << value.native_members.size() << '\n';
    for (auto const& identity : value.native_members) out << std::quoted(identity) << '\n';
    out << value.context.size() << '\n';
    for (auto const& [actor, sequence] : value.context) out << std::quoted(actor) << ' ' << sequence << '\n';
    out << value.fields.size() << '\n';
    for (auto const& [field, dots] : value.fields)
    {
        out << std::quoted(field) << ' ' << dots.size() << '\n';
        for (auto const& dot : dots) out << std::quoted(dot.actor) << ' ' << dot.sequence << ' ' << std::quoted(dot.value) << '\n';
    }
    return out.str();
}
inline FarmingControlState parse_farming_controls(FarmingState const& state, std::string const& body, uint64_t memory_limit)
{
    require(body.size() <= farming_control_size_bound(state) && farming_control_memory_bound(state) < memory_limit,
        "farming controls exceed their parsing memory budget");
    std::istringstream in(body); std::string magic; std::getline(in, magic); FarmingControlState result;
    in >> std::quoted(result.job_id) >> std::quoted(result.fingerprint) >> std::quoted(result.source_policy) >> std::quoted(result.native_binding) >> result.source_revision;
    require(in && magic == "ordinary-group-farming-controls-v1", "unsupported farming control snapshot");
    size_t count = 0; in >> count;
    require(in && count && count <= state.directories.size(), "farming control native member count exceeds its bound");
    for (size_t i = 0; i < count; ++i)
    {
        std::string identity; in >> std::quoted(identity);
        require(in && !identity.empty() && identity.size() <= 256, "invalid farming control native member"); result.native_members.push_back(std::move(identity));
    }
    in >> count;
    require(in && count <= state.directories.size(), "farming control actor count exceeds its bound");
    for (size_t i = 0; i < count; ++i)
    {
        std::string actor; uint64_t sequence = 0; in >> std::quoted(actor) >> sequence;
        require(in && actor.size() <= 256 && result.context.emplace(std::move(actor), sequence).second,
            "invalid or duplicate farming control actor");
    }
    in >> count; require(in && count <= state.directories.size() + 1, "farming control field count exceeds its bound");
    for (size_t i = 0; i < count; ++i)
    {
        std::string field; size_t size = 0; in >> std::quoted(field) >> size;
        require(in && field.size() <= 264 && size && size <= state.directories.size() && !result.fields.contains(field),
            "invalid or duplicate farming control field");
        auto& dots = result.fields[field]; dots.reserve(size);
        for (size_t j = 0; j < size; ++j)
        {
            FarmingControlDot dot; in >> std::quoted(dot.actor) >> dot.sequence >> std::quoted(dot.value);
            require(in && dot.actor.size() <= 256 && dot.value.size() <= 8, "invalid farming control field value"); dots.push_back(std::move(dot));
        }
    }
    in >> std::ws; require(in.eof(), "unexpected farming control snapshot data");
    validate_farming_controls(state, result); return result;
}
inline FarmingControlState rebind_farming_controls(FarmingState const& target, FarmingControlState const& value,
    std::span<FarmingControlNativeBinding const> bindings)
{
    auto const source = farming_control_source_shape(target, bindings); validate_farming_controls(source, value);
    auto translated = [&](std::string const& identity)
    {
        auto const at = std::find_if(bindings.begin(), bindings.end(), [&](auto const& binding) { return binding.source_identity == identity; });
        return at == bindings.end() ? identity : at->target_identity;
    };
    auto result = value; result.context.clear(); result.fields.clear(); result.native_members.clear();
    for (auto const& identity : value.native_members) result.native_members.push_back(translated(identity));
    result.native_binding = farming_control_native_digest(result.job_id, result.native_members);
    for (auto const& [actor, sequence] : value.context)
        require(result.context.emplace(translated(actor), sequence).second, "farming control native handoff collides with another actor");
    for (auto const& [field, dots] : value.fields)
    {
        auto const target_field = field == "control" ? field : "control:" + translated(field.substr(8));
        require(!result.fields.contains(target_field), "farming control native handoff collides with another scope");
        auto& target_dots = result.fields[target_field];
        for (auto const& dot : dots) target_dots.push_back({translated(dot.actor), dot.value, dot.sequence});
        std::sort(target_dots.begin(), target_dots.end(), [](auto const& a, auto const& b) { return a.actor < b.actor; });
    }
    validate_farming_controls(target, result); return result;
}
inline uint64_t farming_control_handoff_size_bound(FarmingState const& target, std::span<FarmingControlNativeBinding const> bindings)
{
    if (bindings.empty()) return farming_control_size_bound(target);
    return std::max(farming_control_size_bound(target), farming_control_size_bound(farming_control_source_shape(target, bindings)));
}
inline uint64_t farming_control_handoff_memory_bound(FarmingState const& target, std::span<FarmingControlNativeBinding const> bindings)
{
    if (bindings.empty()) return farming_control_memory_bound(target);
    auto const bytes = farming_control_handoff_size_bound(target, bindings);
    return grow_add(64 * 1024 + bytes * 16, target.directories.size() * (2 * sizeof(FarmingDirectory) + 1024));
}
inline FarmingControlState parse_farming_controls(FarmingState const& target, std::string const& body, uint64_t memory_limit,
    std::span<FarmingControlNativeBinding const> bindings)
{
    if (bindings.empty()) return parse_farming_controls(target, body, memory_limit);
    require(farming_control_handoff_memory_bound(target, bindings) < memory_limit, "farming control native handoff exceeds --max-ram");
    try { return parse_farming_controls(target, body, memory_limit); }
    catch (std::runtime_error const&)
    {
        if (bindings.empty()) throw;
        auto const source = farming_control_source_shape(target, bindings);
        return rebind_farming_controls(target, parse_farming_controls(source, body, memory_limit), bindings);
    }
}
inline FarmingControlState merge_farming_controls(FarmingState const& state, FarmingControlState const& a, FarmingControlState const& b)
{
    validate_farming_controls(state, a); validate_farming_controls(state, b);
    require(a.source_revision != b.source_revision || a.source_policy == b.source_policy,
        "farming controls disagree about the same source policy revision");
    auto result = a.source_revision >= b.source_revision ? a : b; result.context = a.context; result.fields.clear();
    result.native_members = farming_control_native_members(state); result.native_binding = farming_control_native_digest(state);
    for (auto const& [actor, sequence] : b.context) result.context[actor] = std::max(result.context[actor], sequence);
    std::set<std::string> fields;
    for (auto const& [field, dots] : a.fields) fields.insert(field);
    for (auto const& [field, dots] : b.fields) fields.insert(field);
    for (auto const& field : fields)
    {
        auto append = [&](FarmingControlState const& from, FarmingControlState const& other)
        {
            auto at = from.fields.find(field); if (at == from.fields.end()) return;
            auto const opposite = other.fields.find(field);
            for (auto const& dot : at->second)
            {
                bool shared = false;
                if (opposite != other.fields.end()) for (auto const& candidate : opposite->second)
                    if (candidate.actor == dot.actor && candidate.sequence == dot.sequence)
                    { require(candidate.value == dot.value, "conflicting copies of a farming control causal value"); shared = true; }
                auto const context = other.context.find(dot.actor);
                if (!shared && context != other.context.end() && context->second >= dot.sequence) continue;
                auto& kept = result.fields[field];
                if (std::find(kept.begin(), kept.end(), dot) == kept.end()) kept.push_back(dot);
            }
        };
        append(a, b); append(b, a);
    }
    for (auto& [field, dots] : result.fields)
        std::sort(dots.begin(), dots.end(), [](auto const& x, auto const& y) { return x.actor < y.actor; });
    validate_farming_controls(state, result); return result;
}
inline FarmingControlState advance_farming_controls(FarmingState const& state, FarmingControlState previous,
    std::string const& actor, std::string_view action, std::vector<std::string> const& destinations, std::string source_policy)
{
    validate_farming_controls(state, previous);
    require(action == "pause" || action == "drain" || action == "resume" || action == "retire", "invalid local farming control action");
    auto const member = std::find_if(state.directories.begin(), state.directories.end(),
        [&](auto const& directory) { return directory.directory_identity == actor; });
    require(member != state.directories.end() && farming_member(*member), "local farming control actor is not an active native member");
    previous.native_members = farming_control_native_members(state); previous.native_binding = farming_control_native_digest(state);
    auto& sequence = previous.context[actor]; require(sequence < UINT64_MAX, "farming control actor sequence exhausted"); ++sequence;
    auto const value = action == "resume" ? "active" : action == "pause" ? "paused" : action == "retire" ? "retired" : "draining";
    if (destinations.empty()) previous.fields["control"] = {{actor, value, sequence}};
    else for (auto const& destination : destinations) previous.fields["control:" + destination] = {{actor, value, sequence}};
    previous.source_revision = state.policy_revision; previous.source_policy = std::move(source_policy);
    validate_farming_controls(state, previous); return previous;
}
} // namespace group_grow
