// Local control intent uses compact causal snapshots; queued immutable
// controls remain readable for older jobs. Policy publication keeps its member
// barrier, while each writer may stop locally reserved work independently.
#pragma once
#include "host/GroupFarmUpdate.hpp"

namespace group_grow
{
inline std::string restrictive_farming_control(std::string const& a, std::string const& b)
{
    auto rank = [](std::string const& value)
    { return value == "paused" ? 3 : value == "retired" ? 2 : value == "draining" ? 1 : 0; };
    return rank(a) >= rank(b) ? a : b;
}
struct FarmingControlView
{
    std::string control = "active", digest;
    std::vector<std::string> destinations, conflicts, requests;
    std::string effective(size_t lane) const
    {
        require(lane < destinations.size(), "farming control destination is outside the saved job");
        return restrictive_farming_control(control, destinations[lane]);
    }
};
inline void enforce_farming_retired_controls(FarmingState const& state, FarmingControlView& controls)
{
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
        if (!farming_member(state.directories[lane])) controls.destinations[lane] = "retired";
    if (std::none_of(state.directories.begin(), state.directories.end(), farming_member)) controls.control = "retired";
}
inline FarmingControlView farming_control_view(FarmingState const& state, std::vector<FarmingUpdateRecord> const& records)
{
    require(records.size() <= farming_update_history, "farming control history exceeds its bound");
    FarmingControlView result; result.control = state.control;
    for (auto const& directory : state.directories) result.destinations.push_back(directory.control);
    if (std::none_of(state.directories.begin(), state.directories.end(), farming_member)) result.control = "retired";
    std::map<std::string, std::vector<FarmingUpdateRecord const*>> fields;
    std::map<std::string, FarmingUpdateRecord const*> candidates;
    for (auto const& record : records)
    {
        auto const& update = record.file.update;
        require(update.job_id == state.job_id, "farming control belongs to another job");
        if (!farming_update_control_only(update) || update.id == state.applied_update ||
            (record.receipt && farming_update_terminal(record.receipt->status))) continue;
        // Reuse the policy field preconditions, including the same-revision
        // digest guard. A changed or foreign grant never becomes authority.
        try { (void)farming_update_rebase(update, state); }
        catch (std::runtime_error const&) { continue; }
        candidates.emplace(update.id, &record);
    }
    std::set<std::string> dominated;
    for (auto const& [id, record] : candidates)
    {
        std::vector<std::string> pending;
        std::set<std::string> visited;
        auto add = [&](std::string const& previous)
        {
            if (!visited.insert(previous).second) return;
            require(visited.size() <= farming_update_history, "farming control causality exceeds its bound");
            pending.push_back(previous);
        };
        for (auto const& previous : record->file.update.supersedes) add(previous);
        while (!pending.empty())
        {
            auto previous = std::move(pending.back()); pending.pop_back();
            dominated.insert(previous);
            auto at = std::find_if(records.begin(), records.end(), [&](auto const& row) { return row.file.update.id == previous; });
            if (at == records.end()) continue;
            require(at->file.update.generation < record->file.update.generation, "cyclic farming control supersession");
            for (auto const& ancestor : at->file.update.supersedes) add(ancestor);
        }
    }
    for (auto const& [id, record] : candidates) if (!dominated.contains(id))
    {
        result.requests.push_back(id);
        for (auto const& [field, expected] : record->file.update.expected) fields[field].push_back(record);
    }
    auto rank = [](std::string const& value)
    { return value == "paused" ? 3 : value == "retired" ? 2 : value == "draining" ? 1 : 0; };
    for (auto const& [field, contenders] : fields)
    {
        std::set<std::string> values;
        for (auto const* contender : contenders) values.insert(farming_update_control(contender->file.update.action));
        auto const value = *std::max_element(values.begin(), values.end(), [&](auto const& a, auto const& b) { return rank(a) < rank(b); });
        if (values.size() > 1) result.conflicts.push_back(field);
        if (field == "control") result.control = value;
        else
        {
            auto const identity = field.substr(std::string_view("control:").size());
            auto const at = std::find_if(state.directories.begin(), state.directories.end(),
                [&](auto const& directory) { return directory.directory_identity == identity; });
            require(at != state.directories.end(), "farming control target is not a saved destination");
            result.destinations[size_t(at - state.directories.begin())] = value;
        }
    }
    enforce_farming_retired_controls(state, result);
    std::ostringstream body; body << std::quoted(state.job_id) << ' ' << std::quoted(result.control) << '\n';
    for (size_t lane = 0; lane < result.destinations.size(); ++lane)
        body << std::quoted(state.directories[lane].directory_identity) << ' ' << std::quoted(result.destinations[lane]) << '\n';
    for (auto const& id : result.requests) body << std::quoted(id) << '\n';
    for (auto const& field : result.conflicts) body << std::quoted(field) << '\n';
    result.digest = stream_state::digest(body.str()); return result;
}
inline FarmingControlView farming_control_view(FarmingState const& state, std::vector<FarmingUpdateRecord> const& records,
    FarmingControlState const& controls)
{
    validate_farming_controls(state, controls);
    auto result = farming_control_view(state, records);
    for (auto const& [field, dots] : controls.fields)
    {
        std::string value = "active"; std::set<std::string> values;
        for (auto const& dot : dots) { values.insert(dot.value); value = restrictive_farming_control(value, dot.value); }
        if (values.size() > 1 && std::find(result.conflicts.begin(), result.conflicts.end(), field) == result.conflicts.end())
            result.conflicts.push_back(field);
        if (field == "control") result.control = value;
        else
        {
            auto const at = std::find_if(state.directories.begin(), state.directories.end(),
                [&](auto const& directory) { return directory.directory_identity == field.substr(8); });
            require(at != state.directories.end(), "farming control snapshot targets a foreign destination");
            result.destinations[size_t(at - state.directories.begin())] = value;
        }
    }
    auto const compact = farming_control_state_text(controls);
    if (!controls.context.empty()) result.requests.push_back(stream_state::digest(compact));
    enforce_farming_retired_controls(state, result);
    result.digest = stream_state::digest(result.digest + compact); return result;
}
} // namespace group_grow
