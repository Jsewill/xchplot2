#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace group_grow
{
// Counts refer to new durable members in the remaining job. For an open
// fill, the layout count is conditional on compressed sizes and admission.
struct FillMemberTarget
{
    uint64_t expected = 0, minimum = 0;
    std::optional<uint64_t> maximum = 0;
    std::optional<std::pair<uint64_t, uint64_t>> at_observed_sizes;
    bool open = false, waiting = false;

    FillMemberTarget after(uint64_t published) const
    {
        auto next = *this;
        auto subtract = [published](uint64_t count) { return count - std::min(published, count); };
        next.expected = subtract(expected);
        next.minimum = subtract(minimum);
        if (maximum) next.maximum = subtract(*maximum);
        if (at_observed_sizes) next.at_observed_sizes = std::pair{subtract(at_observed_sizes->first), subtract(at_observed_sizes->second)};
        return next;
    }
};

struct FillRate
{
    double fast = 0, slow = 0; // Observed seconds per newly durable member.
    unsigned samples = 0;
    uint64_t oldest_utc = 0;
    bool scaled = false;
    uint64_t active_destinations = 0, delivery_routes = 0;
    char const* limiting_resource = "unmeasured";
};

struct FillCompletionForecast
{
    static constexpr std::array<std::string_view, 5> policies{"auto", "incremental", "external", "merged", "staged"};
    static constexpr std::array<std::string_view, 2> output_names{"packed members", "encoded groups"};
    FillMemberTarget members;
    std::array<FillRate, policies.size()> history{};
    std::array<FillRate, output_names.size()> worker_outputs{};
    size_t policy = 0;
    std::optional<size_t> faster_policy;
    std::optional<size_t> requested_worker_output;
    char const* source = "unmeasured";
    char const* limiting_resource = "unmeasured";
    std::string unmeasured_costs = "generation and delivery, owner encoding, destination writes and sync";
    std::optional<double> seconds, low, high, lower_bound;
    bool conditional_pool = false;
    bool schedule_observed = false, shared_storage_unmeasured = false;
    uint64_t scheduled_members = 0;
    size_t unmeasured_lanes = 0;

    static size_t policy_index(std::string_view name)
    {
        return size_t(std::find(policies.begin(), policies.end(), name) - policies.begin());
    }
    // With one sample we have a point observation but no observed variation.
    // Even with several samples this range is conditional on the layout count.
    void project(FillRate const& rate, char const* evidence)
    {
        if (!rate.samples || members.waiting || !members.expected) return;
        source = evidence;
        seconds = members.expected * (rate.fast + rate.slow) / 2;
        if (rate.samples > 1 || members.at_observed_sizes)
        {
            auto counts = members.at_observed_sizes.value_or(std::pair{members.expected, members.expected});
            low = std::min(counts.first, members.expected) * rate.fast;
            high = std::max(counts.second, members.expected) * rate.slow;
        }
    }
    void use_history()
    {
        if (members.waiting) return;
        if (!members.expected && !members.maximum.value_or(1))
        {
            source = "complete"; unmeasured_costs.clear(); conditional_pool = false;
            seconds = low = high = 0; return;
        }
        if (policy >= history.size()) return;
        if (history[policy].samples)
        {
            limiting_resource = history[policy].limiting_resource;
            project(history[policy], history[policy].scaled ? "previous-fills-scaled-by-member-count" : "previous-fills");
        }
        else if (policy == 0 && history[1].samples && history[2].samples)
        {
            auto const& ram = history[1]; auto const& scratch = history[2];
            project({std::min(ram.fast, scratch.fast), std::max(ram.slow, scratch.slow), 2},
                "observed-ram-and-scratch-policies");
        }
        else if (policy == policy_index("incremental") && conditional_pool)
        {
            if (requested_worker_output)
                project(worker_outputs[*requested_worker_output], "observed-worker-output");
            else if (worker_outputs[0].samples && worker_outputs[1].samples)
                project({std::min(worker_outputs[0].fast, worker_outputs[1].fast),
                         std::max(worker_outputs[0].slow, worker_outputs[1].slow), 2}, "observed-worker-output-range");
        }
    }
    static void number(std::ostream& out, std::optional<double> value)
    {
        if (value && std::isfinite(*value)) out << *value;
        else out << "null";
    }
    std::string evidence_description() const
    {
        std::string_view const evidence(source);
        if (evidence == "complete") return "All work in this fill has finished.";
        if (evidence == "live-schedule") return "Based on the current generation and storage schedule.";
        if (evidence == "live-publications") return "Based on members published during this fill.";
        if (evidence == "awaiting-publication") return "The point estimate has elapsed and the fill is still running.";
        if (evidence == "previous-fills") return "Based on comparable completed fills.";
        if (evidence == "previous-fills-scaled-by-member-count")
            return "Based on comparable fills, adjusted for the remaining member count.";
        if (evidence == "observed-ram-and-scratch-policies") return "Based on previous RAM and scratch fills.";
        if (evidence == "observed-worker-output") return "Based on fills that completed with this worker output.";
        if (evidence == "observed-worker-output-range") return "Spans comparable fills using packed members and encoded groups.";
        return members.waiting ? "Timing depends on the destination or policy becoming available." :
            "Timing requires a comparable fill or observations from plotting.";
    }
    static std::string observation(std::string_view name, FillRate const& rate)
    {
        std::ostringstream out;
        out << name << ", " << std::fixed << std::setprecision(2) << rate.fast;
        if (rate.samples > 1 && rate.slow > rate.fast) out << " to " << rate.slow;
        out << " s/durable member across " << rate.samples << " comparable fill" << (rate.samples == 1 ? "" : "s");
        if (rate.scaled) out << ", adjusted for member count";
        if (rate.active_destinations) out << ", peak active destinations " << rate.active_destinations;
        if (rate.delivery_routes) out << ", validated delivery routes " << rate.delivery_routes;
        return out.str();
    }
    std::string policy_observation(size_t index) const
    { return observation(std::string(policies[index]) + " construction", history[index]); }
    std::string output_observation(size_t index) const
    { return observation(std::string(output_names[index]) + " with incremental construction", worker_outputs[index]); }
    std::string description() const
    {
        std::ostringstream out;
        out << members.expected << " remaining member" << (members.expected == 1 ? "" : "s")
            << (members.open ? " at forecast sizes" : "");
        if (members.open)
        {
            out << " (" << members.minimum << " committed";
            if (members.maximum) out << ", up to " << *members.maximum << " within the limits";
            else out << ", final count depends on available space";
            if (members.at_observed_sizes) out << ", " << members.at_observed_sizes->first << " to " << members.at_observed_sizes->second << " at observed file sizes";
            out << ')';
        }
        if (members.waiting) return out.str() + ", waiting for a destination or policy change";
        if (!seconds)
        {
            if (std::string_view(source) == "awaiting-publication")
            {
                out << ", waiting for the next durable publication";
                if (high) out << " (observed range extends another " << std::fixed << std::setprecision(1) << *high << " s)";
                return out.str();
            }
            out << ", time through generation, transfer and durable writing is unmeasured";
            if (lower_bound) out << " (modeled lower bound " << std::fixed << std::setprecision(1) << *lower_bound << " s)";
            return out.str();
        }
        out << std::fixed << std::setprecision(1) << ", ";
        if (low && high && *high > *low) out << *low << " to " << *high;
        else out << *seconds;
        out << " s to final publication";
        if (members.expected) out << " (" << std::setprecision(2) << *seconds / members.expected << " s/member)";
        return out.str();
    }
    void json(std::ostream& out) const
    {
        out << "{\"remaining_members\":" << members.expected << ",\"committed_remaining_members\":" << members.minimum
            << ",\"remaining_member_limit\":";
        if (members.maximum) out << *members.maximum; else out << "null";
        out << ",\"remaining_members_at_observed_sizes\":";
        if (members.at_observed_sizes) out << '[' << members.at_observed_sizes->first << ',' << members.at_observed_sizes->second << ']';
        else out << "null";
        out << ",\"open_capacity\":" << (members.open ? "true" : "false")
            << ",\"waiting\":" << (members.waiting ? "true" : "false")
            << ",\"source\":\"" << source << "\",\"limiting_resource\":\"" << limiting_resource
            << "\",\"unmeasured_costs\":\"" << unmeasured_costs
            << "\",\"pool_capacity_unconfirmed\":" << (conditional_pool ? "true" : "false") << ",\"seconds\":";
        number(out, seconds); out << ",\"seconds_low\":"; number(out, low);
        out << ",\"seconds_high\":"; number(out, high);
        out << ",\"modeled_lower_bound_seconds\":"; number(out, lower_bound);
        out << ",\"schedule_observed\":" << (schedule_observed ? "true" : "false")
            << ",\"scheduled_remaining_members\":";
        if (schedule_observed) out << scheduled_members; else out << "null";
        out << ",\"unmeasured_lanes\":";
        if (schedule_observed) out << unmeasured_lanes; else out << "null";
        out << ",\"shared_storage_bandwidth_unmeasured\":";
        if (schedule_observed) out << (shared_storage_unmeasured ? "true" : "false"); else out << "null";
        out << ",\"faster_observed_policy\":";
        if (faster_policy) out << '"' << policies[*faster_policy] << '"'; else out << "null";
        out << ",\"seconds_per_durable_member\":";
        number(out, members.expected && seconds ? std::optional(*seconds / members.expected) : std::nullopt);
        auto rate_json = [&](FillRate const& rate)
        {
            out << "{\"samples\":" << rate.samples;
            if (rate.samples)
            {
                out << ",\"fast_seconds_per_member\":" << rate.fast << ",\"slow_seconds_per_member\":"
                    << rate.slow << ",\"oldest_utc_seconds\":" << rate.oldest_utc << ",\"scaled_by_member_count\":" << (rate.scaled ? "true" : "false");
                if (rate.active_destinations) out << ",\"peak_active_destinations\":" << rate.active_destinations;
                if (rate.delivery_routes) out << ",\"validated_delivery_routes\":" << rate.delivery_routes;
                out << ",\"limiting_resource\":\"" << rate.limiting_resource << '"';
            }
            out << '}';
        };
        out << ",\"construction_history\":{";
        for (size_t i = 0; i < history.size(); ++i)
        {
            if (i) out << ',';
            out << '"' << policies[i] << "\":"; rate_json(history[i]);
        }
        out << "},\"incremental_worker_output_history\":{\"packed\":"; rate_json(worker_outputs[0]);
        out << ",\"encoded\":"; rate_json(worker_outputs[1]);
        out << "}}";
    }
};
}
