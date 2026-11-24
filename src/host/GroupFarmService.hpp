#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace group_grow
{
// A completed signage-point lookup from one selected harvester log. Logs do
// not enumerate arrivals, outstanding challenges or independently verify a proof.
struct FarmingServiceRecord
{
    std::chrono::steady_clock::time_point at{};
    uint64_t response_ns = 0, eligible = 0, loaded = 0;
    bool counts_known = false, gap = false, began_before_sampling = false, log_seen = false;
};
struct FarmingServiceWindow
{
    uint64_t completions = 0, eligible_completions = 0, zero_eligible_completions = 0, loaded_plots = 0;
    uint64_t p50_ns = 0, p95_ns = 0, max_ns = 0, missed_deadline = 0, age_ms = 0, started_before_window = 0;
    bool counts_known = false, deadline_known = false, incomplete = false, log_observed = false;
    std::optional<bool> lookup_pressure;
    std::chrono::steady_clock::time_point feedback_expires{};
};
class FarmingServiceHistory
{
    using Clock = std::chrono::steady_clock;
    static constexpr std::size_t limit = 128;
    struct Entry { FarmingServiceRecord record; uint64_t deadline_ns; };
    std::array<Entry, limit> recent_{};
    std::size_t next_ = 0, count_ = 0;
    Clock::time_point gap_{}, log_seen_{};
public:
    void add(FarmingServiceRecord const& value, uint64_t deadline_ns)
    {
        if (value.log_seen) log_seen_ = std::max(value.at, log_seen_);
        if (value.gap) { gap_ = std::max(value.at, gap_); return; }
        if (value.log_seen) return;
        if (count_ == limit) gap_ = std::max(recent_[next_].record.at, gap_);
        recent_[next_] = {value, deadline_ns}; next_ = (next_ + 1) % limit;
        count_ = std::min(count_ + 1, limit);
    }
    FarmingServiceWindow window(Clock::time_point now = Clock::now()) const
    {
        auto const first = now - std::chrono::seconds(30);
        std::array<uint64_t, limit> elapsed{};
        FarmingServiceWindow result; result.incomplete = gap_ != Clock::time_point{} && gap_ >= first;
        result.log_observed = log_seen_ != Clock::time_point{} && log_seen_ >= first;
        Clock::time_point last{};
        bool counts = true, deadlines = true;
        uint64_t feedback = 0, slow = 0;
        for (std::size_t i = 0; i < count_; ++i)
        {
            auto const& entry = recent_[i]; auto const& value = entry.record;
            if (value.at < first || value.at > now) continue;
            elapsed[result.completions++] = value.response_ns;
            if (value.began_before_sampling || value.at - first < std::chrono::nanoseconds(value.response_ns)) ++result.started_before_window;
            if (value.counts_known)
            {
                if (value.eligible) ++result.eligible_completions;
                else ++result.zero_eligible_completions;
            }
            else counts = false;
            if (!entry.deadline_ns) deadlines = false;
            else if (value.response_ns > entry.deadline_ns) ++result.missed_deadline;
            // Fast empty challenges do not establish spare lookup capacity.
            // Slow completions still provide evidence with absent or zero counts.
            if (entry.deadline_ns)
            {
                bool const late = value.response_ns >= entry.deadline_ns / 2;
                if (late || (value.counts_known && value.eligible))
                {
                    auto const expires = value.at + std::chrono::seconds(30);
                    if (!feedback || expires < result.feedback_expires) result.feedback_expires = expires;
                    ++feedback; slow += late;
                }
            }
            if (value.at >= last) { last = value.at; result.loaded_plots = value.loaded; }
        }
        if (!result.completions) return result;
        result.counts_known = counts; result.deadline_known = deadlines;
        result.age_ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count());
        std::sort(elapsed.begin(), elapsed.begin() + result.completions);
        result.p50_ns = elapsed[(50 * result.completions - 1) / 100];
        result.p95_ns = elapsed[(95 * result.completions - 1) / 100];
        result.max_ns = elapsed[result.completions - 1];
        if (feedback) result.lookup_pressure = slow > feedback / 20; // nearest-rank p95 at half the deadline
        return result;
    }
};
} // namespace group_grow
