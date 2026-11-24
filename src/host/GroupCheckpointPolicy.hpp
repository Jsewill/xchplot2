// Runtime checkpoint selection for sequential group output.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace group_grow
{
inline long double construction_margin(size_t observations) { return observations >= 4 ? 0.10L : 0.15L; }

struct PackingCheckpointHistory
{
    static constexpr uint64_t forecast_bytes = 1024ull * 1024 * 1024;
    static constexpr uint64_t maximum_bytes = 4 * forecast_bytes;
    struct Sample { uint64_t bytes, elapsed_ns, checkpoint_ns, codec_ns = 0, interval = 0; bool terminal_trial = false; };
    std::array<Sample, 8> samples{};
    size_t count = 0, next = 0;
    uint64_t trial_from = 0, trial_bytes = 0, learned_ceiling = maximum_bytes;
    long double trial_baseline = 0;
    std::array<long double, 3> trial_rates{};
    size_t trial_count = 0, trial_baseline_count = 0, trial_goal = 3;
    static uint64_t memory_bytes()
    {
        return sizeof(PackingCheckpointHistory) + 2 * sizeof(std::array<long double, 8>) +
            sizeof(std::array<uint64_t, 8>);
    }
    void add(Sample sample)
    {
        if (!sample.bytes || !sample.elapsed_ns || sample.checkpoint_ns > sample.elapsed_ns) return;
        samples[next] = sample; next = (next + 1) % samples.size();
        count = std::min(count + 1, samples.size());
    }
    uint64_t fastest_completed_interval() const
    {
        long double best_rate = 0; uint64_t best = 0;
        for (size_t i = 0; i < count; ++i)
        {
            uint64_t interval = samples[i].interval;
            if (!interval) continue;
            std::array<long double, 8> rates{}; size_t n = 0;
            for (size_t j = 0; j < count; ++j)
                if (samples[j].interval == interval && (samples[j].bytes >= interval || samples[j].terminal_trial))
                    rates[n++] = static_cast<long double>(samples[j].elapsed_ns) / samples[j].bytes;
            if (!n) continue;
            std::sort(rates.begin(), rates.begin() + n);
            auto const rate = (rates[(n - 1) / 2] + rates[n / 2]) / 2;
            if (!best || rate < best_rate || (rate == best_rate && interval < best))
            { best = interval; best_rate = rate; }
        }
        return best;
    }
    // Called once after each accepted add, so each completed trial interval
    // contributes exactly one observation.
    uint64_t select(uint64_t current, uint64_t ceiling, uint64_t remaining, bool ordinary = false)
    {
        if (ordinary && current > ceiling)
        { trial_bytes = 0; trial_count = 0; return ceiling; }
        if (!count || !remaining) return current;
        auto active_rate = [](Sample const& sample)
        { return static_cast<long double>(std::max(sample.elapsed_ns - sample.checkpoint_ns, sample.codec_ns)) / sample.bytes; };
        auto completed_rate = [&](Sample const& sample)
        { return active_rate(sample) + static_cast<long double>(sample.checkpoint_ns) / sample.bytes; };
        if (trial_bytes)
        {
            // Larger dirty ranges need not have constant checkpoint latency.
            // Price completed end-to-end intervals, including reclamation,
            // before keeping a larger replay and retained-scratch frontier.
            auto const& sample = samples[(next + samples.size() - 1) % samples.size()];
            if (sample.bytes < trial_bytes) return current;
            trial_rates[trial_count++] = completed_rate(sample);
            if (trial_count < trial_goal) return current;
            std::sort(trial_rates.begin(), trial_rates.begin() + trial_count);
            long double const rate = (trial_rates[(trial_count - 1) / 2] + trial_rates[trial_count / 2]) / 2;
            bool const accepted = rate < trial_baseline * (ordinary ? 1 : 1 - construction_margin(trial_baseline_count));
            trial_bytes = 0; trial_count = 0;
            if (!accepted) { learned_ceiling = trial_from; return trial_from; }
        }
        ceiling = std::min({ceiling, maximum_bytes, learned_ceiling});
        if (current >= ceiling) return current;
        bool const trial = current >= forecast_bytes;
        if (trial && !ordinary)
        {
            // Extrapolate only one geometric step, and keep enough work for
            // completed trial intervals. Short tails retain their frontier.
            ceiling = std::min(ceiling, current * 2);
            if (remaining / ceiling <= trial_rates.size()) return current;
        }
        else if (!ordinary) ceiling = std::min(ceiling, forecast_bytes);
        std::array<long double, 8> streaming{}, checkpoint{};
        size_t observations = 0;
        for (size_t i = 0; i < count; ++i)
        {
            auto const& sample = samples[i];
            if (trial && sample.bytes < current) continue;
            // A pending encoder can use checkpoint waits. Keep completed
            // codec work as a floor rather than treating all of it as free.
            // Codec counters arrive per joined window; short intervals have
            // coarse attribution, so this remains a forecast with a margin.
            streaming[observations] = active_rate(sample);
            checkpoint[observations++] = sample.checkpoint_ns;
        }
        // A median-sized baseline prevents one unusually slow checkpoint
        // from justifying a larger retained-scratch and replay frontier.
        if (!observations || (!ordinary && trial && observations < trial_rates.size())) return current;
        auto median = [&](auto& values)
        {
            std::sort(values.begin(), values.begin() + observations);
            return (values[(observations - 1) / 2] + values[observations / 2]) / 2;
        };
        long double const active = remaining * median(streaming), latency = median(checkpoint);
        auto cost = [&](uint64_t bytes)
        { return active + std::ceil(static_cast<long double>(remaining) / bytes) * latency; };
        long double const original = cost(current); auto best_cost = original; uint64_t best = current;
        std::array<uint64_t, 10> const candidates{4ull * 1024 * 1024, 16ull * 1024 * 1024,
            64ull * 1024 * 1024, 256ull * 1024 * 1024, 512ull * 1024 * 1024,
            forecast_bytes, 2 * forecast_bytes, maximum_bytes, current, ceiling};
        for (auto bytes : candidates) if (bytes > current && bytes <= ceiling && cost(bytes) < best_cost)
        { best_cost = cost(bytes); best = bytes; }
        // A fresh ordinary file has no retained scratch to reclaim. Compare
        // even small predicted savings against actual paced write throughput.
        if (best_cost >= original * (ordinary ? 1 : 1 - construction_margin(observations))) return current;
        if (trial || ordinary)
        {
            // Keep the baseline's actual total rather than a model that
            // treats every future checkpoint as the same fixed latency.
            observations = 0;
            for (size_t i = 0; i < count; ++i) if (samples[i].bytes >= current)
                streaming[observations++] = completed_rate(samples[i]);
            if (!observations || (!ordinary && remaining < best)) return current;
            trial_goal = ordinary ? size_t(std::clamp<uint64_t>(remaining / best, 1, trial_rates.size())) : trial_rates.size();
            trial_baseline = median(streaming); trial_baseline_count = observations;
            trial_from = current; trial_bytes = best;
        }
        return best;
    }
};
} // namespace group_grow
