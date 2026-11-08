#pragma once
#include "host/BatchPlotter.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <optional>

namespace pos2gpu {
// Returns false on malformed input (caller prints usage + exits 1).
inline bool parse_devices_arg(std::string const& s, pos2gpu::BatchOptions& opts)
{
    // Accept comma-separated mix of:
    //   "all"             → all GPUs + CPU worker
    //   "gpu"             → all GPUs
    //   "cpu"             → CPU worker (CPU-only; suppresses the implicit
    //                       default GPU that a bare run would otherwise add)
    //   "<int>"           → explicit GPU id
    //   "gpu:<tier>"      → all GPUs default to <tier>
    //   "all:<tier>"      → all GPUs default to <tier>, plus CPU worker
    //   "<int>:<tier>"    → explicit GPU id + per-GPU tier override
    //
    // <tier> is one of plain/compact/minimal/tiny/pinned/auto. The "auto"
    // sentinel explicitly opts back into auto-pick — useful for
    // overriding `gpu:<tier>` or global `--tier` on one specific GPU.
    //
    // Compose freely: `gpu,2:tiny` = all GPUs auto-pick except GPU 2
    // which uses tiny. `gpu:tiny,2:plain` = all GPUs tiny except GPU 2
    // which uses plain. `cpu:<tier>` is rejected — CPU worker has no
    // streaming tier.
    auto is_valid_tier = [](std::string const& t) {
        return t == "plain" || t == "compact" || t == "minimal" ||
               t == "tiny"  || t == "pinned"  || t == "auto";
    };
    auto bad = [&](char const* why) {
        std::cerr << "Error: --devices: " << why
                  << " (token list: \"" << s << "\")\n";
        return false;
    };

    // A --devices list REPLACES any earlier one, so every selection it can
    // express resets here — CPU included. Leaving the CPU selection standing
    // would make `--devices all --devices gpu` keep a CPU nobody asked for.
    // cpu_workers is deliberately NOT reset: it is a count, not a selection,
    // and an explicit --cpu-workers must survive a later --devices.
    opts.device_ids.clear();
    opts.use_all_devices = false;
    opts.per_device_tier.clear();
    opts.all_gpus_tier.clear();
    opts.cpu_node_ids.clear();
    opts.use_all_cpu_nodes = false;
    bool any_token = false;
    bool any_gpu_token = false;
    size_t start = 0;
    while (start <= s.size()) {
        size_t const end = s.find(',', start);
        std::string const tok = s.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (tok.empty()) return false;
        any_token = true;

        std::string selector = tok;
        std::string tier_suffix;
        if (size_t const colon = tok.find(':'); colon != std::string::npos) {
            selector    = tok.substr(0, colon);
            tier_suffix = tok.substr(colon + 1);
            if (tier_suffix.empty())  return bad("empty tier after `:`");
            if (!is_valid_tier(tier_suffix)) {
                return bad("invalid tier (expect plain|compact|minimal|tiny|pinned|auto)");
            }
        }

        // "gpu7" / "cpu1" — a kind plus an index. Bare "7" is the historical
        // spelling of "gpu7" and still parses, below.
        auto indexed = [](std::string const& tok, char const* kind) -> std::optional<int> {
            std::string const prefix(kind);
            if (tok.size() <= prefix.size() || tok.compare(0, prefix.size(), prefix) != 0) {
                return std::nullopt;
            }
            std::string const digits = tok.substr(prefix.size());
            char* endp = nullptr;
            errno = 0;
            long const v = std::strtol(digits.c_str(), &endp, 10);
            if (endp == digits.c_str() || endp != digits.c_str() + digits.size() || errno == ERANGE || v < 0 || v > 1023) {
                return std::nullopt;
            }
            return static_cast<int>(v);
        };

        auto add_gpu = [&](int id) -> bool {
            opts.device_ids.push_back(id);
            any_gpu_token = true;
            if (!tier_suffix.empty()) {
                auto const it = opts.per_device_tier.find(id);
                if (it != opts.per_device_tier.end() && it->second != tier_suffix) {
                    return bad("same GPU id given two different :tier overrides");
                }
                opts.per_device_tier[id] = tier_suffix;
            }
            return true;
        };

        if (selector == "all") {
            opts.use_all_devices  = true;
            opts.use_all_cpu_nodes = true;  // "all" means every GPU *and* every CPU node
            any_gpu_token = true;
            if (!tier_suffix.empty()) opts.all_gpus_tier = tier_suffix;
        } else if (selector == "gpu") {
            opts.use_all_devices = true;
            any_gpu_token = true;
            if (!tier_suffix.empty()) opts.all_gpus_tier = tier_suffix;
        } else if (selector == "cpu") {
            if (!tier_suffix.empty()) return bad("cpu token cannot carry a tier");
            // Every CPU node, exactly as `gpu` means every GPU. It is a
            // SELECTION, not a count: --cpu-workers says how many plots run per
            // node. So repeats dedup — `cpu,cpu` names the same hardware twice
            // and cannot mean more of it, any more than `0,0` means two cards.
            opts.use_all_cpu_nodes = true;
        } else if (auto const node = indexed(selector, "cpu")) {
            if (!tier_suffix.empty()) return bad("cpu token cannot carry a tier");
            opts.cpu_node_ids.push_back(*node);
        } else if (auto const id = indexed(selector, "gpu")) {
            if (!add_gpu(*id)) return false;
        } else {
            char* endp = nullptr;
            errno = 0;
            long const v = std::strtol(selector.c_str(), &endp, 10);
            if (endp == selector.c_str() || endp != selector.c_str() + selector.size() || errno == ERANGE || v < 0 || v > 1023) {
                return bad("unrecognised device token "
                           "(expect all|gpu|cpu|gpu<n>|cpu<n>|<id>)");
            }
            if (!add_gpu(static_cast<int>(v))) return false;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (!any_token) return false;
    bool const any_cpu_token = opts.use_all_cpu_nodes || !opts.cpu_node_ids.empty();
    if (!any_gpu_token && !any_cpu_token) return false;

    // `cpu` covers every node, so listing nodes as well is redundant.
    if (opts.use_all_cpu_nodes) opts.cpu_node_ids.clear();

    auto dedup = [](std::vector<int>& v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    dedup(opts.device_ids);
    dedup(opts.cpu_node_ids);
    // Any successful --devices — even a CPU-only one that left device_ids empty —
    // is an explicit selection. resolve_batch_devices keys off this to avoid
    // materialising a default GPU on top of an explicit CPU-only request.
    opts.devices_specified = true;
    return true;
}

} // namespace pos2gpu
