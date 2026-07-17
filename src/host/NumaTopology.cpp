#include "host/NumaTopology.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

// sched_setaffinity / cpu_set_t are Linux-specific, and so is the sysfs the
// enumeration reads. The rest of the file is portable by construction: on a
// host without /sys/devices/system/node the reads come back empty and
// host_numa_nodes() reports one node meaning "the machine", which is exactly
// the right answer for a platform we cannot pin on. Windows is a supported
// build target (see the _WIN32 branches in GpuBufferPool.cpp), so only the
// affinity call is walled off — not the header, and not the callers.
#ifndef _WIN32
#include <sched.h>
#endif

namespace pos2gpu {

namespace {

// Read a whole sysfs file. They are small, and absent files are the normal way
// sysfs says "this host has no such thing" — so a miss is empty, not an error.
std::string read_sysfs(std::string const& path)
{
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

}  // namespace

std::vector<int> parse_cpu_list(std::string const& text)
{
    std::vector<int> out;
    std::size_t      start = 0;
    while (start < text.size()) {
        std::size_t const comma = text.find(',', start);
        std::string const tok =
            text.substr(start, comma == std::string::npos ? std::string::npos
                                                          : comma - start);
        if (!tok.empty()) {
            // "3" or "0-15". Anything else is sysfs speaking a dialect we do not
            // know, and guessing at it would produce a mask rather than an error
            // — so an unparsable token contributes nothing.
            std::size_t const dash = tok.find('-');
            char*             endp = nullptr;
            if (dash == std::string::npos) {
                long const v = std::strtol(tok.c_str(), &endp, 10);
                if (endp != tok.c_str() && *endp == '\0' && v >= 0) {
                    out.push_back(static_cast<int>(v));
                }
            } else {
                std::string const lo_s = tok.substr(0, dash);
                std::string const hi_s = tok.substr(dash + 1);
                char*             endp2 = nullptr;
                long const lo = std::strtol(lo_s.c_str(), &endp, 10);
                long const hi = std::strtol(hi_s.c_str(), &endp2, 10);
                bool const ok = endp != lo_s.c_str() && *endp == '\0'
                                && endp2 != hi_s.c_str() && *endp2 == '\0'
                                && lo >= 0 && hi >= lo;
                if (ok) {
                    for (long v = lo; v <= hi; ++v) out.push_back(static_cast<int>(v));
                }
            }
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string format_cpu_list(std::vector<int> const& cpus)
{
    if (cpus.empty()) return {};

    std::vector<int> s(cpus);
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());

    std::string out;
    for (std::size_t i = 0; i < s.size();) {
        // Extend while the ids stay consecutive; that run becomes "lo-hi".
        std::size_t j = i;
        while (j + 1 < s.size() && s[j + 1] == s[j] + 1) ++j;
        if (!out.empty()) out += ',';
        out += std::to_string(s[i]);
        // A run of exactly two ("4-5") is no shorter than listing them, but a
        // single id must not become "4-4".
        if (j > i) {
            out += '-';
            out += std::to_string(s[j]);
        }
        i = j + 1;
    }
    return out;
}

std::vector<NumaNode> host_numa_nodes()
{
    std::vector<NumaNode> nodes;

    // "0" on a single-node host, "0-1" or "0,2" on bigger ones. Absent on
    // kernels built without NUMA, which is a legitimate answer, not a failure.
    auto const online = parse_cpu_list(read_sysfs("/sys/devices/system/node/online"));
    for (int id : online) {
        auto const cpus = parse_cpu_list(read_sysfs(
            "/sys/devices/system/node/node" + std::to_string(id) + "/cpulist"));
        // A node with no CPUs is real (CXL / memory-only nodes) and cannot host
        // a worker, so it is not a node for our purposes.
        if (!cpus.empty()) nodes.push_back(NumaNode{id, cpus});
    }

    if (nodes.empty()) {
        // No NUMA sysfs: one node that is the whole machine. Its cpu list stays
        // EMPTY rather than 0..hardware_concurrency()-1 — an empty list means
        // "do not pin", and inventing a mask from a CPU count we did not read
        // from the kernel is how you pin a worker to CPUs that aren't there
        // (offline cores, or a cgroup that already narrowed us).
        nodes.push_back(NumaNode{0, {}});
    }
    return nodes;
}

bool pin_thread_to_cpus(std::vector<int> const& cpus)
{
    if (cpus.empty()) return false;

#ifdef _WIN32
    // No affinity here yet. Returning false is honest — the caller logs that it
    // is running unpinned and carries on, which is the same graceful path a
    // Linux host takes when sched_setaffinity is refused. Windows would need
    // SetThreadGroupAffinity + processor groups, and none of the multi-socket
    // measurement this exists for has happened on Windows.
    (void)cpus;
    return false;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) {
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    }
    if (CPU_COUNT(&set) == 0) return false;

    // 0 = the calling thread. NOT the process: like nice, Linux scopes affinity
    // per-thread and copies it at clone(), which is exactly what carries this
    // down into pos2-chip's fan-out.
    return ::sched_setaffinity(0, sizeof(set), &set) == 0;
#endif
}

std::vector<int> current_thread_cpus()
{
    std::vector<int> out;
#ifndef _WIN32
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0) return out;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &set)) out.push_back(c);
    }
#endif
    return out;
}

ScopedThreadAffinity::ScopedThreadAffinity()
    : saved_(current_thread_cpus())
{
    // An empty mask is not something the kernel reports, but if it somehow did,
    // restoring it would pin the thread to nothing. Treat it as "no snapshot"
    // and leave the mask alone on the way out.
    valid_ = !saved_.empty();
}

ScopedThreadAffinity::~ScopedThreadAffinity()
{
    if (valid_) pin_thread_to_cpus(saved_);
}

}  // namespace pos2gpu
