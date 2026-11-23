#pragma once
#include "host/GroupDashboard.hpp"
#include "host/GroupHarvesterProfile.hpp"
#include "host/GroupStorage.hpp"
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#ifdef __linux__
#include <dlfcn.h>
#endif
#ifdef _WIN32
#include <psapi.h>
#endif

namespace group_grow
{
// Host counters are sampled exclusively by the observer. Physical counters
// include other processes, and phase byte counters include cached I/O.
class HardwareTelemetry
{
    using Clock = std::chrono::steady_clock;
    struct Io
    {
        uint64_t read = 0, write = 0, ops = 0, milliseconds = 0;
    };
    Clock::time_point previous_{};
    double cpu_ = 0;
    bool cpu_known_ = false;
    std::map<std::string, Io> last_;
    std::set<std::string> leaves_;
    static std::optional<uint64_t> number(std::filesystem::path const& p)
    {
        uint64_t v;
        std::ifstream in(p);
        if (in >> v) return v;
        return {};
    }
#ifdef __linux__
    // A PCI mass-storage function is the parent of its local block paths.
    // Attribute the selected paths once, including alternate SAS paths.
    // Device-mapper and filesystem counters are already excluded by leaves_.
    std::map<std::string, std::string> controller_by_disk_;
    std::map<std::string, std::filesystem::path> controllers_;
    std::set<std::string> nvme_subsystems_;
    void identify_controller(std::string const& leaf)
    {
        std::error_code ec;
        auto path = std::filesystem::canonical(std::filesystem::path("/sys/dev/block") / leaf, ec);
        while (!ec && path.has_parent_path() && path != path.root_path())
        {
            std::ifstream type(path / "class");
            unsigned value = 0;
            if (type >> std::hex >> value && (value >> 16) == 1 &&
                std::filesystem::read_symlink(path / "subsystem", ec).filename() == "pci")
            {
                auto const id = path.filename().string();
                controller_by_disk_.emplace("disk/" + leaf, id);
                controllers_.emplace(id, path);
                return;
            }
            path = path.parent_path();
        }
    }
    struct Nvml
    {
        struct Util
        {
            unsigned gpu, memory;
        };
        struct Memory
        {
            unsigned long long total, free, used;
        };
        void* library = ::dlopen("libnvidia-ml.so.1", RTLD_LOCAL | RTLD_LAZY);
        using Device = void*;
        using QueryCount = int (*)(unsigned*);
        using GetDevice = int (*)(unsigned, Device*);
        using GetUtil = int (*)(Device, Util*);
        using GetMemory = int (*)(Device, Memory*);
        using GetSensor = int (*)(Device, unsigned, unsigned*);
        using Control = int (*)();
        template <class T> T symbol(char const* name)
        {
            return library ? reinterpret_cast<T>(::dlsym(library, name)) : nullptr;
        }
        QueryCount count = symbol<QueryCount>("nvmlDeviceGetCount_v2");
        GetDevice device = symbol<GetDevice>("nvmlDeviceGetHandleByIndex_v2");
        GetUtil utilization = symbol<GetUtil>("nvmlDeviceGetUtilizationRates");
        GetMemory memory = symbol<GetMemory>("nvmlDeviceGetMemoryInfo");
        GetSensor clock = symbol<GetSensor>("nvmlDeviceGetClockInfo");
        GetSensor temperature = symbol<GetSensor>("nvmlDeviceGetTemperature");
        Control shutdown = symbol<Control>("nvmlShutdown");
        bool ready = false;
        Nvml()
        {
            auto init = symbol<Control>("nvmlInit_v2");
            ready = init && shutdown && count && device && !init();
        }
        ~Nvml()
        {
            if (ready) shutdown();
            if (library) ::dlclose(library);
        }
    };
    Nvml nvml_;
#endif
  public:
    void add(StorageDevice const& device)
    {
#ifdef __linux__
        auto admit = [&](std::string const& id) { if (leaves_.insert(id).second) identify_controller(id); };
        for (auto const& leaf : device.drives)
        {
            if (!leaf.starts_with("nvme:")) { admit(leaf); continue; }
            // Storage admission groups NVMe namespaces by subsystem NQN. The
            // kernel publishes statistics for visible whole namespaces.
            if (!nvme_subsystems_.insert(leaf).second) continue;
            std::error_code ec;
            for (auto const& block : std::filesystem::directory_iterator("/sys/class/block", ec))
            {
                if (!block.path().filename().string().starts_with("nvme")) continue;
                auto path = std::filesystem::canonical(block.path(), ec);
                if (ec || std::filesystem::exists(path / "partition", ec) || number(path / "hidden").value_or(0)) continue;
                std::ifstream subsystem(path.parent_path() / "subsysnqn"), dev(path / "dev");
                std::string nqn, id;
                if (subsystem >> nqn && "nvme:" + nqn == leaf && dev >> id) admit(id);
            }
        }
#else
        leaves_.insert(device.drives.begin(), device.drives.end());
#endif
    }
    void sample(DashboardFrame& frame, uint64_t ram, unsigned cpus, DiskIoGovernor const* bandwidth = nullptr)
    {
        auto const now = Clock::now();
        auto const dt = std::chrono::duration<double>(now - previous_).count();
        double cpu = 0;
        bool cpu_known = false;
        std::optional<uint64_t> rss;
#ifdef _WIN32
        FILETIME start{}, exit{}, kernel{}, user{};
        if (::GetProcessTimes(::GetCurrentProcess(), &start, &exit, &kernel, &user))
        {
            auto ns = [](FILETIME f) { return (uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
            cpu = double(ns(kernel) + ns(user)) / 1e7;
            cpu_known = true;
        }
        PROCESS_MEMORY_COUNTERS memory{};
        memory.cb = sizeof(memory);
        if (::K32GetProcessMemoryInfo(::GetCurrentProcess(), &memory, sizeof(memory))) rss = memory.WorkingSetSize;
#else
        timespec process_time{};
        if (!::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_time))
        {
            cpu = double(process_time.tv_sec) + double(process_time.tv_nsec) / 1e9;
            cpu_known = true;
        }
#ifdef __linux__
        std::ifstream statm("/proc/self/statm");
        uint64_t size, resident;
        long page = ::sysconf(_SC_PAGESIZE);
        if (page > 0 && statm >> size >> resident) rss = resident * uint64_t(page);
#endif
#endif
        auto add = [&](std::string s, std::string detail = {}) {
            frame.add(DashboardFrame::resources, std::move(s), std::move(detail));
        };
        if (cpu_known && cpu_known_ && dt > 0 && cpu >= cpu_)
            add("Owner CPU  " + dashboard_number((cpu - cpu_) / dt, 1) + " active cores   budget " +
                    std::to_string(cpus) + " threads",
                "Process CPU time over " + dashboard_number(dt, 1) + " s, including controller and receiver threads");
        else if (!cpu_known)
            add("Owner CPU usage unavailable");
        else
            add("Owner CPU  collecting interval   budget " + std::to_string(cpus) + " threads");
        auto mib = [](uint64_t bytes) { return std::to_string(bytes / (1024 * 1024)) + " MiB"; };
        add("Owner RAM  " + (rss ? mib(*rss) + " resident" : "resident unknown") + "   plotting cap " + mib(ram));
        auto const available = profile_detail::available_memory();
        if (available) add("Available host memory  " + mib(available));
#ifdef __linux__
        std::map<std::string, Io> next;
        struct ControllerTraffic
        {
            double read = 0, write = 0;
            unsigned paths = 0;
        };
        std::map<std::string, ControllerTraffic> controller_traffic;
        size_t disks_seen = 0;
        auto rate = [&](std::string const& key, std::string name, Io io, bool disk, std::string extra,
                        double link_mbps = 0) {
            auto old = last_.find(key);
            next.emplace(key, io);
            if (old == last_.end() || dt <= 0 || io.read < old->second.read || io.write < old->second.write) return;
            double r = double(io.read - old->second.read) / dt, w = double(io.write - old->second.write) / dt;
            extra += "   " + dashboard_number(dt, 1) + " s interval";
            if (disk)
            {
                if (auto owner = controller_by_disk_.find(key); owner != controller_by_disk_.end())
                {
                    auto& traffic = controller_traffic[owner->second];
                    traffic.read += r;
                    traffic.write += w;
                    ++traffic.paths;
                }
                if (io.ops > old->second.ops && io.milliseconds >= old->second.milliseconds)
                    extra += "   mean " +
                             dashboard_number(
                                 double(io.milliseconds - old->second.milliseconds) / (io.ops - old->second.ops), 1) +
                             " ms";
                add(name + "   read " + dashboard_number(r / 1048576, 1) + "   write " +
                        dashboard_number(w / 1048576, 1) + " MiB/s",
                    extra);
            }
            else
            {
                if (link_mbps > 0)
                    extra += "   RX " + dashboard_number(r * 8 / (link_mbps * 10000), 0) + "% / TX " +
                             dashboard_number(w * 8 / (link_mbps * 10000), 0) + "% of negotiated line rate";
                add(name + "   RX " + dashboard_number(r * 8 / 1e9) + "   TX " + dashboard_number(w * 8 / 1e9) +
                        " Gb/s",
                    extra);
            }
        };
        std::string line;
        std::ifstream disks("/proc/diskstats");
        while (std::getline(disks, line))
        {
            unsigned major, minor;
            std::string name;
            std::istringstream fields(line);
            if (!(fields >> major >> minor >> name)) continue;
            auto key = std::to_string(major) + ":" + std::to_string(minor);
            if (!leaves_.contains(key)) continue;
            ++disks_seen;
            uint64_t reads, writes, merged, rsec, wsec, rms, wms, inflight;
            if (fields >> reads >> merged >> rsec >> rms >> writes >> merged >> wsec >> wms >> inflight)
            {
                std::string detail = "Physical device  " + key + "   outstanding I/O " + std::to_string(inflight);
                if (bandwidth)
                {
                    auto const estimate = bandwidth->observed_bulk_bandwidth(key);
                    if (estimate)
                        detail += "   measured bulk service " +
                                  dashboard_number(double(estimate->first) * 1e9 / 1048576, 1) +
                                  " MiB/s, estimated available " +
                                  dashboard_number(double(estimate->second) * 1e9 / 1048576, 1) + " MiB/s";
                }
                rate("disk/" + key, name, {rsec * 512, wsec * 512, reads + writes, rms + wms}, true, std::move(detail));
            }
        }
        if (!disks_seen)
            add("Physical disk rates unavailable",
                "No requested local device could be matched to kernel disk counters");
        else if (previous_ == Clock::time_point{})
            add("Physical disk rates   collecting the first interval");
        for (auto const& [id, traffic] : controller_traffic)
        {
            auto const& path = controllers_.at(id);
            std::error_code ec;
            auto const driver = std::filesystem::read_symlink(path / "driver", ec).filename().string();
            std::string detail = std::to_string(traffic.paths) + " observed block paths over " +
                                 dashboard_number(dt, 1) + " s. Includes other users of those disks.";
            std::string speed;
            auto const lanes = number(path / "current_link_width");
            std::ifstream negotiated(path / "current_link_speed");
            if (lanes && std::getline(negotiated, speed))
                detail += "   Negotiated " + speed + " x" + std::to_string(*lanes);
            add("Controller " + id + (driver.empty() ? "" : " " + driver) + "   read " +
                    dashboard_number(traffic.read / 1048576, 1) + "   write " +
                    dashboard_number(traffic.write / 1048576, 1) + " MiB/s",
                std::move(detail));
        }
        std::ifstream net("/proc/net/dev");
        while (std::getline(net, line))
        {
            auto split = line.find(':');
            if (split == line.npos) continue;
            auto begin = line.find_first_not_of(' ');
            auto name = line.substr(begin, split - begin);
            if (name == "lo" || name.empty() || name.size() > 64 || name.find('/') != name.npos) continue;
            std::istringstream fields(line.substr(split + 1));
            uint64_t rx, tx, unused;
            if (!(fields >> rx >> unused >> unused >> unused >> unused >> unused >> unused >> unused >> tx)) continue;
            std::ifstream speed(std::filesystem::path("/sys/class/net") / name / "speed");
            long long mbps = 0;
            std::string detail = "All traffic on this host interface";
            if (speed >> mbps && mbps > 0) detail += "   negotiated " + std::to_string(mbps) + " Mb/s";
            rate("net/" + name, name, {rx, tx}, false, detail, double(mbps));
        }
        last_.swap(next);
        unsigned count = 0;
        bool measured_gpu = false;
        if (nvml_.ready && !nvml_.count(&count))
            for (unsigned i = 0; i < std::min(32u, count); ++i)
            {
                Nvml::Device d{};
                Nvml::Util u{};
                Nvml::Memory m{};
                if (nvml_.device(i, &d)) continue;
                auto s = "NVIDIA GPU " + std::to_string(i);
                if (nvml_.utilization && !nvml_.utilization(d, &u))
                {
                    s += "   " + std::to_string(u.gpu) + "% kernel-active";
                    measured_gpu = true;
                }
                if (nvml_.memory && !nvml_.memory(d, &m)) s += "   VRAM " + mib(m.used) + " / " + mib(m.total);
                std::string detail = "Device-wide NVML sample, including farming and other processes";
                unsigned value = 0;
                if (nvml_.clock && !nvml_.clock(d, 1, &value))
                    detail += "   SM clock " + std::to_string(value) + " MHz";
                if (nvml_.temperature && !nvml_.temperature(d, 0, &value))
                    detail += "   " + std::to_string(value) + " C";
                add(std::move(s), std::move(detail));
            }
        std::error_code ec;
        for (auto const& card : std::filesystem::directory_iterator("/sys/class/drm", ec))
        {
            auto name = card.path().filename().string();
            if (!name.starts_with("card") || name.size() <= 4 || name.find_first_not_of("0123456789", 4) != name.npos)
                continue;
            auto busy = number(card.path() / "device" / "gpu_busy_percent");
            if (!busy || *busy > 100) continue;
            std::string s = name + "   " + std::to_string(*busy) + "% GPU busy";
            if (auto used = number(card.path() / "device" / "mem_info_vram_used"))
                s += "   VRAM " + mib(*used) + " used";
            add(std::move(s), "DRM device-wide driver sample");
            measured_gpu = true;
        }
        if (!measured_gpu) add("GPU activity  unavailable from local drivers");
#else
        add("Physical disk, interface and GPU rates  unavailable on this platform");
#endif
        add("HBA utilization  unknown", "Physical disk throughput and controller link capacity have different scopes");
        cpu_ = cpu;
        cpu_known_ = cpu_known;
        previous_ = now;
    }
};
} // namespace group_grow
