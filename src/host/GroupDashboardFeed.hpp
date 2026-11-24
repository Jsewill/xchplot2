// Read-only attachment to a proven local farming owner. This transient page
// contains display rows only. An observer never reopens the plotting data.
#pragma once
#include "host/GroupDashboard.hpp"
#include "host/GroupFarmOwner.hpp"
#include <cstring>
#include <memory>
#ifdef __linux__
#include <dirent.h>
#include <sys/file.h>
#include <sys/mman.h>
#elif defined(_WIN32)
#include "host/WindowsFile.hpp"
#endif

namespace group_grow
{
class DashboardFeed
{
    using Clock = std::chrono::steady_clock;
    struct Key
    {
        std::string job;
        uint64_t pid, start;
        bool operator==(Key const&) const = default;
        std::string name() const { return "xchplot2-dashboard-" + job + '-' + std::to_string(pid) + '-' + std::to_string(start); }
    };
    struct Page
    {
        uint32_t version = 3, bytes = sizeof(Page);
        uint32_t sharing = 0, used = 0;
        uint64_t pid = 0, start = 0, at_ms = 0;
        char job[65]{}, title[192]{}, summary[256]{};
        std::array<uint32_t, DashboardFrame::count> totals{};
        std::array<char, 60 * 1024> data{};
    };
    static_assert(std::is_trivially_copyable_v<Page> && sizeof(Page) <= 64 * 1024);
    template<size_t N> static void copy(char (&dest)[N], std::string const& source)
    { std::memcpy(dest, source.data(), std::min(N - 1, source.size())); }
    template<size_t N> static std::string string(char const (&value)[N])
    { return {value, std::find(value, value + N, '\0')}; }
    static uint64_t process_start(uint64_t pid)
    {
        uint64_t result = 0;
#if defined(__linux__) || defined(_WIN32)
        auto const start = farming_owner_detail::process_start(pid);
        auto parsed = std::from_chars(start.data(), start.data() + start.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != start.data() + start.size()) return 0;
#endif
        return result;
    }
    struct Endpoint
    {
#ifdef __linux__
        int fd = -1;
        ~Endpoint() { if (fd >= 0) ::close(fd); }
        bool create(Key const& key)
        {
            fd = ::memfd_create(key.name().c_str(), MFD_CLOEXEC | MFD_ALLOW_SEALING);
            return fd >= 0 && !::fchmod(fd, S_IRUSR) && !::ftruncate(fd, sizeof(Page)) &&
                !::fcntl(fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL);
        }
        bool attach(Key const& key)
        {
            auto const root = "/proc/" + std::to_string(key.pid) + "/fd/";
            auto const target = "/memfd:" + key.name() + " (deleted)";
            struct Directory { DIR* value; ~Directory() { if (value) ::closedir(value); } } directory{::opendir(root.c_str())};
            if (!directory.value) return false;
            auto const until = Clock::now() + std::chrono::milliseconds(20);
            for (unsigned walked = 0; walked < 4096 && Clock::now() < until; ++walked)
            {
                auto* entry = ::readdir(directory.value);
                if (!entry) break;
                if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
                std::array<char, 256> link{};
                auto const path = root + entry->d_name;
                auto const n = ::readlink(path.c_str(), link.data(), link.size());
                if (n < 0 || std::string_view(link.data(), size_t(n)) != target) continue;
                // Pin before reopening so descriptor recycling cannot turn the
                // read into an open of some other type of file or device.
                farm::inventory_detail::Handle pinned(::open(path.c_str(), O_PATH | O_CLOEXEC));
                struct stat info{};
                if (pinned.fd < 0 || ::fstat(pinned.fd, &info) || !S_ISREG(info.st_mode) || info.st_nlink ||
                    info.st_uid != ::geteuid() || (info.st_mode & 0777) != S_IRUSR || info.st_size != sizeof(Page)) continue;
                auto const own = "/proc/self/fd/" + std::to_string(pinned.fd);
                auto const check = ::readlink(own.c_str(), link.data(), link.size());
                if (check < 0 || std::string_view(link.data(), size_t(check)) != target) continue;
                fd = ::open(own.c_str(), O_RDONLY | O_CLOEXEC);
                return fd >= 0 && !::flock(fd, LOCK_SH | LOCK_NB);
            }
            return false;
        }
        bool interested() const
        {
            if (::flock(fd, LOCK_EX | LOCK_NB)) return errno == EWOULDBLOCK || errno == EAGAIN;
            (void)::flock(fd, LOCK_UN); return false;
        }
        bool lock(short mode) const
        {
            struct flock region{}; region.l_type = mode; region.l_whence = SEEK_SET; region.l_len = 1;
            return !::fcntl(fd, F_OFD_SETLK, &region);
        }
        bool transfer(Page& page, bool writing) const
        {
            if (!lock(writing ? F_WRLCK : F_RDLCK)) return false;
            auto const bytes = writing ? ::pwrite(fd, &page, sizeof(page), 0) : ::pread(fd, &page, sizeof(page), 0);
            (void)lock(F_UNLCK); return bytes == sizeof(page);
        }
#elif defined(_WIN32)
        HANDLE mapping = nullptr, guard = nullptr, interest = nullptr;
        void* view = nullptr;
        mutable Clock::time_point wanted_until{};
        ~Endpoint()
        {
            if (view) ::UnmapViewOfFile(view);
            if (interest) ::CloseHandle(interest);
            if (guard) ::CloseHandle(guard);
            if (mapping) ::CloseHandle(mapping);
        }
        bool open(Key const& key, bool writing)
        {
            auto const text = key.name();
            auto const name = L"Local\\" + std::wstring(text.begin(), text.end());
            auto const mutex_name = name + L"-guard", event_name = name + L"-interest";
            pos2gpu::PrivateFileSecurity security;
            if (writing)
            {
                mapping = ::CreateFileMappingW(INVALID_HANDLE_VALUE, &security.attributes, PAGE_READWRITE, 0, sizeof(Page), name.c_str());
                if (!mapping || ::GetLastError() == ERROR_ALREADY_EXISTS) return false;
                guard = ::CreateMutexW(&security.attributes, FALSE, mutex_name.c_str());
                if (!guard || ::GetLastError() == ERROR_ALREADY_EXISTS) return false;
                interest = ::CreateEventW(&security.attributes, FALSE, FALSE, event_name.c_str());
                if (!interest || ::GetLastError() == ERROR_ALREADY_EXISTS) return false;
            }
            else
            {
                mapping = ::OpenFileMappingW(FILE_MAP_READ | READ_CONTROL, FALSE, name.c_str());
                if (!mapping) return false;
                PSID owner = nullptr; PSECURITY_DESCRIPTOR descriptor = nullptr;
                auto const result = ::GetSecurityInfo(mapping, SE_KERNEL_OBJECT, OWNER_SECURITY_INFORMATION,
                    &owner, nullptr, nullptr, nullptr, &descriptor);
                bool const own = result == ERROR_SUCCESS && owner && ::EqualSid(owner, security.sid());
                if (descriptor) ::LocalFree(descriptor);
                if (!own) return false;
                guard = ::OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, mutex_name.c_str());
                interest = ::OpenEventW(EVENT_MODIFY_STATE, FALSE, event_name.c_str());
                if (!guard || !interest) return false;
            }
            view = ::MapViewOfFile(mapping, writing ? FILE_MAP_WRITE : FILE_MAP_READ, 0, 0, sizeof(Page));
            return view != nullptr;
        }
        bool create(Key const& key) { return open(key, true); }
        bool attach(Key const& key) { return open(key, false); }
        bool interested() const
        {
            auto const now = Clock::now();
            if (::WaitForSingleObject(interest, 0) == WAIT_OBJECT_0) wanted_until = now + std::chrono::seconds(5);
            return now < wanted_until;
        }
        bool transfer(Page& page, bool writing) const
        {
            if (!writing) ::SetEvent(interest);
            auto const lock = ::WaitForSingleObject(guard, 0);
            if (lock != WAIT_OBJECT_0 && lock != WAIT_ABANDONED) return false;
            if (writing) std::memcpy(view, &page, sizeof(page));
            else std::memcpy(&page, view, sizeof(page));
            ::ReleaseMutex(guard); return true;
        }
#else
        bool create(Key const&) { return false; }
        bool attach(Key const&) { return false; }
        bool interested() const { return false; }
        bool transfer(Page&, bool) const { return false; }
#endif
    };
    struct Reader { Key key; Clock::time_point seen, next; std::unique_ptr<Endpoint> endpoint; };
    Key key_{};
    std::unique_ptr<Endpoint> endpoint_;
public:
    explicit DashboardFeed(std::string const& job) noexcept
    {
        try
        {
            if (!farming_update_digest(job)) return;
#ifdef _WIN32
            auto const pid = uint64_t(::GetCurrentProcessId());
#else
            auto const pid = uint64_t(::getpid());
#endif
            key_ = {job, pid, process_start(pid)};
            auto endpoint = std::make_unique<Endpoint>();
            if (key_.start && endpoint->create(key_)) endpoint_ = std::move(endpoint);
        }
        catch (...) {} // An unavailable observer cannot interrupt construction.
    }
    bool wants_frame() const { return endpoint_ && endpoint_->interested(); }
    void publish(DashboardFrame const& frame) noexcept
    {
        if (!endpoint_) return;
        try
        {
            Page page; page.pid = key_.pid; page.start = key_.start; page.at_ms = farming_owner_detail::now_ms();
            page.sharing = static_cast<uint32_t>(frame.sharing);
            copy(page.job, key_.job); copy(page.title, frame.title); copy(page.summary, frame.summary);
            std::array<bool, DashboardFrame::count> full{};
            for (size_t i = 0; i < page.totals.size(); ++i)
                page.totals[i] = uint32_t(std::min<size_t>(UINT32_MAX, frame.rows[i].size() + frame.omitted[i]));
            // Interleave views so a large device inventory leaves room for
            // actions and farming evidence. Each view retains its own order.
            for (size_t row = 0; row < 256; ++row)
            {
                bool added = false;
                for (size_t i = 0; i < frame.rows.size(); ++i)
                {
                    if (full[i] || row >= frame.rows[i].size()) continue;
                    auto const& source = frame.rows[i][row];
                    if (source.value.size() > UINT16_MAX || source.detail.size() > UINT16_MAX ||
                        5 + source.value.size() + source.detail.size() > page.data.size() - page.used)
                    { full[i] = true; continue; }
                    auto const value = uint16_t(source.value.size()), detail = uint16_t(source.detail.size());
                    auto* out = page.data.data() + page.used;
                    out[0] = char(i); std::memcpy(out + 1, &value, 2); std::memcpy(out + 3, &detail, 2);
                    std::memcpy(out + 5, source.value.data(), value);
                    std::memcpy(out + 5 + value, source.detail.data(), detail);
                    page.used += 5u + value + detail; added = true;
                }
                if (!added) break;
            }
            (void)endpoint_->transfer(page, true);
        }
        catch (...) {}
    }
    static std::optional<DashboardFrame> read(std::string const& job, FarmingOwnerHealth const& owner, uint64_t* age_ms = nullptr) noexcept
    {
        try
        {
            if (!owner.local_process_proven || !owner.lock_held || !owner.heartbeat_current || owner.last_state == "stopped" ||
                !owner.local_pid || !owner.local_process_start || !farming_update_digest(job) ||
                process_start(owner.local_pid) != owner.local_process_start) return {};
            Key key{job, owner.local_pid, owner.local_process_start};
            static std::mutex mutex;
            static std::vector<Reader> readers;
            std::unique_lock lock(mutex, std::try_to_lock);
            if (!lock.owns_lock()) return {};
            auto const now = Clock::now();
            std::erase_if(readers, [&](auto const& r) { return now - r.seen > std::chrono::seconds(10); });
            auto at = std::find_if(readers.begin(), readers.end(), [&](auto const& r) { return r.key == key; });
            if (at == readers.end())
            {
                if (readers.size() == 16) readers.erase(std::min_element(readers.begin(), readers.end(),
                    [](auto const& a, auto const& b) { return a.seen < b.seen; }));
                at = readers.insert(readers.end(), {key, now, {}, {}});
            }
            at->seen = now;
            if (!at->endpoint && now >= at->next)
            {
                at->next = now + std::chrono::seconds(1);
                auto endpoint = std::make_unique<Endpoint>();
                if (endpoint->attach(key)) at->endpoint = std::move(endpoint);
            }
            Page page;
            if (!at->endpoint || !at->endpoint->transfer(page, false)) return {};
            auto const wall = farming_owner_detail::now_ms();
            if (page.version != 3 || page.bytes != sizeof(Page) || page.used > page.data.size() || page.pid != key.pid || page.start != key.start ||
                string(page.job) != key.job || !page.at_ms || page.at_ms > wall) return {};
            if (wall - page.at_ms > 5000) { at->endpoint.reset(); return {}; }
            DashboardFrame frame; frame.title = string(page.title); frame.summary = string(page.summary);
            if (page.sharing <= static_cast<uint32_t>(DashboardFrame::Sharing::enrolled))
                frame.sharing = static_cast<DashboardFrame::Sharing>(page.sharing);
            for (uint32_t offset = 0; offset < page.used;)
            {
                if (page.used - offset < 5) return {};
                auto const* in = page.data.data() + offset;
                auto const tab = size_t(uint8_t(in[0])); uint16_t value, detail;
                std::memcpy(&value, in + 1, 2); std::memcpy(&detail, in + 3, 2);
                if (tab >= frame.rows.size() || 5u + value + detail > page.used - offset ||
                    frame.rows[tab].size() >= std::min<uint32_t>(256, page.totals[tab])) return {};
                frame.add(DashboardFrame::View(tab), {in + 5, value}, {in + 5 + value, detail});
                offset += 5u + value + detail;
            }
            for (size_t i = 0; i < page.totals.size(); ++i) if (page.totals[i] > frame.rows[i].size())
            {
                if (frame.rows[i].size() == 256) frame.rows[i].pop_back();
                frame.add(DashboardFrame::View(i), std::to_string(page.totals[i] - frame.rows[i].size()) + " additional rows on the plotting host");
            }
            if (age_ms) *age_ms = wall - page.at_ms;
            return frame;
        }
        catch (...) { return {}; }
    }
};
} // namespace group_grow
