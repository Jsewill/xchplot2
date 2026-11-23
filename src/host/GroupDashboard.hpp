#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace group_grow
{
struct DashboardFrame
{
    struct Row
    {
        std::string value, detail;
    };
    enum View : size_t
    {
        overview,
        resources,
        workers,
        decisions,
        farm,
        count
    };
    enum class Sharing : uint8_t
    {
        unknown,
        local,
        enrolled
    };
    Sharing sharing = Sharing::unknown;
    std::string title, summary;
    std::array<std::vector<Row>, count> rows;
    std::array<size_t, count> omitted{};
    void add(View tab, std::string value, std::string detail = {})
    {
        if (rows[tab].size() < 256) rows[tab].push_back({std::move(value), std::move(detail)});
        else ++omitted[tab];
    }
};

inline std::string dashboard_number(double value, unsigned precision = 2)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(int(precision)) << value;
    return out.str();
}

class DashboardJob
{
    using Clock = std::chrono::steady_clock;
    struct Delivery
    {
        std::string worker, instance, local_device;
        uint64_t members = 0, bytes = 0;
        double last_seconds = 0;
        uint64_t last_units = 0, last_bytes = 0, worker_ns = 0, send_ns = 0, receiver_ns = 0;
        uint32_t ingress = 0;
    };
    mutable std::mutex mutex_;
    Clock::time_point const started_;
    Clock::time_point first_publication_{}, last_publication_{};
    std::deque<std::pair<Clock::time_point, uint64_t>> publications_;
    std::deque<Delivery> deliveries_;
    std::deque<DashboardFrame::Row> decisions_;
    std::vector<DashboardFrame::Row> roster_;
    std::vector<DashboardFrame::Row> admission_rows_;
    std::map<std::string,std::string> recovering_local_gpus_;
    std::optional<unsigned> registered_;
    std::string roster_state_;
    struct Grant
    {
        std::string description;
        Clock::time_point since;
        double expected;
    };
    std::map<std::string, Grant> grants_;
    struct AdmissionWait
    {
        Clock::time_point since;
        std::string resource, holder;
    };
    std::map<std::string, AdmissionWait> waiting_;
    uint64_t decision_count_ = 0;
    uint64_t fresh_ = 0, inherited_ = 0, files_ = 0;
    // Saved farm output is ordered within a lane. Re-inspecting a revision or
    // extending its last file must count only the additional durable members.
    std::map<size_t, std::pair<uint64_t, uint64_t>> lane_publications_;

  public:
    explicit DashboardJob(Clock::time_point started = Clock::now()) : started_(started) {}
    struct Route
    {
        std::string id, instance, host, gpu, endpoint;
        unsigned slots;
    };
    struct Progress
    {
        uint64_t fresh, inherited, files;
        double first_seconds, last_seconds, elapsed_seconds;
    };
    using LaneFile = std::pair<size_t, uint64_t>;
    void published(uint64_t members, uint64_t inherited, std::optional<LaneFile> file = {})
    {
        if (!members) return;
        std::lock_guard lock(mutex_);
        inherited = std::min(members, inherited);
        bool new_file = true;
        if (file)
        {
            auto [at, inserted] = lane_publications_.try_emplace(file->first, file->second, members);
            if (!inserted)
            {
                auto& [ordinal, previous] = at->second;
                if (file->second < ordinal || (file->second == ordinal && members <= previous)) return;
                new_file = file->second > ordinal;
                auto const credited = new_file ? 0 : previous;
                ordinal = file->second;
                previous = members;
                members -= credited;
                inherited -= std::min(inherited, credited);
            }
        }
        if (new_file) ++files_;
        inherited_ += inherited;
        if (members == inherited) return;
        fresh_ += members - inherited;
        last_publication_ = Clock::now();
        if (first_publication_ == Clock::time_point{}) first_publication_ = last_publication_;
        publications_.emplace_back(last_publication_, fresh_);
        if (publications_.size() > 32) publications_.pop_front();
    }
    Progress progress() const
    {
        std::lock_guard lock(mutex_);
        return {fresh_, inherited_, files_,
                fresh_ ? std::chrono::duration<double>(first_publication_ - started_).count() : 0,
                fresh_ ? std::chrono::duration<double>(last_publication_ - started_).count() : 0,
                std::chrono::duration<double>(Clock::now() - started_).count()};
    }
    void inherited_snapshot(uint64_t files, uint64_t members)
    {
        std::lock_guard lock(mutex_);
        inherited_ += members;
        files_ += files;
    }
    void granted(std::string target, std::string description, double expected_seconds)
    {
        std::lock_guard lock(mutex_);
        waiting_.erase(target);
        if (grants_.size() < 256)
            grants_.insert_or_assign(std::move(target), Grant{std::move(description), Clock::now(), expected_seconds});
    }
    void waiting(std::string target, std::string resource, std::string holder)
    {
        std::lock_guard lock(mutex_);
        if (auto it = waiting_.find(target); it != waiting_.end())
        {
            it->second.resource = std::move(resource);
            it->second.holder = std::move(holder);
        }
        else if (waiting_.size() < 256)
            waiting_.emplace(std::move(target), AdmissionWait{Clock::now(), std::move(resource), std::move(holder)});
    }
    void released(std::string const& target, bool published)
    {
        std::lock_guard lock(mutex_);
        waiting_.erase(target);
        auto const it = grants_.find(target);
        if (it == grants_.end()) return;
        if (published && it->second.expected > 0)
        {
            auto const actual = std::chrono::duration<double>(Clock::now() - it->second.since).count();
            if (decisions_.size() == 32) decisions_.pop_front();
            ++decision_count_;
            decisions_.push_back({"Published  " + target,
                                  "Forecast for this group " + dashboard_number(it->second.expected) +
                                      " s, observed " + dashboard_number(actual) + " s from grant to publication"});
        }
        grants_.erase(it);
    }
    std::optional<std::string> delivery(std::string const& id, std::string const& instance, uint64_t members,
                                        uint64_t bytes, uint64_t ns, uint64_t worker_ns, uint64_t send_ns,
                                        uint64_t receiver_ns, uint32_t ingress, std::string local_device = {})
    {
        std::lock_guard lock(mutex_);
        auto at = std::find_if(deliveries_.begin(), deliveries_.end(),
                               [&](auto const& d) { return d.worker == id && d.instance == instance && d.local_device == local_device; });
        bool const first = at == deliveries_.end();
        if (first)
        {
            if (deliveries_.size() == 128) deliveries_.pop_front();
            deliveries_.push_back({id, instance, std::move(local_device)});
            at = std::prev(deliveries_.end());
        }
        at->members += members;
        at->bytes += bytes;
        at->last_seconds = double(ns) / 1e9;
        at->last_units = members;
        at->last_bytes = bytes;
        at->worker_ns = worker_ns;
        at->send_ns = send_ns;
        at->receiver_ns = receiver_ns;
        at->ingress = ingress;
        if (!first) return {};
        if (!at->local_device.empty()) return "Local " + at->local_device;
        auto const route = std::find_if(previous_routes_.begin(), previous_routes_.end(),
                                        [&](auto const& r) { return r.id == id && r.instance == instance; });
        return route == previous_routes_.end() || route->host.empty() ? id : route->host;
    }
    std::string roster(uint8_t k, uint8_t strength, unsigned slots, bool sampled, bool observed, bool complete,
                       std::vector<Route> const& routes, std::optional<unsigned> registered = {},
                       std::vector<DashboardFrame::Row> admission = {})
    {
        std::lock_guard lock(mutex_);
        std::string change;
        if (observed && complete && roster_complete_ && k == last_k_ && strength == last_strength_)
        {
            unsigned joined = 0, left = 0;
            auto has = [](auto const& list, Route const& candidate) {
                return std::any_of(list.begin(), list.end(), [&](auto const& r) {
                    return r.id == candidate.id && r.instance == candidate.instance;
                });
            };
            std::string first;
            for (auto const& route : routes)
                if (!has(previous_routes_, route))
                {
                    if (!joined++) first = route.host.empty() ? route.id : route.host;
                }
            for (auto const& route : previous_routes_)
                if (!has(routes, route)) ++left;
            if (joined)
                change = (joined == 1 ? first : std::to_string(joined) + " helpers") + " now eligible for new work";
            if (left)
                change += (change.empty() ? "" : ". ") + std::to_string(left) +
                          (left == 1 ? " helper left the eligible pool" : " helpers left the eligible pool");
        }
        if (change.empty() && registered && registered_ && *registered > *registered_)
            change = "Helper registered. Waiting for an eligible assignment.";
        registered_ = registered;
        admission_rows_ = std::move(admission);
        previous_routes_ = routes;
        roster_complete_ = observed && complete;
        last_k_ = k;
        last_strength_ = strength;
        roster_state_ = std::to_string(slots) + " eligible slots for k" + std::to_string(k) + " strength " +
                        std::to_string(strength);
        if (!sampled)
            roster_state_ = "Broker capacity unavailable. Waiting for a fresh observation.";
        else if (!observed)
            roster_state_ += "   roster unavailable";
        else if (!complete)
            roster_state_ += "   partial roster";
        roster_.clear();
        for (auto const& route : routes)
            roster_.push_back({(route.host.empty() ? route.id : route.host) + "   " +
                                   (route.gpu.empty() ? "CPU" : route.gpu) + "   " + std::to_string(route.slots) +
                                   " eligible slots",
                               route.endpoint + "   worker " + route.id});
        if (!change.empty())
        {
            ++decision_count_;
            if (decisions_.size() == 32) decisions_.pop_front();
            decisions_.push_back({change, "Existing grants finish. The owner re-evaluates remaining work."});
        }
        return change;
    }
    void local_gpu_health(std::string const& device, std::string const& cause)
    {
        std::lock_guard lock(mutex_);
        if (cause.empty()) recovering_local_gpus_.erase(device);
        else recovering_local_gpus_[device] = cause;
    }
    void decision(std::string chosen, std::string why)
    {
        std::lock_guard lock(mutex_);
        if (!decisions_.empty() && decisions_.back().value == chosen && decisions_.back().detail == why) return;
        ++decision_count_;
        if (decisions_.size() == 32) decisions_.pop_front();
        decisions_.push_back({std::move(chosen), std::move(why)});
    }
    void fill(DashboardFrame& frame, bool shared, uint64_t target = 0) const
    {
        std::lock_guard lock(mutex_);
        frame.sharing = shared ? DashboardFrame::Sharing::enrolled : DashboardFrame::Sharing::local;
        auto const now = Clock::now();
        auto const elapsed = std::chrono::duration<double>(now - started_).count();
        auto const last = std::chrono::duration<double>(last_publication_ - started_).count();
        frame.summary = std::to_string(fresh_) + " new durable members   " +
                        (fresh_ ? dashboard_number(last / fresh_) + " s/member through publication"
                                : "Waiting for first new publication");
        frame.add(DashboardFrame::overview,
                  std::to_string(files_) + " published files   " + std::to_string(inherited_) + " inherited members");
        frame.add(DashboardFrame::overview,
                  "Elapsed  " + dashboard_number(elapsed, 1) + " s" +
                      (fresh_ ? "    Last new publication  " + dashboard_number(last, 1) + " s" : ""),
                  fresh_
                      ? "First new publication at " +
                            dashboard_number(std::chrono::duration<double>(first_publication_ - started_).count(), 1) +
                            " s"
                      : "");
        auto release = [&](Grant const& grant) {
            if (!grant.expected) return std::string("Release time unmeasured");
            auto const remaining = grant.expected - std::chrono::duration<double>(now - grant.since).count();
            return remaining > 0 ? "Estimated release in " + dashboard_number(remaining, 1) + " s"
                                 : std::string("Past estimated release");
        };
        if (!waiting_.empty())
        {
            auto const& [target, wait] =
                *std::min_element(waiting_.begin(), waiting_.end(),
                                  [](auto const& a, auto const& b) { return a.second.since < b.second.since; });
            std::string detail = target + "   " + wait.resource + "   ";
            auto const held = grants_.find(wait.holder);
            if (held != grants_.end())
                detail += "Held by " + held->first + ". " + release(held->second);
            else
                detail += "Holder and release time unmeasured";
            frame.add(DashboardFrame::overview,
                      "Oldest waiting admission   " + wait.resource + "   " +
                          dashboard_number(std::chrono::duration<double>(now - wait.since).count(), 1) + " s",
                      detail);
            frame.add(DashboardFrame::decisions, "Waiting for admission   " + target, detail);
        }
        size_t shown_grants = 0;
        for (auto const& [target, grant] : grants_)
        {
            if (shown_grants++ == 32)
            {
                frame.add(DashboardFrame::decisions, std::to_string(grants_.size() - 32) + " more active grants");
                break;
            }
            frame.add(DashboardFrame::decisions, "Active grant   " + target,
                      grant.description + "   held " +
                          dashboard_number(std::chrono::duration<double>(now - grant.since).count(), 1) + " s. " +
                          release(grant));
        }
        if (publications_.size() > 1)
        {
            auto const& first = publications_.front();
            auto const rate =
                std::chrono::duration<double>(last_publication_ - first.first).count() / (fresh_ - first.second);
            frame.add(DashboardFrame::overview, "Recent  " + dashboard_number(rate) + " s/member across " +
                                                    std::to_string(fresh_ - first.second) + " members");
            if (publications_.size() >= 3)
            {
                std::vector<double> intervals;
                for (size_t i = 1; i < publications_.size(); ++i)
                {
                    auto const& [at, count] = publications_[i];
                    auto const& [before, previous_count] = publications_[i - 1];
                    intervals.push_back(std::chrono::duration<double>(at - before).count() / (count - previous_count));
                }
                auto const [low, high] = std::minmax_element(intervals.begin(), intervals.end());
                auto const remaining = target > fresh_ + inherited_ ? target - fresh_ - inherited_ : 0;
                frame.add(DashboardFrame::overview,
                          "Publication intervals  " + dashboard_number(*low) + " to " + dashboard_number(*high) +
                              " s/member",
                          std::to_string(intervals.size()) +
                              " observed intervals. Concurrent files can publish close together." +
                              (remaining ? " At those rates the remaining " + std::to_string(remaining) +
                                               " members take " + dashboard_number(remaining * *low, 0) + " to " +
                                               dashboard_number(remaining * *high, 0) + " s."
                                         : ""));
            }
            if (target > fresh_ + inherited_)
                frame.add(DashboardFrame::overview, "Estimated remaining at recent rate  " +
                                                        dashboard_number((target - fresh_ - inherited_) * rate, 0) +
                                                        " s");
        }
        if (target) frame.add(DashboardFrame::overview, "Target  " + std::to_string(target) + " members");
        frame.add(DashboardFrame::workers,
                  roster_state_.empty() ? (shared ? "Waiting for a broker capacity observation" : "Local plotting")
                                        : roster_state_);
        for (auto const& row : admission_rows_)
            frame.add(DashboardFrame::workers, row.value, row.detail);
        for (auto const& [device, cause] : recovering_local_gpus_)
            frame.add(DashboardFrame::workers, "Local " + device + "   GPU recovering", cause);
        auto delivered = [&](Delivery const& d, std::string const& name) {
            if (!d.local_device.empty())
            {
                frame.add(DashboardFrame::workers, "Local " + d.local_device + "   " + std::to_string(d.members) +
                    " delivered   last " + dashboard_number(d.last_seconds) + " s",
                    "Last " + std::to_string(d.last_units) + " members. Owner elapsed includes admission and in-process delivery.");
                return;
            }
            frame.add(DashboardFrame::workers,
                      name + "   " + std::to_string(d.members) + " delivered   last " +
                          dashboard_number(d.last_seconds) + " s",
                      "Last delivery " + std::to_string(d.last_units) + " members / " +
                          dashboard_number(double(d.last_bytes) / (1u << 20), 1) + " MiB. Worker " +
                          dashboard_number(double(d.worker_ns) / 1e9) + " s, send " +
                          dashboard_number(double(d.send_ns) / 1e9) + " s, receiver " +
                          dashboard_number(double(d.receiver_ns) / 1e9) + " s (may overlap)." +
                          (d.ingress ? " Owner interface index " + std::to_string(d.ingress) + "." : "") +
                          " Total validated " + dashboard_number(double(d.bytes) / (1u << 20), 1) + " MiB.");
        };
        for (size_t i = 0; i < previous_routes_.size(); ++i)
        {
            auto const& r = previous_routes_[i];
            frame.add(DashboardFrame::workers, roster_[i].value, roster_[i].detail);
            auto const d = std::find_if(deliveries_.begin(), deliveries_.end(), [&](auto const& entry) {
                return entry.local_device.empty() && entry.worker == r.id && entry.instance == r.instance;
            });
            if (d != deliveries_.end()) delivered(*d, r.host.empty() ? r.id : r.host);
        }
        for (auto const& d : deliveries_)
            if (!d.local_device.empty() || std::none_of(previous_routes_.begin(), previous_routes_.end(),
                             [&](auto const& r) { return r.id == d.worker && r.instance == d.instance; }))
                delivered(d, d.worker);
        frame.add(DashboardFrame::workers,
                  shared ? "Press a for helper setup" : "Shared work requires an enrolled plotting host");
        if (decisions_.empty()) frame.add(DashboardFrame::decisions, "Waiting for a construction decision");
        for (auto it = decisions_.rbegin(); it != decisions_.rend(); ++it)
            frame.add(DashboardFrame::decisions, it->value, it->detail);
        if (decision_count_ > decisions_.size())
            frame.add(DashboardFrame::decisions,
                      std::to_string(decision_count_ - decisions_.size()) + " earlier decisions outside this window");
        if (!decisions_.empty())
            frame.add(DashboardFrame::overview, "Latest decision  " + decisions_.back().value,
                      decisions_.back().detail);
    }

  private:
    std::vector<Route> previous_routes_;
    bool roster_complete_ = false;
    uint8_t last_k_ = 0, last_strength_ = 0;
};

// This terminal is an observer. Only stdin belonging to the same foreground
// terminal supplies keys. In particular a piped plan or password is never read.
class DashboardTerminal
{
    std::mutex mutex_;
    bool enabled_ = false, input_ = false, screen_ = false, help_ = false, adding_ = false, details_ = false,
         filtering_ = false;
    std::array<std::string, DashboardFrame::count> filters_;
    size_t tab_ = 0, row_ = 0, scroll_ = 0;
    std::pair<unsigned, unsigned> dimensions_{};
    DashboardFrame attached_frame_;
    static constexpr std::array views{"Overview", "Resources", "Workers", "Decisions", "Farm"};
#ifdef _WIN32
    HANDLE output_ = INVALID_HANDLE_VALUE;
    HANDLE console_input_ = INVALID_HANDLE_VALUE;
    DWORD old_mode_ = 0, old_input_mode_ = 0;
    int repeated_key_ = 0;
    WORD repeats_left_ = 0;
#else
    termios old_input_{};
    unsigned escape_ = 0;
#endif
    void leave()
    {
        if (screen_)
        {
            std::cerr << "\033[?25h\033[?1049l" << std::flush;
            screen_ = false;
        }
    }
    void stop()
    {
        leave();
#ifdef _WIN32
        if (input_) ::SetConsoleMode(console_input_, old_input_mode_);
        if (enabled_) ::SetConsoleMode(output_, old_mode_);
#else
        if (input_) ::tcsetattr(STDIN_FILENO, TCSANOW, &old_input_);
#endif
        enabled_ = input_ = false;
    }
    // Strip terminal controls in paths, worker names and status reasons. Keep
    // complete UTF-8 characters and use a conservative byte-width limit.
    static std::string printable(std::string_view source, size_t width)
    {
        std::string out;
        for (size_t i = 0; i < source.size() && out.size() < width;)
        {
            unsigned char c = source[i];
            size_t len = c < 128                  ? 1
                         : c >= 0xc2 && c <= 0xdf ? 2
                         : c >= 0xe0 && c <= 0xef ? 3
                         : c >= 0xf0 && c <= 0xf4 ? 4
                                                  : 0;
            if (len && i + len <= source.size() && out.size() + len <= width && c >= 32 && c != 127 &&
                !(c == 0xc2 && uint8_t(source[i + 1]) <= 0x9f) &&
                std::all_of(source.begin() + i + 1, source.begin() + i + len,
                            [](unsigned char b) { return (b & 0xc0) == 0x80; }))
            {
                out.append(source.substr(i, len));
                i += len;
            }
            else
            {
                out += '?';
                ++i;
            }
        }
        return out;
    }
    std::pair<unsigned, unsigned> size() const
    {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (::GetConsoleScreenBufferInfo(output_, &info))
            return {unsigned(info.srWindow.Right - info.srWindow.Left + 1),
                    unsigned(info.srWindow.Bottom - info.srWindow.Top + 1)};
#else
        winsize info{};
        if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &info) == 0 && info.ws_col && info.ws_row)
            return {info.ws_col, info.ws_row};
#endif
        return {80, 24};
    }

  public:
    static std::string label(std::string_view value) { return printable(value, 240); }
    static bool plain_terminal()
    {
        auto const* term = std::getenv("TERM");
        return term && std::string_view(term) == "dumb";
    }
    explicit DashboardTerminal(bool json)
    {
        if (plain_terminal()) return;
#ifdef _WIN32
        if (!::_isatty(::_fileno(stderr)) || (json && ::_isatty(::_fileno(stdout)))) return;
        output_ = ::GetStdHandle(STD_ERROR_HANDLE);
        enabled_ = ::GetConsoleMode(output_, &old_mode_) &&
                   ::SetConsoleMode(output_, old_mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        console_input_ = ::GetStdHandle(STD_INPUT_HANDLE);
        input_ = enabled_ && ::GetConsoleMode(console_input_, &old_input_mode_) &&
                 ::SetConsoleMode(console_input_, old_input_mode_ & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_VIRTUAL_TERMINAL_INPUT));
#else
        if (!::isatty(STDERR_FILENO)) return;
        struct stat dst{}, in{}, out{};
        if (::fstat(STDERR_FILENO, &dst)) return;
        if (json && !::fstat(STDOUT_FILENO, &out) && dst.st_rdev == out.st_rdev && ::isatty(STDOUT_FILENO)) return;
        enabled_ = true;
        if (::isatty(STDIN_FILENO) && !::fstat(STDIN_FILENO, &in) && in.st_rdev == dst.st_rdev &&
            ::tcgetpgrp(STDIN_FILENO) == ::getpgrp() && !::tcgetattr(STDIN_FILENO, &old_input_))
        {
            auto mode = old_input_;
            mode.c_lflag &= ~(ICANON | ECHO);
            mode.c_cc[VMIN] = 0;
            mode.c_cc[VTIME] = 0;
            input_ = ::tcsetattr(STDIN_FILENO, TCSANOW, &mode) == 0;
        }
#endif
    }
    ~DashboardTerminal() { stop(); }
    bool active()
    {
        std::lock_guard lock(mutex_);
        return enabled_;
    }
    bool interactive()
    {
        std::lock_guard lock(mutex_);
        return input_;
    }
    // Restore the normal screen before a published path, warning or final
    // result is printed. The next snapshot resumes the alternate screen.
    void suspend()
    {
        std::lock_guard lock(mutex_);
        leave();
    }
    template <class Print> void output(Print print)
    {
        std::lock_guard lock(mutex_);
        leave();
        print();
    }
    bool keys()
    {
        std::lock_guard lock(mutex_);
        if (!input_) return false;
        bool changed = false;
        for (unsigned i = 0; i < 32; ++i)
        {
            int c;
#ifdef _WIN32
            if (!repeats_left_)
            {
                // A pending release or resize is not a pending character.
                // Consume only queued records, including non-key events.
                DWORD pending = 0, read = 0;
                INPUT_RECORD record{};
                if (!::GetNumberOfConsoleInputEvents(console_input_, &pending) || !pending ||
                    !::ReadConsoleInputA(console_input_, &record, 1, &read) || !read) break;
                if (record.EventType != KEY_EVENT || !record.Event.KeyEvent.bKeyDown) continue;
                auto const& key = record.Event.KeyEvent;
                switch (key.wVirtualKeyCode)
                {
                    case VK_UP: repeated_key_ = 'k'; break;
                    case VK_DOWN: repeated_key_ = 'j'; break;
                    case VK_LEFT: repeated_key_ = 'h'; break;
                    case VK_RIGHT: repeated_key_ = 'l'; break;
                    default: repeated_key_ = uint8_t(key.uChar.AsciiChar); break;
                }
                if (!repeated_key_) continue;
                repeats_left_ = std::max<WORD>(1, key.wRepeatCount);
            }
            --repeats_left_;
            c = repeated_key_;
#else
            pollfd fd{STDIN_FILENO, POLLIN, 0};
            unsigned char byte;
            if (::poll(&fd, 1, 0) <= 0 || !(fd.revents & POLLIN) || ::read(STDIN_FILENO, &byte, 1) != 1) break;
            c = byte;
            if (c == 27)
            {
                escape_ = 1;
                continue;
            }
            if (escape_ == 1 && c == '[')
            {
                escape_ = 2;
                continue;
            }
            if (escape_ == 2) c = c == 'A' ? 'k' : c == 'B' ? 'j' : c == 'D' || c == 'Z' ? 'h' : c == 'C' ? 'l' : 0;
            escape_ = 0;
#endif
            changed = true;
            if (filtering_)
            {
                auto& filter = filters_[tab_];
                if (c == '\r' || c == '\n')
                    filtering_ = false;
                else if (c == 21)
                    filter.clear();
                else if ((c == 8 || c == 127) && !filter.empty())
                {
                    auto end = filter.size() - 1;
                    while (end && (uint8_t(filter[end]) & 0xc0) == 0x80)
                        --end;
                    filter.resize(end);
                }
                else if (c >= 32 && c != 127 && c <= 255 && filter.size() < 80)
                    filter.push_back(char(c));
                row_ = 0;
                continue;
            }
            if (c == '/')
            {
                filters_[tab_].clear();
                filtering_ = true;
                help_ = details_ = false;
                row_ = 0;
                continue;
            }
            if (c == 'q' || c == 'Q')
            {
                stop();
                break;
            }
            if (c == '?' || c == 'a')
            {
                bool const add = c == 'a';
                help_ = !help_ || adding_ != add;
                adding_ = add;
                details_ = false;
                scroll_ = 0;
                continue;
            }
            if (c == '\r' || c == '\n')
            {
                if (help_)
                    help_ = false;
                else
                    details_ = !details_;
                scroll_ = 0;
                continue;
            }
            if ((details_ || help_) && (c == 'j' || c == 'k'))
            {
                if (c == 'j')
                    ++scroll_;
                else if (scroll_)
                    --scroll_;
                continue;
            }
            if (c == 'j')
            {
                row_ = std::min<size_t>(255, row_ + 1);
                details_ = false;
                continue;
            }
            if (c == 'k')
            {
                if (row_) --row_;
                details_ = false;
                continue;
            }
            if (c >= '1' && c <= '5')
                tab_ = size_t(c - '1');
            else if (c == '\t' || c == 'l')
                tab_ = (tab_ + 1) % views.size();
            else if (c == 'h')
                tab_ = (tab_ + views.size() - 1) % views.size();
            else
                continue;
            row_ = scroll_ = 0;
            details_ = help_ = false;
        }
        return changed;
    }
    void observe(DashboardFrame frame)
    {
        attached_frame_ = std::move(frame);
        draw(attached_frame_, true);
    }
    void navigate()
    {
        bool changed = keys();
        {
            std::lock_guard lock(mutex_);
            changed = enabled_ && (changed || size() != dimensions_);
        }
        if (changed) draw(attached_frame_, true);
    }
    void draw(DashboardFrame const& frame, bool attached = false)
    {
        std::lock_guard lock(mutex_);
        if (!enabled_) return;
        auto [cols, rows] = size();
        dimensions_ = {cols, rows};
        unsigned width = std::clamp(cols, 2u, 240u) - 1, height = std::clamp(rows, 2u, 100u);
        std::ostringstream out;
        if (!screen_)
        {
            out << "\033[?1049h\033[?25l";
            screen_ = true;
        }
        out << "\033[H";
        unsigned used = 0;
        auto line = [&](std::string const& text) {
            if (used == height) return;
            if (used++) out << "\r\n";
            out << printable(text, width) << "\033[K";
        };
        std::vector<std::string> overlay;
        auto wrap = [&](std::string text) {
            text = printable(text, 4096);
            std::string_view remaining(text);
            while (!remaining.empty() && overlay.size() < 256)
            {
                size_t len = std::min<size_t>(width, remaining.size());
                if (len < remaining.size())
                {
                    auto const space = remaining.rfind(' ', len);
                    if (space != std::string_view::npos && space > 0)
                        len = space;
                    else
                        while (len > 1 && (uint8_t(remaining[len]) & 0xc0) == 0x80)
                            --len;
                }
                overlay.emplace_back(remaining.substr(0, len));
                remaining.remove_prefix(len);
                while (remaining.starts_with(' '))
                    remaining.remove_prefix(1);
            }
        };
        line("xchplot2  " + frame.title);
        std::string nav;
        for (size_t i = 0; i < views.size(); ++i)
        {
            if (i) nav += "  ";
            nav += i == tab_ ? std::string("[") + views[i] + "]" : views[i];
        }
        line(width >= nav.size() ? nav : std::string(views[tab_]) + "   Tab to change view");
        if (!help_ && !details_)
        {
            line(frame.summary);
            if (frame.omitted[tab_]) line(std::to_string(frame.omitted[tab_]) + " additional rows beyond this view");
        }
        if (help_ && !adding_)
        {
            wrap("Dashboard help");
            wrap("Tab or 1-5 changes view. Up and Down select a row. Enter opens or closes its details.");
            wrap("/ filters the current view by name or detail. Type a search and press Enter. Use / then Enter to "
                 "show all rows.");
            wrap("Press a for the commands to add a helper.");
            wrap(attached ? "q or Ctrl-C closes this observer." : "q returns to progress. Ctrl-C stops plotting.");
        }
        else if (help_)
        {
            if (frame.sharing == DashboardFrame::Sharing::local)
            {
                wrap("This job uses local compute.");
                wrap("For the next shared job, run xchplot2 compute setup on the plotting host and use automatic "
                     "compute.");
            }
            else
                wrap("Add a helper during plotting");
            wrap("1. Create an invitation on the pool administrator.");
            wrap("xchplot2 compute invite helper.invite --name helper");
            wrap("2. Transfer the invitation privately. On the helper run");
            wrap("xchplot2 compute setup helper.invite");
            wrap("3. Open Workers to see eligibility and the first delivery.");
            wrap("xchplot2 compute status -v shows registrations and queue details.");
            wrap(attached ? "q or Ctrl-C closes this observer." : "q returns to progress. Ctrl-C stops plotting.");
        }
        else
        {
            auto const& query = filters_[tab_];
            std::vector<DashboardFrame::Row const*> items;
            auto matches = [&](std::string const& value) {
                auto folded = [](unsigned char c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : int(c); };
                return std::search(value.begin(), value.end(), query.begin(), query.end(),
                                   [&](char x, char y) { return folded(x) == folded(y); }) != value.end();
            };
            for (auto const& item : frame.rows[tab_])
                if (query.empty() || matches(item.value) || matches(item.detail)) items.push_back(&item);
            if (filtering_ || !query.empty()) line("Filter  " + query + (filtering_ ? "_" : ""));
            if (!query.empty() && items.empty()) line("No matching rows");
            row_ = items.empty() ? 0 : std::min(row_, items.size() - 1);
            auto const room = height > used + 1 ? height - used - 1 : 0;
            auto const first = room && row_ >= room ? row_ - room + 1 : 0;
            if (details_ && !items.empty())
            {
                wrap(items[row_]->value);
                if (!items[row_]->detail.empty()) wrap(items[row_]->detail);
            }
            else
                for (size_t i = first; i < items.size() && used + 1 < height; ++i)
                    line((i == row_ ? "> " : "  ") + items[i]->value);
        }
        if (help_ || details_)
        {
            auto room = height > used + 1 ? height - used - 1 : 0;
            scroll_ = std::min(scroll_, overlay.size() > room ? overlay.size() - room : 0);
            for (size_t i = scroll_; i < overlay.size() && used + 1 < height; ++i)
                line(overlay[i]);
            if (input_) line(width >= 34 ? "Arrows Scroll  Enter Back  q Close" : "Up/Down  Enter Back  q Exit");
        }
        else if (filtering_)
            line("Enter Apply  Ctrl-U Clear");
        else if (input_)
            line(width >= 72   ? "1-5 Views  Arrows Browse  Enter Details  / Filter  a Add helper  ? Help  q Close"
                 : width >= 35 ? "Tab Views  / Filter  a Add  ? Help  q Exit"
                               : "Tab Views  ? Help  q Exit");
        out << "\033[J";
        std::cerr << out.str() << std::flush;
    }
};
} // namespace group_grow
