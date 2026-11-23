// Dated execution evidence only. All storage/workspace admission stays
// with the ordinary planner and every file remains independently recoverable.
#pragma once
#include "host/GroupPolicy.hpp"
#include "host/GroupFarmState.hpp"
#include "host/GroupFarmMetadata.hpp"

namespace group_grow
{
class ConstructionCache
{
public:
    enum class Kind { Publication, Generation };
private:
    using Clock = std::chrono::system_clock;
    struct Sample { std::string key; bool external; uint64_t at, elapsed; };
    static constexpr size_t limit = 48;
    static constexpr uint64_t lifetime = 24 * 60 * 60;
    Kind kind_;
    char const* filename() const { return kind_ == Kind::Publication ? "publication-v1" : "generation-v1"; }
    char const* format() const { return kind_ == Kind::Publication ? "xchplot2-publication-v1" : "xchplot2-generation-v1"; }
    std::filesystem::path directory_;
    std::optional<farm::inventory_detail::RootHandle> root_;
    std::optional<stream_state::WorkLock> lock_;
    std::optional<stream_state::Stamp> observed_;
    std::vector<Sample> prior_, recent_;
    int exceptions_ = std::uncaught_exceptions();
    static uint64_t now() { return uint64_t(std::chrono::duration_cast<std::chrono::seconds>(Clock::now().time_since_epoch()).count()); }
    static bool valid(std::string const& key)
    { return key.size() == 64 && key.find_first_not_of("0123456789abcdef") == key.npos; }
    static bool fresh(Sample const& value, uint64_t at)
    { return value.at && value.at <= at && at - value.at <= lifetime; }
    static void append(std::vector<Sample>& values, Sample value)
    { if (values.size() == limit) values.erase(values.begin()); values.push_back(std::move(value)); }
    std::optional<stream_state::Stamp> observed() const
    {
        auto const path = directory_ / filename();
        file_io::Stat info{};
        if (file_io::inspect_link(path, &info) && errno == ENOENT) return {};
        return stream_state::stamp(path);
    }
    void load()
    {
        auto const path = directory_ / filename();
        file_io::Stat info{};
        if (file_io::inspect_link(path, &info)) return;
        if (!file_io::is_regular(info.st_mode) || info.st_nlink != 1 || info.st_size <= 65 || info.st_size > 8192) return;
        farm::inventory_detail::Handle file(file_io::open_file(path, file_io::read_only | file_io::no_follow | file_io::close_on_exec));
        auto const opened = farm::inventory_detail::stat(file);
        if (opened.st_dev != info.st_dev || opened.st_ino != info.st_ino || opened.st_size != info.st_size) return;
#ifdef _WIN32
        pos2gpu::validate_private_file_handle(file_io::handle(file.fd), false);
#else
        if (opened.st_uid != ::geteuid() || (opened.st_mode & 0777) != 0600) return;
#endif
        std::string text(size_t(opened.st_size), '\0');
        if (file_io::read_at(file.fd, text.data(), text.size(), 0) != file_io::Count(text.size()) || text[64] != '\n') return;
        auto const body = text.substr(65);
        if (stream_state::digest(body) != text.substr(0, 64)) return;
        std::istringstream input(body); std::string magic; size_t count = 0;
        if (!(input >> magic >> count) || magic != format() || count > limit) return;
        std::vector<Sample> saved; saved.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            Sample s{};
            if (!(input >> s.key >> s.external >> s.at >> s.elapsed) || !valid(s.key) || !s.elapsed) return;
            if (fresh(s, now())) saved.push_back(std::move(s));
        }
        input >> std::ws; if (input.eof()) prior_ = std::move(saved);
    }
public:
    static std::filesystem::path location(Kind kind = Kind::Publication)
    {
        auto const folder = kind == Kind::Publication ? "publications" : "compute-performance";
#ifdef _WIN32
        if (auto path = ::_wgetenv(L"LOCALAPPDATA"); path && *path) return std::filesystem::path(path) / L"xchplot2" / L"cache" / folder;
#else
        if (auto path = std::getenv("XDG_CACHE_HOME"); path && *path) return std::filesystem::path(path) / "xchplot2" / folder;
        if (auto path = std::getenv("HOME"); path && *path) return std::filesystem::path(path) / ".cache" / "xchplot2" / folder;
#endif
        return {};
    }
    explicit ConstructionCache(std::filesystem::path directory, Kind kind = Kind::Publication) noexcept : kind_(kind)
    {
        try
        {
            if (!directory.is_absolute()) return;
            std::filesystem::create_directories(directory.parent_path());
            farm::inventory_detail::RootHandle parent(directory.parent_path());
            if (file_io::make_directory(directory, 0700) && errno != EEXIST) return;
            root_.emplace(directory); farm_metadata_detail::private_directory(parent, *root_);
            directory_ = std::move(directory);
            // A competing invoking process stays cold. A crash releases the
            // OS lock and the atomic replacement preserves the preceding file.
            lock_.emplace(directory_); observed_ = observed(); load();
            if (observed_ != observed()) lock_.reset();
        }
        catch (...) { lock_.reset(); }
    }
    ConstructionCache(ConstructionCache const&) = delete;
    ConstructionCache& operator=(ConstructionCache const&) = delete;
    bool available() const { return lock_.has_value(); }
    static std::chrono::seconds remaining(ConstructionPrior const& prior, uint64_t at = now())
    {
        if (!prior.samples || !prior.oldest_utc || at < prior.oldest_utc || at - prior.oldest_utc >= lifetime) return {};
        return std::chrono::seconds(lifetime - (at - prior.oldest_utc));
    }
    ConstructionPrior evidence(std::string const& key, bool external, uint64_t at = now()) const
    {
        ConstructionPrior result;
        for (auto i = prior_.rbegin(); i != prior_.rend() && result.samples < 8; ++i)
        {
            if (i->key != key || i->external != external || !fresh(*i, at)) continue;
            if (!result.samples) { result.low_ns = i->elapsed; result.oldest_utc = i->at; }
            result.low_ns = std::min(result.low_ns, i->elapsed); result.high_ns = std::max(result.high_ns, i->elapsed);
            result.oldest_utc = std::min(result.oldest_utc, i->at); ++result.samples;
        }
        return result.samples >= 2 ? result : ConstructionPrior{};
    }
    void completed(std::string const& key, bool external, uint64_t elapsed, uint64_t at = now()) noexcept
    {
        try { if (available() && valid(key) && elapsed) append(recent_, {key, external, at, elapsed}); } catch (...) {}
    }
    ~ConstructionCache()
    {
        if (!available() || recent_.empty() || exceptions_ != std::uncaught_exceptions()) return;
        try
        {
            farm_metadata_detail::same_directory(*root_);
            if (observed_ != observed()) return;
            for (auto& s : recent_) append(prior_, std::move(s));
            std::erase_if(prior_, [&](auto const& sample) { return !fresh(sample, now()); });
            std::ostringstream out; out << format() << ' ' << prior_.size() << '\n';
            for (auto const& s : prior_) out << s.key << ' ' << s.external << ' ' << s.at << ' ' << s.elapsed << '\n';
            auto const body = out.str();
            stream_state::atomic_save(directory_ / filename(), stream_state::digest(body) + '\n' + body);
        }
        catch (...) {}
    }
};
} // namespace group_grow
