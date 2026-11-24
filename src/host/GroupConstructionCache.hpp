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
    enum class Kind { Publication, Generation, PoolPublication, Fill };
    enum class Access { read_write, read_only };
private:
    using Clock = std::chrono::system_clock;
    struct Sample { std::string key; bool external; uint64_t at, elapsed; std::vector<uint64_t> phases; };
    static constexpr size_t limit = 48;
    static constexpr size_t max_phases = 8; // age and total service of up to four committed peers
    static constexpr uint64_t lifetime = 24 * 60 * 60;
    Kind kind_;
    struct File { char const* folder; char const* name; char const* format; };
    static constexpr std::array<File, 4> files{{
        {"publications", "publication-v1", "xchplot2-publication-v1"},
        {"compute-performance", "generation-v1", "xchplot2-generation-v1"},
        {"pool-publications", "pool-publication-v1", "xchplot2-pool-publication-v1"},
        {"fill-performance", "fill-v1", "xchplot2-fill-v1"},
    }};
    char const* filename() const { return files[size_t(kind_)].name; }
    char const* format() const { return files[size_t(kind_)].format; }
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
    void append(std::vector<Sample>& values, Sample value) const
    {
        auto drop = values.end();
        if (kind_ == Kind::PoolPublication || kind_ == Kind::Fill)
        {
            // Keep up to eight whole observations of each form. A long series
            // of winners never displaces its opposing form or tears a peer row.
            auto same = [&](Sample const& s) { return s.key == value.key && s.external == value.external; };
            if (std::count_if(values.begin(), values.end(), same) >= 8) drop = std::find_if(values.begin(), values.end(), same);
            else if (values.size() == limit) drop = std::find_if(values.begin(), values.end(),
                [&](Sample const& s) { return s.key != value.key; });
        }
        if (drop == values.end() && values.size() == limit) drop = values.begin();
        if (drop != values.end()) values.erase(drop);
        values.push_back(std::move(value));
    }
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
        auto const max_bytes = kind_ == Kind::PoolPublication || kind_ == Kind::Fill ? 16384 : 8192;
        if (!file_io::is_regular(info.st_mode) || info.st_nlink != 1 || info.st_size <= 65 || info.st_size > max_bytes) return;
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
            if (kind_ == Kind::PoolPublication || kind_ == Kind::Fill)
            {
                size_t phases = 0;
                if (!(input >> phases) || phases > max_phases || phases % 2) return;
                s.phases.resize(phases);
                for (auto& duration : s.phases) if (!(input >> duration) || !duration) return;
            }
            if (fresh(s, now())) saved.push_back(std::move(s));
        }
        input >> std::ws; if (input.eof()) prior_ = std::move(saved);
    }
public:
    static std::filesystem::path location(Kind kind = Kind::Publication)
    {
        auto const folder = files[size_t(kind)].folder;
#ifdef _WIN32
        if (auto path = ::_wgetenv(L"LOCALAPPDATA"); path && *path) return std::filesystem::path(path) / L"xchplot2" / L"cache" / folder;
#else
        if (auto path = std::getenv("XDG_CACHE_HOME"); path && *path) return std::filesystem::path(path) / "xchplot2" / folder;
        if (auto path = std::getenv("HOME"); path && *path) return std::filesystem::path(path) / ".cache" / "xchplot2" / folder;
#endif
        return {};
    }
    explicit ConstructionCache(std::filesystem::path directory, Kind kind = Kind::Publication,
                               Access access = Access::read_write) noexcept : kind_(kind)
    {
        try
        {
            if (!directory.is_absolute()) return;
            if (access == Access::read_write) std::filesystem::create_directories(directory.parent_path());
            farm::inventory_detail::RootHandle parent(directory.parent_path());
            if (access == Access::read_write && file_io::make_directory(directory, 0700) && errno != EEXIST) return;
            root_.emplace(directory); farm_metadata_detail::private_directory(parent, *root_);
            directory_ = std::move(directory);
            if (access == Access::read_only)
            {
                // Publications are atomically replaced. A preview takes no
                // writer lock and can safely discard a concurrently changed sample.
                auto const before = observed(); load();
                if (before != observed()) prior_.clear();
                return;
            }
            // A competing invoking process stays cold. A crash releases the
            // OS lock and the atomic replacement preserves the preceding file.
            lock_.emplace(directory_); observed_ = observed(); load();
            if (observed_ != observed()) lock_.reset();
        }
        catch (...) { lock_.reset(); prior_.clear(); }
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
    { return phase_evidence(key, external, 0, at); }
    ConstructionPrior phase_evidence(std::string const& key, bool external, size_t phase, uint64_t at = now()) const
    {
        ConstructionPrior result;
        for (auto i = prior_.rbegin(); i != prior_.rend() && result.samples < 8; ++i)
        {
            if (i->key != key || i->external != external || !fresh(*i, at)) continue;
            if (phase > i->phases.size()) return {};
            auto const elapsed = phase ? i->phases[phase - 1] : i->elapsed;
            if (!result.samples) { result.low_ns = elapsed; result.oldest_utc = i->at; }
            result.low_ns = std::min(result.low_ns, elapsed); result.high_ns = std::max(result.high_ns, elapsed);
            result.oldest_utc = std::min(result.oldest_utc, i->at); ++result.samples;
        }
        return result.samples >= (kind_ == Kind::Fill ? 1u : 2u) ? result : ConstructionPrior{};
    }
    // Retain startup and the last drain when scaling a completed sequence of
    // otherwise comparable publications. A single publication gives no rate
    // for subsequent groups. Return nanoseconds per member at the new size.
    ConstructionPrior fill_evidence(std::string const& key, bool external, uint64_t members, uint64_t at = now()) const
    {
        ConstructionPrior result;
        if (!members || kind_ != Kind::Fill) return result;
        for (auto i = prior_.rbegin(); i != prior_.rend() && result.samples < 8; ++i)
        {
            if (i->key != key || i->external != external || !fresh(*i, at) || i->phases.size() != 4) continue;
            auto const first_count = i->phases[0], last_count = i->phases[1];
            auto const first_ns = i->phases[2], last_ns = i->phases[3];
            if (last_count <= first_count || members < first_count || last_ns <= first_ns) continue;
            auto const elapsed = first_ns + static_cast<long double>(last_ns - first_ns) *
                (members - first_count) / (last_count - first_count);
            auto const per_member = elapsed / members;
            if (per_member >= static_cast<long double>(UINT64_MAX)) continue;
            auto const ns = uint64_t(std::max(1.L, per_member));
            if (!result.samples) { result.low_ns = ns; result.oldest_utc = i->at; }
            result.low_ns = std::min(result.low_ns, ns); result.high_ns = std::max(result.high_ns, ns);
            result.oldest_utc = std::min(result.oldest_utc, i->at); ++result.samples;
        }
        return result;
    }
    void completed(std::string const& key, bool external, uint64_t elapsed, uint64_t at = now()) noexcept
    { completed(key, external, elapsed, {}, at); }
    void completed(std::string const& key, bool external, uint64_t elapsed, std::span<uint64_t const> phases, uint64_t at = now()) noexcept
    {
        try { if (available() && valid(key) && elapsed && phases.size() <= max_phases)
            append(recent_, {key, external, at, elapsed, {phases.begin(), phases.end()}}); } catch (...) {}
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
            for (auto const& s : prior_)
            {
                out << s.key << ' ' << s.external << ' ' << s.at << ' ' << s.elapsed;
                if (kind_ == Kind::PoolPublication || kind_ == Kind::Fill)
                {
                    out << ' ' << s.phases.size();
                    for (auto const phase : s.phases) out << ' ' << phase;
                }
                out << '\n';
            }
            auto const body = out.str();
            stream_state::atomic_save(directory_ / filename(), stream_state::digest(body) + '\n' + body);
        }
        catch (...) {}
    }
};
// A terminal RAM group can overlap other, already-committed physical writers.
// Each peer's lifetime starts at its submitted whole-group append. A repeated
// observation prices only phases already covered by both measured formats.
struct TerminalPublication
{
    static constexpr size_t max_peers = 4;
    struct Peer { size_t owner; uint64_t started, durable = 0; };
    std::string key;
    size_t owner = 0;
    uint64_t started = 0, durable = 0;
    std::vector<Peer> peers;
    struct Range { uint64_t low = 0, high = 0; unsigned samples = 0; };
    std::optional<bool> choice(ConstructionCache const& cache, std::array<Range, 2>* observed = nullptr) const
    {
        auto range = [&](bool encoded) -> std::optional<Range>
        {
            auto const subject = cache.evidence(key, encoded);
            if (!ConstructionCache::remaining(subject).count()) return {};
            Range finish{subject.low_ns, subject.high_ns, subject.samples};
            for (size_t i = 0; i < peers.size(); ++i)
            {
                auto const age = cache.phase_evidence(key, encoded, 2 * i + 1);
                auto const life = cache.phase_evidence(key, encoded, 2 * i + 2);
                auto const elapsed = started - peers[i].started;
                if (!ConstructionCache::remaining(age).count() || !ConstructionCache::remaining(life).count() ||
                    elapsed < age.low_ns || elapsed > age.high_ns || elapsed >= life.high_ns) return {};
                finish.samples = std::min({finish.samples, age.samples, life.samples});
                finish.low = std::max(finish.low, life.low_ns - std::min(life.low_ns, elapsed));
                finish.high = std::max(finish.high, life.high_ns - elapsed);
            }
            return finish;
        };
        auto const packed = range(false), fresh = range(true);
        if (packed && fresh)
        {
            if (observed) *observed = {*packed, *fresh};
            auto const margin = 1 - construction_margin(std::min(packed->samples, fresh->samples));
            if (fresh->high < packed->low * margin) return true;
            if (packed->high < fresh->low * margin) return false;
        }
        return {};
    }
    bool publish(size_t writer, uint64_t now)
    {
        if (writer == owner) durable = now;
        else for (auto& peer : peers) if (peer.owner == writer) peer.durable = now;
        return durable > started && std::all_of(peers.begin(), peers.end(), [&](auto const& peer)
            { return peer.durable > started && peer.started < started; });
    }
    void completed(ConstructionCache& cache, bool encoded) const
    {
        std::array<uint64_t, 2 * max_peers> phases{};
        if (peers.size() > max_peers) return;
        for (size_t i = 0; i < peers.size(); ++i)
        {
            phases[2 * i] = started - peers[i].started;
            phases[2 * i + 1] = peers[i].durable - peers[i].started;
        }
        cache.completed(key, encoded, durable - started, std::span(phases).first(2 * peers.size()));
    }
};
} // namespace group_grow
