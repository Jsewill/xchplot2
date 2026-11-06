// Durable logical runs inside one destination. Like GroupCheckpoint.hpp, a
// checkpoint publishes output and consumed input positions together. Freed
// blocks are not reused until that checkpoint has reached stable storage.
#pragma once
#include "host/GroupStorage.hpp"

namespace group_grow
{
class MergeState
{
public:
    static constexpr uint32_t free_block = UINT32_MAX, end_block = UINT32_MAX - 1,
                              reserved_block = UINT32_MAX - 2;
    static constexpr uint8_t construction_version = 0xf3, bootstrap_version = 0xf4;
    static constexpr size_t max_runs = 32;
    enum class Phase : uint8_t { stage, merge, pack, layout };
    struct Run
    {
        uint64_t bytes = 0, consumed = 0, released = 0;
        uint32_t head = end_block, tail = end_block;
        uint16_t first = 0, count = 0;
        bool ordinary = false;
    };
    struct Record
    {
        std::array<char, 16> magic{};
        uint64_t sequence = 0, alignment = 0, block_bytes = 0, slot_bytes = 0;
        uint64_t anchor = 0, storage_bytes = 0, original_bytes = 0;
        uint64_t live_bytes = 0, growth_bound = 0, chunk_bound = 0, codec_required = 0, transaction_bytes = 0;
        uint64_t chunk = 0, final_bytes = 0, layout_done = 0;
        uint32_t map_count = 0, block_limit = 0, merge_first = 0, layout_blocks = 0;
        uint16_t members = 0, run_count = 0;
        uint8_t merge_inputs = 2, merge_count = 0;
        Phase phase = Phase::stage;
        bool finishing = false;
        Header original{}, target{};
        std::array<Run, max_runs> runs{};
        Run summary{}, output{};
        std::array<uint8_t, 32> summary_digest{};
        std::array<uint8_t, 255> memo{};
        std::array<char, 64> build{};
    };
    struct Locator
    {
        std::array<char, 16> magic{};
        uint64_t slots = 0, slot_bytes = 0;
        std::array<uint8_t, 32> digest{};
    };
    static_assert(sizeof(Locator) == 64);

    Config config;
    Identity identity;
    Record state{};
    std::vector<uint32_t> map;
    mutable Stats stats{};
    bool active = false, tail_records = false;

    explicit MergeState(std::string path, Config cfg, bool resume, std::optional<Identity> initial);
    ~MergeState();
    MergeState(MergeState const&) = delete;
    MergeState& operator=(MergeState const&) = delete;
    static bool pending(std::string const& path);

    static uint64_t align(uint64_t bytes, uint64_t unit)
    { require(unit, "zero merge alignment"); return grow_add(bytes, bytes % unit ? unit - bytes % unit : 0); }
    static uint64_t multiply(uint64_t a, uint64_t b)
    { require(!b || a <= uint64_t(INT64_MAX) / b, "merge size overflow"); return a * b; }
    uint64_t size() const;
    void check_path() const;
    void read(uint64_t offset, std::span<uint8_t> bytes) const;
    void write(uint64_t offset, std::span<uint8_t const> bytes);
    template<class T> T value(uint64_t offset) const
    { T result{}; read(offset, {reinterpret_cast<uint8_t*>(&result), sizeof(result)}); return result; }
    template<class T> void value(uint64_t offset, T const& data)
    { write(offset, {reinterpret_cast<uint8_t const*>(&data), sizeof(data)}); }
    void sync();
    void trim(uint64_t bytes);
    void evict(uint64_t offset, uint64_t bytes) const;
    template<class NextBlock> void evict_blocks(uint64_t offset, uint64_t bytes, NextBlock next) const
    {
        if (!config.release_cache || !bytes) return;
        uint64_t begin = 0, length = 0;
        while (bytes)
        {
            uint32_t const block = next();
            require(block < state.block_limit && offset < state.block_bytes, "invalid merge cache range");
            uint64_t const position = uint64_t(block) * state.block_bytes + offset;
            uint64_t const count = std::min(bytes, state.block_bytes - offset);
            if (length && position != begin + length) { evict(begin, length); length = 0; }
            if (!length) begin = position;
            length += count; bytes -= count; offset = 0;
        }
        evict(begin, length);
    }
    void confirm();
    void restore_extent();
    Record layout(uint64_t block_bytes) const;
    uint64_t required_bytes(Record const& record, uint64_t payload) const;
    void prepare(Record record);
    void initialize(Record record, uint64_t bytes);
    void reserve(uint64_t bytes);
    void checkpoint(std::string const& label);
    void relocate_checkpoint();
    void publish();
    void validate() const;

    uint32_t allocate(uint32_t preferred = free_block, uint32_t fallback_begin = 0);
    std::vector<uint32_t> blocks(Run const& run) const;
    bool releasing(Run const& run, uint64_t position, uint32_t block) const;
    void release(Run& run, uint64_t position);
    void discard(Run& run) { release(run, run.bytes); run = {}; }
    uint64_t footer_bytes() const { return grow_add(multiply(state.slot_bytes, 2), multiply(alignment_, 2)); }
    uint64_t footer_slots() const { return state.storage_bytes - footer_bytes(); }
    uint64_t alignment() const { return alignment_; }
    std::string const& path() const { return path_; }

    class Reader
    {
    public:
        Reader(MergeState& owner, Run const& run, uint64_t buffer_bytes, uint64_t position = UINT64_MAX)
            : owner_(owner), run_(run), blocks_(owner.blocks(run)), buffer_(size_t(buffer_bytes)),
              position_(position == UINT64_MAX ? run.consumed : position), evicted_(position_)
        {
            require(buffer_bytes && buffer_bytes <= SIZE_MAX && position_ >= run.released && position_ <= run.bytes,
                    "invalid logical run read position");
        }
        uint64_t position() const { return position_; }
        uint64_t remaining() const { return run_.bytes - position_; }
        void read(std::span<uint8_t> bytes)
        {
            require(bytes.size() <= remaining(), "truncated logical run");
            while (!bytes.empty())
            {
                auto cached = peek();
                size_t const count = std::min(cached.size(), bytes.size());
                std::memcpy(bytes.data(), cached.data(), count); position_ += count; bytes = bytes.subspan(count);
            }
        }
        template<class T> T value()
        { T result{}; read({reinterpret_cast<uint8_t*>(&result), sizeof(result)}); return result; }
        void skip(uint64_t bytes)
        { require(bytes <= remaining(), "truncated logical run"); position_ += bytes; }
        void seek(uint64_t position)
        { require(position >= run_.consumed && position <= run_.bytes, "invalid logical run seek"); position_ = position; }
        std::span<uint8_t const> peek()
        {
            require(remaining(), "unexpected end of logical run");
            if (position_ < cached_begin_ || position_ - cached_begin_ >= cached_size_)
            {
                cached_begin_ = position_;
                cached_size_ = size_t(std::min<uint64_t>(buffer_.size(), remaining()));
                copy(position_, std::span(buffer_).first(cached_size_));
            }
            return std::span<uint8_t const>(buffer_).subspan(size_t(position_ - cached_begin_),
                cached_size_ - size_t(position_ - cached_begin_));
        }
        void copy(uint64_t position, std::span<uint8_t> bytes) const
        {
            require(position >= run_.released && position <= run_.bytes && bytes.size() <= run_.bytes - position,
                    "invalid logical run copy bounds");
            while (!bytes.empty())
            {
                uint64_t const logical = position - run_.released, offset = position % owner_.state.block_bytes;
                size_t const index = size_t(logical / owner_.state.block_bytes);
                require(index < blocks_.size(), "missing logical run block");
                size_t const count = size_t(std::min<uint64_t>(bytes.size(), owner_.state.block_bytes - offset));
                owner_.read(uint64_t(blocks_[index]) * owner_.state.block_bytes + offset, bytes.first(count));
                position += count; bytes = bytes.subspan(count);
            }
        }
        void evict(uint64_t position)
        {
            require(position >= evicted_ && position <= position_, "invalid consumed merge cache frontier");
            if (position == evicted_) return;
            size_t index = size_t((evicted_ - run_.released) / owner_.state.block_bytes);
            // The reader owns the original physical map. release() may have
            // retired these links before the checkpoint became durable.
            // Its terminal page belongs only to this run, even when the last
            // block is partial; include it once the whole run is consumed.
            uint64_t const end = position == run_.bytes ? align(position, owner_.alignment()) : position;
            owner_.evict_blocks(evicted_ % owner_.state.block_bytes, end - evicted_, [&]
            {
                require(index < blocks_.size(), "missing consumed merge cache block");
                return blocks_[index++];
            });
            evicted_ = position;
        }
        void evict_read()
        {
            // Read-only validation/planning has no replacement checkpoint.
            // Keep its cache bounded without advising once for every chunk.
            if (position_ - evicted_ >= buffer_.size() || !remaining()) evict(position_);
        }
    private:
        MergeState& owner_;
        Run run_;
        std::vector<uint32_t> blocks_;
        std::vector<uint8_t> buffer_;
        uint64_t position_, evicted_, cached_begin_ = UINT64_MAX;
        size_t cached_size_ = 0;
    };
    class Writer
    {
    public:
        Writer(MergeState& owner, Run& run, uint64_t buffer_bytes, bool final_positions = false)
            : owner_(owner), run_(run), buffer_(size_t(buffer_bytes)), final_positions_(final_positions)
        { require(buffer_bytes && buffer_bytes <= SIZE_MAX && !run.consumed, "invalid logical run writer"); }
        uint64_t position() const { return run_.bytes + used_; }
        void write(std::span<uint8_t const> bytes)
        {
            while (!bytes.empty())
            {
                size_t const count = std::min(bytes.size(), buffer_.size() - used_);
                std::memcpy(buffer_.data() + used_, bytes.data(), count); used_ += count; bytes = bytes.subspan(count);
                if (used_ == buffer_.size()) flush();
            }
        }
        template<class T> void value(T const& data)
        { write({reinterpret_cast<uint8_t const*>(&data), sizeof(data)}); }
        void flush()
        {
            auto bytes = std::span<uint8_t const>(buffer_).first(used_);
            while (!bytes.empty())
            {
                uint64_t const offset = run_.bytes % owner_.state.block_bytes;
                if (!offset)
                {
                    uint32_t const logical = uint32_t(run_.bytes / owner_.state.block_bytes);
                    // Keep vacant future canonical positions available. The
                    // admission bound covers the final stream, including its
                    // index; tail preference is optional under tight capacity.
                    uint32_t const tail = final_positions_ ? uint32_t(std::min<uint64_t>(owner_.state.block_limit,
                        align(grow_add(owner_.state.live_bytes, owner_.state.growth_bound), owner_.state.block_bytes) /
                            owner_.state.block_bytes)) : 0;
                    uint32_t const block = owner_.allocate(final_positions_ ? logical : free_block, tail);
                    if (final_positions_ && block == logical) ++owner_.stats.placed_blocks;
                    if (run_.tail != end_block) owner_.map[run_.tail] = block;
                    else run_.head = block;
                    run_.tail = block;
                }
                size_t const count = size_t(std::min<uint64_t>(bytes.size(), owner_.state.block_bytes - offset));
                owner_.write(uint64_t(run_.tail) * owner_.state.block_bytes + offset, bytes.first(count));
                run_.bytes += count; bytes = bytes.subspan(count);
            }
            used_ = 0;
        }
        void evict(bool completed = false)
        {
            uint64_t const end = completed ? align(run_.bytes, owner_.alignment()) : run_.bytes;
            if (!owner_.config.release_cache || evicted_ == end) return;
            require(evicted_ <= run_.bytes, "merge output cache frontier moved backwards");
            uint32_t next = evicted_ ? evicted_block_ : run_.head;
            if (evicted_ && evicted_ % owner_.state.block_bytes == 0)
            {
                require(next < owner_.state.block_limit, "missing merged output cache frontier");
                next = owner_.map[next];
            }
            owner_.evict_blocks(evicted_ % owner_.state.block_bytes, end - evicted_, [&]
            {
                require(next < owner_.state.block_limit, "missing merged output cache block");
                evicted_block_ = next; next = owner_.map[next]; return evicted_block_;
            });
            evicted_ = run_.bytes;
        }
    private:
        MergeState& owner_;
        Run& run_;
        std::vector<uint8_t> buffer_;
        size_t used_ = 0;
        bool final_positions_ = false;
        uint64_t evicted_ = 0;
        uint32_t evicted_block_ = end_block;
    };

private:
    friend class MergeEngine;
    struct ReadOnly {};
    MergeState(std::string path, Config config, ReadOnly, ConstructionInspectionLimits const& limits, std::optional<Identity> const& initial, ConstructionStamp& target,
               std::array<uint8_t, 32>& authority);
    void discover(bool resume, std::optional<Identity> initial,
                  std::array<uint8_t, 32>* authority = nullptr);
    std::string path_;
    ConstructionInspectionLimits const* inspection_limits_ = nullptr;
    int fd_ = -1;
    uint64_t device_ = 0, inode_ = 0, alignment_ = 4096, expected_bytes_ = 0;
    uint32_t allocation_hint_ = 0, placement_hint_ = 0, placement_begin_ = 0, backward_hint_ = free_block;
    bool created_ = false, confirmed_ = false, cow_ = false;
    std::optional<std::pair<Record, std::vector<uint32_t>>> load(uint64_t slots, uint64_t slot_bytes,
        std::array<uint8_t, 32>* authority = nullptr) const;
    std::optional<Locator> locator(uint64_t offset) const;
    void locators(uint64_t first, uint64_t slots);
    void checkpoint_bytes();
    void load_ordinary(Header header);
};

inline uint64_t MergeState::size() const
{
    file_io::Stat info{};
    require(!file_io::inspect_fd(fd_, &info) && info.st_size >= 0, "cannot inspect merged destination");
    return uint64_t(info.st_size);
}
inline void MergeState::check_path() const
{
    file_io::Stat info{};
    require(!file_io::inspect_link(path_.c_str(), &info) && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
        uint64_t(info.st_dev) == device_ && uint64_t(info.st_ino) == inode_ && size() == expected_bytes_,
        "merged destination path, links or size changed while open");
}
inline void MergeState::read(uint64_t offset, std::span<uint8_t> bytes) const
{
    require(grow_add(offset, bytes.size()) <= size(), "read outside merged destination");
    while (!bytes.empty())
    {
        if (inspection_limits_) inspection_limits_->check();
        auto const begin = std::chrono::steady_clock::now();
        auto const n = file_io::read_at(fd_, bytes.data(), std::min(bytes.size(), size_t(file_io::max_transfer)), file_io::Offset(offset));
        stats.read_ns += elapsed_ns(begin); ++stats.read_calls;
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "cannot read merged destination");
        stats.read_bytes += uint64_t(n); offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
    }
}
inline void MergeState::write(uint64_t offset, std::span<uint8_t const> bytes)
{
    require(grow_add(offset, bytes.size()) <= expected_bytes_, "write outside admitted merge workspace");
    while (!bytes.empty())
    {
        auto const begin = std::chrono::steady_clock::now();
        auto const n = file_io::write_at(fd_, bytes.data(), std::min(bytes.size(), size_t(file_io::max_transfer)), file_io::Offset(offset));
        stats.write_ns += elapsed_ns(begin); ++stats.write_calls;
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "cannot write merged destination; retain this file for resume");
        stats.written_bytes += uint64_t(n); offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
    }
}
inline void MergeState::sync()
{
    auto const begin = std::chrono::steady_clock::now(); int status;
    do { status = file_io::flush(fd_); } while (status && errno == EINTR);
    stats.sync_ns += elapsed_ns(begin); ++stats.syncs;
    require(!status, "cannot sync merged destination; retain this file for resume");
}
inline void MergeState::trim(uint64_t bytes)
{
    require(!file_io::resize_file(fd_, file_io::Offset(bytes)), "cannot resize merged destination"); expected_bytes_ = bytes;
}
inline void MergeState::evict(uint64_t offset, uint64_t bytes) const
{
#ifdef __linux__
    if (config.release_cache && bytes)
    {
        // Revisit the boundary page once it has been fully consumed. Advice
        // ignores partial pages; passing disjoint unaligned ranges forever
        // would otherwise leave one cached page at every checkpoint.
        uint64_t const requested = offset + bytes;
        uint64_t const end = requested == expected_bytes_ ? requested : requested / alignment_ * alignment_;
        offset = offset / alignment_ * alignment_;
        if (end > offset) (void)::posix_fadvise(fd_, file_io::Offset(offset), file_io::Offset(end - offset), POSIX_FADV_DONTNEED);
    }
#else
    (void)offset; (void)bytes;
#endif
}
inline std::optional<MergeState::Locator> MergeState::locator(uint64_t offset) const
{
    if (offset > size() || size() - offset < sizeof(Locator)) return {};
    auto result = value<Locator>(offset);
    if (std::memcmp(result.magic.data(), "XCHPLOT2MERGELOC", 16)) return {};
    auto const digest = grow_digest({reinterpret_cast<uint8_t const*>(&result), offsetof(Locator, digest)});
    if (digest != result.digest) return {};
    return result;
}
inline void MergeState::locators(uint64_t first, uint64_t slots)
{
    Locator location{}; std::memcpy(location.magic.data(), "XCHPLOT2MERGELOC", 16);
    location.slots = slots; location.slot_bytes = state.slot_bytes;
    location.digest = grow_digest({reinterpret_cast<uint8_t const*>(&location), offsetof(Locator, digest)});
    value(first, location); grow_failpoint("merge-locator-partial"); value(first + alignment_, location);
}
inline std::optional<std::pair<MergeState::Record, std::vector<uint32_t>>>
MergeState::load(uint64_t slots, uint64_t slot_bytes, std::array<uint8_t, 32>* authority) const
{
    if (slot_bytes < sizeof(Record) + 32 || slot_bytes % alignment_ || slots % alignment_ ||
        slots > size() || slot_bytes > (size() - slots) / 2) return {};
    require(slot_bytes <= config.memory_limit_bytes / 4 && slot_bytes <= SIZE_MAX,
            "merge checkpoints exceed memory budget");
    if (inspection_limits_) inspection_limits_->check(grow_add(multiply(slot_bytes, 4), 65536));
    std::optional<std::pair<Record, std::vector<uint32_t>>> result;
    std::vector<uint8_t> bytes(size_t(slot_bytes), 0);
    for (unsigned i = 0; i < 2; ++i)
    {
        read(slots + i * slot_bytes, bytes);
        auto const digest = grow_digest(std::span(bytes).first(bytes.size() - 32), config.cpu_scheduler.get());
        if (!std::equal(digest.begin(), digest.end(), bytes.end() - 32)) continue;
        Record saved{}; std::memcpy(&saved, bytes.data(), sizeof(saved));
        if (std::memcmp(saved.magic.data(), "XCHPLOT2MERGEv1", 16) || saved.slot_bytes != slot_bytes ||
            !saved.sequence || saved.map_count > (slot_bytes - sizeof(saved) - 32) / sizeof(uint32_t)) continue;
        if (result && result->first.sequence >= saved.sequence) continue;
        std::vector<uint32_t> blocks(saved.map_count);
        std::memcpy(blocks.data(), bytes.data() + sizeof(saved), blocks.size() * sizeof(uint32_t));
        result = std::pair{saved, std::move(blocks)};
        if (authority) *authority = digest;
    }
    return result;
}
inline bool MergeState::pending(std::string const& path)
{
    int fd = file_io::open_file(path.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow);
    if (fd < 0) return false;
    struct Close { int fd; ~Close() { file_io::close_file(fd); } } close{fd};
    file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
    if (file_io::inspect_fd(fd, &info) || !file_io::is_regular(info.st_mode) || info.st_size < file_io::Offset(sizeof(Header)) ||
        file_io::space_fd(fd, &allocation) || page <= 0 || info.st_blksize <= 0) return false;
    Header header{};
    if (file_io::read_at(fd, &header, sizeof(header), 0) == sizeof(header) &&
        header.magic == PlotGroupFile::MAGIC &&
        (header.version == construction_version || header.version == bootstrap_version)) return true;
    uint64_t const unit = std::max({uint64_t(4096), uint64_t(page), uint64_t(info.st_blksize),
        uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize)});
    if (uint64_t(info.st_size) < 2 * unit) return false;
    for (unsigned i = 1; i <= 2; ++i)
    {
        Locator saved{};
        if (file_io::read_at(fd, &saved, sizeof(saved), file_io::Offset(uint64_t(info.st_size) - i * unit)) != sizeof(saved) ||
            std::memcmp(saved.magic.data(), "XCHPLOT2MERGELOC", 16)) continue;
        if (grow_digest({reinterpret_cast<uint8_t const*>(&saved), offsetof(Locator, digest)}) == saved.digest) return true;
    }
    return false;
}

inline void MergeState::load_ordinary(Header header)
{
    require(header.magic == PlotGroupFile::MAGIC && header.version == PlotGroupFile::FORMAT_VERSION &&
        header.k >= 18 && header.k <= 28 && !(header.k & 1) && header.group_size &&
        header.strength >= 2 && header.strength <= header.k - 3, "invalid ordinary merge source");
    identity.header = header; identity.memo.resize(header.memo_length); read(sizeof(Header), identity.memo);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
    if (inspection_limits_) inspection_limits_->check(Engine::inspection_memory_bytes(header.k));
    require(header.chunk_index_offset >= sizeof(Header) + header.memo_length && header.chunk_index_offset < size(),
            "invalid ordinary merge source index");
    uint64_t const available = std::min(size() - header.chunk_index_offset, multiply(chunks, 8));
    require(multiply(chunks, 24) <= config.memory_limit_bytes, "ordinary merge index exceeds memory budget");
    std::vector<uint64_t> words(size_t((available + 7) / 8), 0), sizes(size_t(chunks), 0);
    read(header.chunk_index_offset, {reinterpret_cast<uint8_t*>(words.data()), size_t(available)});
    BitReader bits(words, available * 8);
    uint64_t end = sizeof(Header) + header.memo_length;
    uint64_t const expected = gsz_expected_for_g(header.group_size);
    unsigned const rice = unsigned(gsz_rice_k_for_g(header.group_size));
    for (auto& bytes : sizes)
    {
        if (inspection_limits_) inspection_limits_->check();
        bool const negative = bits.read_u64(1);
        uint64_t const remainder = bits.read_u64(rice), quotient = bits.read_unary_64();
        require(quotient + rice + 2 <= 64, "invalid ordinary merge size code");
        uint64_t const magnitude = (quotient << rice) | remainder;
        require(!negative || magnitude < expected, "invalid ordinary merge size delta");
        bytes = negative ? expected - magnitude : grow_add(expected, magnitude);
        check_gsz_size(bytes, header.group_size); end = grow_add(end, bytes);
    }
    require(end == header.chunk_index_offset, "ordinary merge index does not cover payload");
    BitWriter canonical; gsz_encode(sizes, header.group_size, canonical); auto encoded = canonical.asBytes();
    require(encoded.size() <= available && std::equal(encoded.begin(), encoded.end(),
        reinterpret_cast<uint8_t const*>(words.data())), "noncanonical ordinary merge source index");
    identity.bytes = grow_add(header.chunk_index_offset, encoded.size());
    identity.old_members = identity.target_members = header.group_size;
}

inline MergeState::MergeState(std::string path, Config cfg, bool resume, std::optional<Identity> initial)
    : config(cfg), path_(std::move(path))
{
    require(cfg.window_bytes >= sizeof(Header) && cfg.window_bytes <= SIZE_MAX && cfg.codec_bytes &&
        cfg.checkpoint_bytes && cfg.workers && cfg.capacity_bytes <= uint64_t(INT64_MAX), "invalid merge budgets");
    if (initial && !resume)
    { fd_ = file_io::open_file(path_.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600); created_ = fd_ >= 0; }
    else fd_ = file_io::open_file(path_.c_str(), file_io::read_write | file_io::close_on_exec | file_io::no_follow);
    if (fd_ < 0 && errno == ENOENT && initial)
    { fd_ = file_io::open_file(path_.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600); created_ = fd_ >= 0; }
    require(fd_ >= 0, "cannot open merged destination: " + path_);
    try
    {
        require(!file_io::lock_file(fd_, file_io::lock_exclusive | file_io::lock_nonblocking), "merged destination is already in use");
        file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
        require(!file_io::inspect_fd(fd_, &info) && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
            !file_io::space_fd(fd_, &allocation) && allocation.f_bsize && info.st_blksize > 0 && page > 0,
            "cannot inspect merged destination allocation");
        device_ = info.st_dev; inode_ = info.st_ino;
        alignment_ = std::max({uint64_t(4096), uint64_t(page), uint64_t(info.st_blksize),
            uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize)});
        expected_bytes_ = size();
#ifdef __linux__
        struct statfs filesystem{}; require(!::fstatfs(fd_, &filesystem), "cannot inspect merge filesystem");
        cow_ = unsigned(filesystem.f_type) == 0x9123683eu;
#endif
        discover(resume, std::move(initial));
    }
    catch (...)
    {
        if (created_ && !size()) file_io::unlink_file(path_.c_str());
        file_io::close_file(fd_); fd_ = -1; throw;
    }
}
inline MergeState::MergeState(std::string path, Config cfg, ReadOnly, ConstructionInspectionLimits const& limits, std::optional<Identity> const& initial, ConstructionStamp& target,
                             std::array<uint8_t, 32>& authority)
    : config(cfg), path_(std::move(path)), inspection_limits_(&limits)
{
    limits.check(65536);
    require(cfg.window_bytes >= sizeof(Header) && cfg.window_bytes <= SIZE_MAX && cfg.codec_bytes &&
        cfg.checkpoint_bytes && cfg.workers && cfg.capacity_bytes <= uint64_t(INT64_MAX), "invalid merge budgets");
    fd_ = file_io::open_file(path_.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow);
    require(fd_ >= 0, "cannot inspect merged destination: " + path_);
    try
    {
        target = construction_stamp(fd_, path_); device_ = target.device; inode_ = target.inode;
        file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
        require(!file_io::inspect_fd(fd_, &info) && !file_io::space_fd(fd_, &allocation) &&
            allocation.f_bsize && info.st_blksize > 0 && page > 0, "cannot inspect merged destination allocation");
        alignment_ = std::max({uint64_t(4096), uint64_t(page), uint64_t(info.st_blksize),
            uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize)});
        expected_bytes_ = target.bytes; discover(true, initial, &authority);
        require(size() <= cfg.capacity_bytes && identity.bytes <= cfg.capacity_bytes &&
            (!active || state.storage_bytes <= cfg.capacity_bytes), "saved merge workspace exceeds capacity");
        require(construction_stamp(fd_, path_) == target, "merged destination changed during inspection");
    }
    catch (...) { file_io::close_file(fd_); fd_ = -1; throw; }
}
inline void MergeState::discover(bool resume, std::optional<Identity> initial,
                                 std::array<uint8_t, 32>* authority)
{
    std::optional<std::pair<Record, std::vector<uint32_t>>> saved;
    auto consider = [&](std::optional<Locator> location)
    {
        if (!location) return;
        std::array<uint8_t, 32> digest{}; auto candidate = load(location->slots, location->slot_bytes, &digest);
        if (candidate && (!saved || candidate->first.sequence > saved->first.sequence))
        { saved = std::move(candidate); if (authority) *authority = digest; }
    };
    if (size() >= 2 * alignment_)
    { consider(locator(size() - alignment_)); consider(locator(size() - 2 * alignment_)); }
    Header header{};
    if (size() >= sizeof(Header))
    {
        require(resume, "existing merged or ordinary groups require --resume");
        header = value<Header>(0);
        if (header.magic == PlotGroupFile::MAGIC &&
            (header.version == construction_version || header.version == bootstrap_version) &&
            header.chunk_index_offset <= size() && size() - header.chunk_index_offset >= 2 * alignment_)
        {
            consider(locator(header.chunk_index_offset));
            consider(locator(header.chunk_index_offset + alignment_));
        }
    }
    if (saved)
    {
        require(resume, "in-progress merged group is not farmable; use --resume");
        state = saved->first; map = std::move(saved->second); active = true;
        tail_records = state.phase == Phase::layout;
        std::array<char, 64> build{}; auto const name = std::string_view(GROW_IDENTITY);
        std::copy_n(name.begin(), std::min(name.size(), build.size()), build.begin());
        require(state.build == build, "merge checkpoint requires the same codec/build identity");
        validate();
        identity = {state.original, {state.memo.begin(), state.memo.begin() + state.original.memo_length},
            state.original_bytes, true, state.original.group_size, state.members, state.storage_bytes};
    }
    else if (size() && header.magic == PlotGroupFile::MAGIC && header.version == PlotGroupFile::FORMAT_VERSION)
        load_ordinary(header);
    else
    {
        // A new-file bootstrap has no committed member until its first
        // valid checkpoint. Its private header is enough to resume it.
        if (size())
        {
            require(header.magic == PlotGroupFile::MAGIC && header.version == bootstrap_version && !header.group_size,
                    "damaged merged destination requires a valid checkpoint");
            Identity bootstrap; bootstrap.header = header; bootstrap.header.version = PlotGroupFile::FORMAT_VERSION;
            bootstrap.memo.resize(header.memo_length); read(sizeof(Header), bootstrap.memo);
            bootstrap.header.chunk_index_offset = sizeof(Header) + bootstrap.memo.size(); initial = bootstrap;
        }
        require(initial.has_value(), "empty merged destination needs identity"); identity = *initial;
        require(!identity.header.group_size && identity.memo.size() == identity.header.memo_length && identity.memo.size() <= 255,
                "invalid new merge identity");
        identity.header.version = PlotGroupFile::FORMAT_VERSION;
        identity.header.chunk_index_offset = sizeof(Header) + identity.memo.size(); identity.bytes = 0;
    }
    require(identity.header.magic == PlotGroupFile::MAGIC && identity.header.k >= 18 && identity.header.k <= 28 &&
        !(identity.header.k & 1) && identity.header.strength >= 2 && identity.header.strength <= identity.header.k - 3,
        "invalid merge identity");
}

inline MergeState::~MergeState()
{
    if (fd_ < 0) return;
    if (created_ && !size())
    {
        file_io::Stat named{};
        if (!file_io::inspect_link(path_.c_str(), &named) && uint64_t(named.st_dev) == device_ && uint64_t(named.st_ino) == inode_)
            file_io::unlink_file(path_.c_str());
    }
    if (!active && stats.syncs) evict(0, size());
    file_io::close_file(fd_);
}
inline void MergeState::confirm()
{
    check_path(); require(identity.bytes <= config.capacity_bytes && (!active || state.storage_bytes <= config.capacity_bytes),
        "saved merge workspace exceeds capacity"); confirmed_ = true;
}
inline void MergeState::restore_extent()
{
    require(confirmed_, "merge mutation requires confirmed identity"); check_path();
    uint64_t const committed = active ? state.storage_bytes : identity.bytes;
    if (size() != committed) { trim(committed); sync(); }
    if (active && value<Header>(0).version == bootstrap_version)
    {
        // A crash after the first checkpoint can leave the bootstrap marker.
        // Retire it before any further durable work, so later damage to both
        // checkpoint slots cannot be mistaken for a never-started file.
        Header header = state.original; header.version = construction_version; header.chunk_index_offset = state.anchor;
        value(0, header); sync();
    }
}
inline MergeState::Record MergeState::layout(uint64_t block_bytes) const
{
    Record result{}; std::memcpy(result.magic.data(), "XCHPLOT2MERGEv1", 16);
    result.alignment = alignment_; result.block_bytes = align(block_bytes, alignment_);
    uint64_t const count = config.capacity_bytes / result.block_bytes;
    require(count && count < reserved_block, "merge block map exceeds addressable capacity");
    result.map_count = uint32_t(count);
    result.slot_bytes = align(grow_add(sizeof(Record) + 32, multiply(count, sizeof(uint32_t))), alignment_);
    result.anchor = align(std::max<uint64_t>(identity.bytes, sizeof(Header) + identity.memo.size()), result.block_bytes);
    result.original = result.target = identity.header; result.original_bytes = identity.bytes;
    result.transaction_bytes = config.checkpoint_bytes;
    result.merge_inputs = uint8_t(config.merge_inputs);
    result.members = identity.header.group_size;
    std::copy(identity.memo.begin(), identity.memo.end(), result.memo.begin());
    auto const build = std::string_view(GROW_IDENTITY);
    std::copy_n(build.begin(), std::min(build.size(), result.build.size()), result.build.begin());
    return result;
}
inline uint64_t MergeState::required_bytes(Record const& record, uint64_t payload) const
{
    uint64_t const metadata = align(grow_add(multiply(record.slot_bytes, 2), multiply(alignment_, 2)), record.block_bytes);
    uint64_t const blocks = align(grow_add(payload, grow_add(metadata, record.block_bytes)), record.block_bytes);
    uint64_t const arena = std::max(blocks, grow_add(record.anchor, metadata));
    return grow_add(arena, grow_add(multiply(record.slot_bytes, 2), multiply(alignment_, 2)));
}
inline void MergeState::prepare(Record record)
{
    require(!active, "cannot replace an active merge layout"); state = record;
    map.assign(state.map_count, free_block);
    map[0] = reserved_block;
    uint64_t const end = align(state.anchor + 2 * alignment_ + 2 * state.slot_bytes, state.block_bytes);
    require(end / state.block_bytes <= map.size(), "merge metadata exceeds capacity");
    for (uint64_t block = state.anchor / state.block_bytes; block < end / state.block_bytes; ++block)
        map[size_t(block)] = reserved_block;
    if (identity.bytes)
    {
        uint32_t const count = uint32_t(align(identity.bytes, state.block_bytes) / state.block_bytes);
        for (uint32_t i = 0; i < count; ++i) map[i] = i + 1 < count ? i + 1 : end_block;
        state.runs[0] = {identity.bytes, sizeof(Header) + identity.memo.size(), 0,
            0, count - 1, 0, identity.header.group_size, true}; state.run_count = 1;
    }
    // Read-only admission may walk the original source before reserve().
    state.block_limit = state.map_count;
}
inline void MergeState::initialize(Record record, uint64_t bytes)
{
    require(!active && confirmed_, "invalid merge initialization"); restore_extent(); prepare(record);
    reserve(bytes);
    if (!identity.bytes)
    {
        Header header = identity.header; header.version = bootstrap_version; header.chunk_index_offset = state.anchor;
        value(0, header); write(sizeof(Header), identity.memo); sync();
        auto parent = file_io::native_path(path_).parent_path(); if (parent.empty()) parent = ".";
        int directory = file_io::open_file(parent.c_str(), file_io::read_only | file_io::close_on_exec | file_io::directory);
        require(directory >= 0, "cannot open merge parent directory"); int status = file_io::flush(directory); file_io::close_file(directory);
        require(!status, "cannot sync merge parent directory"); grow_failpoint("merge-bootstrap");
    }
    locators(state.anchor, state.anchor + 2 * alignment_);
    locators(state.storage_bytes - 2 * alignment_, state.anchor + 2 * alignment_);
    sync(); checkpoint("merge-initialized");
    Header header = identity.header; header.version = construction_version; header.chunk_index_offset = state.anchor;
    value(0, header); sync(); active = true;
    identity.pending = true; identity.pending_peak_bytes = state.storage_bytes;
}
inline void MergeState::reserve(uint64_t bytes)
{
    require(confirmed_ && bytes <= config.capacity_bytes && bytes >= footer_bytes(), "merge workspace exceeds capacity");
    check_path();
#ifdef __linux__
    if (size())
    {
        alignas(struct fiemap) std::array<uint8_t, sizeof(struct fiemap) + sizeof(struct fiemap_extent) * 64> data{};
        for (uint64_t position = 0; position < size();)
        {
            data.fill(0); auto* query = reinterpret_cast<struct fiemap*>(data.data());
            query->fm_start = position; query->fm_length = size() - position;
            query->fm_flags = FIEMAP_FLAG_SYNC; query->fm_extent_count = 64;
            if (::ioctl(fd_, FS_IOC_FIEMAP, query))
            {
                require(!cow_ && (errno == EOPNOTSUPP || errno == ENOTTY || errno == ENOSYS), "cannot inspect merge target sharing"); break;
            }
            if (!query->fm_mapped_extents) break;
            auto* extents = reinterpret_cast<struct fiemap_extent*>(data.data() + sizeof(struct fiemap));
            for (uint32_t i = 0; i < query->fm_mapped_extents; ++i)
                require(!(extents[i].fe_flags & FIEMAP_EXTENT_SHARED), "shared merge target requires unsharing");
            auto const& last = extents[query->fm_mapped_extents - 1];
            uint64_t const next = grow_add(last.fe_logical, last.fe_length);
            require(next > position, "invalid merge extent map"); position = next;
            if (last.fe_flags & FIEMAP_EXTENT_LAST) break;
        }
    }
    if (cow_)
    {
        file_io::Space space{}; require(!file_io::space_fd(fd_, &space) && space.f_frsize, "cannot inspect merge CoW headroom");
        uint64_t const transaction = std::min(config.checkpoint_bytes, state.transaction_bytes);
        uint64_t const extra = grow_add(std::max(transaction, state.block_bytes),
            grow_add(config.codec_bytes, grow_add(config.window_bytes, footer_bytes())));
        uint64_t const needed = grow_add(bytes > size() ? bytes - size() : 0, extra);
        require(needed / space.f_frsize + (needed % space.f_frsize != 0) <= space.f_bavail,
                "merge physical CoW workspace exceeds available space");
    }
#endif
#if defined(__linux__) || defined(_WIN32)
    int status;
    do { status = file_io::allocate_file(fd_, file_io::keep_size, 0, file_io::Offset(bytes)); } while (status && errno == EINTR);
    require(!status, "merge physical allocation preflight failed without changing committed bytes");
    sync(); grow_failpoint("merge-allocated");
#else
    throw std::runtime_error("same-file merging requires supported native physical allocation");
#endif
    if (bytes > size()) trim(bytes);
    state.storage_bytes = std::max(bytes, size());
    require((state.storage_bytes - footer_bytes()) % state.block_bytes == 0, "unaligned merge workspace");
    state.block_limit = uint32_t((state.storage_bytes - footer_bytes()) / state.block_bytes);
    require(state.block_limit <= map.size(), "merge workspace exceeds block map");
    // During growth the front checkpoint remains authoritative. Keep two
    // discoverable locators at the new end before any old footer is reused.
    if (active)
    { locators(state.storage_bytes - 2 * alignment_, state.anchor + 2 * alignment_); sync(); }
}
inline void MergeState::checkpoint_bytes()
{
    require(state.sequence < UINT64_MAX, "merge checkpoint sequence overflow"); ++state.sequence;
    std::vector<uint8_t> bytes(size_t(state.slot_bytes), 0);
    require(sizeof(state) + map.size() * sizeof(uint32_t) + 32 <= bytes.size(), "merge checkpoint map overflow");
    std::memcpy(bytes.data(), &state, sizeof(state));
    std::memcpy(bytes.data() + sizeof(state), map.data(), map.size() * sizeof(uint32_t));
    auto const digest = grow_digest(std::span(bytes).first(bytes.size() - 32), config.cpu_scheduler.get());
    std::copy(digest.begin(), digest.end(), bytes.end() - 32);
    uint64_t const base = tail_records ? footer_slots() : state.anchor + 2 * alignment_;
    uint64_t const offset = base + state.sequence % 2 * state.slot_bytes;
    size_t const first = bytes.size() / 2;
    write(offset, std::span(bytes).first(first)); grow_failpoint("merge-checkpoint-partial");
    write(offset + first, std::span(bytes).subspan(first)); sync(); ++stats.checkpoints;
}
inline void MergeState::checkpoint(std::string const& label)
{
    check_path(); sync(); grow_failpoint(label + "-flushed");
    checkpoint_bytes(); grow_failpoint(label + "-committed");
    identity.pending = true; identity.target_members = state.members; identity.pending_peak_bytes = state.storage_bytes;
}
inline uint32_t MergeState::allocate(uint32_t preferred, uint32_t fallback_begin)
{
    if (preferred < state.block_limit && map[preferred] == free_block)
    { map[preferred] = end_block; return preferred; }
    if (fallback_begin)
    {
        if (placement_begin_ != fallback_begin) placement_hint_ = placement_begin_ = fallback_begin;
        for (uint32_t i = placement_hint_; i < state.block_limit; ++i)
        {
            placement_hint_ = i + 1;
            if (map[i] == free_block) { map[i] = end_block; return i; }
        }
    }
    if (preferred < state.block_limit)
    {
        // A displaced final block may reuse a position whose turn has
        // already passed. Otherwise take the farthest future vacancy: a
        // low-first fallback can occupy the next final position and repeat
        // that displacement through the whole remaining stream.
        for (uint32_t i = allocation_hint_; i < preferred; ++i)
            if (map[i] == free_block) { map[i] = end_block; allocation_hint_ = i + 1; return i; }
        allocation_hint_ = std::max(allocation_hint_, preferred);
        uint32_t const end = fallback_begin ? std::min(fallback_begin, state.block_limit) : state.block_limit;
        for (uint32_t i = std::min(backward_hint_, end); i > preferred + 1;)
        {
            --i; backward_hint_ = i;
            if (map[i] == free_block) { map[i] = end_block; return i; }
        }
    }
    for (uint32_t i = allocation_hint_; i < state.block_limit; ++i)
        if (map[i] == free_block) { map[i] = end_block; allocation_hint_ = i + 1; return i; }
    throw std::runtime_error("proven merge workspace exhausted; retain destination for resume");
}
inline std::vector<uint32_t> MergeState::blocks(Run const& run) const
{
    require(run.released <= run.consumed && run.consumed <= run.bytes && run.released % state.block_bytes == 0,
            "invalid merge run frontier");
    uint64_t const count = run.consumed == run.bytes ? 0 : align(run.bytes - run.released, state.block_bytes) / state.block_bytes;
    require(count <= state.block_limit, "invalid merge run block count");
    std::vector<uint32_t> result; result.reserve(size_t(count));
    uint32_t next = run.head;
    for (uint64_t i = 0; i < count; ++i)
    {
        if (inspection_limits_) inspection_limits_->check();
        require(next < state.block_limit, "invalid merge run block link"); result.push_back(next); next = map[next];
    }
    require(next == end_block && (result.empty() ? run.head == end_block && run.tail == end_block : result.back() == run.tail),
            "invalid merge run tail");
    return result;
}
inline bool MergeState::releasing(Run const& run, uint64_t position, uint32_t block) const
{
    require(position >= run.consumed && position <= run.bytes, "merge consumption moved backwards");
    if (run.consumed == run.bytes || !block || block >= state.block_limit) return false;
    uint64_t const end = position == run.bytes ? align(run.bytes, state.block_bytes) : position / state.block_bytes * state.block_bytes;
    uint32_t next = run.head;
    for (uint64_t offset = run.released; offset < end; offset += state.block_bytes)
    {
        require(next < state.block_limit, "invalid consumed merge block");
        if (next == block) return true;
        next = map[next];
    }
    return false;
}
inline void MergeState::release(Run& run, uint64_t position)
{
    require(position >= run.consumed && position <= run.bytes, "merge consumption moved backwards");
    run.consumed = position;
    uint64_t const end = position == run.bytes ? align(run.bytes, state.block_bytes) : position / state.block_bytes * state.block_bytes;
    while (run.released < end)
    {
        uint32_t const block = run.head; require(block < state.block_limit, "invalid consumed merge block");
        run.head = map[block];
        map[block] = block ? free_block : reserved_block;
        if (block)
        {
            allocation_hint_ = std::min(allocation_hint_, block);
            if (block >= placement_begin_) placement_hint_ = std::min(placement_hint_, block);
            backward_hint_ = std::max(backward_hint_, block + 1);
            stats.reclaimed_bytes += state.block_bytes;
        }
        run.released += state.block_bytes;
    }
    if (position == run.bytes)
    { require(run.head == end_block, "unconsumed merge run blocks"); run.tail = end_block; run.released = run.bytes / state.block_bytes * state.block_bytes; }
}
inline void MergeState::validate() const
{
    if (inspection_limits_) inspection_limits_->check(grow_add(multiply(map.size(), 32), 65536));
    require(state.alignment == alignment_ && state.block_bytes >= alignment_ && state.block_bytes % alignment_ == 0 &&
        state.block_bytes <= state.storage_bytes && state.transaction_bytes &&
        state.map_count == map.size() && state.map_count && state.block_limit <= state.map_count &&
        state.slot_bytes >= sizeof(Record) + map.size() * sizeof(uint32_t) + 32 && state.slot_bytes % alignment_ == 0 &&
        state.storage_bytes <= size() && state.block_limit <= state.storage_bytes / state.block_bytes &&
        state.storage_bytes == grow_add(multiply(state.block_limit, state.block_bytes), footer_bytes()) &&
        state.anchor % state.block_bytes == 0 && state.anchor >= state.original_bytes &&
        grow_add(state.anchor, grow_add(2 * alignment_, 2 * state.slot_bytes)) <= multiply(state.block_limit, state.block_bytes) &&
        state.run_count <= max_runs && (state.merge_inputs == 2 || state.merge_inputs == 4) &&
        state.members >= state.original.group_size && state.target.group_size == state.members &&
        state.original.magic == PlotGroupFile::MAGIC && state.target.magic == PlotGroupFile::MAGIC &&
        state.original.version == PlotGroupFile::FORMAT_VERSION && state.target.version == PlotGroupFile::FORMAT_VERSION &&
        state.original.k >= 18 && state.original.k <= 28 && !(state.original.k & 1) &&
        state.original.strength >= 2 && state.original.strength <= state.original.k - 3 &&
        state.original.group_id == state.target.group_id && state.original.k == state.target.k &&
        state.original.strength == state.target.strength && state.original.meta_group == state.target.meta_group &&
        state.original.memo_length == state.target.memo_length &&
        (state.original.group_size ? state.original.chunk_index_offset >= sizeof(Header) + state.original.memo_length &&
            state.original.chunk_index_offset < state.original_bytes : !state.original_bytes) &&
        uint8_t(state.phase) <= uint8_t(Phase::layout), "inconsistent merge checkpoint");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(state.original.k);
    require(state.chunk <= chunks, "invalid merge chunk frontier");
    std::vector<uint8_t> used(state.map_count, 0);
    if (state.phase == Phase::layout)
    {
        require(!state.merge_count && state.finishing && state.storage_bytes == size() && state.target.chunk_index_offset >= sizeof(Header) + state.target.memo_length &&
            state.final_bytes > state.target.chunk_index_offset && state.final_bytes <= multiply(state.block_limit, state.block_bytes) &&
            state.layout_blocks == align(state.final_bytes, state.block_bytes) / state.block_bytes &&
            state.layout_blocks <= state.map_count && state.layout_done <= state.layout_blocks,
            "invalid merge layout checkpoint");
        for (uint32_t i = 0; i < state.layout_blocks; ++i)
        {
            if (inspection_limits_) inspection_limits_->check();
            require(map[i] < state.block_limit && !used[map[i]] && (i >= state.layout_done || map[i] == i),
                    "duplicate or misplaced final merge block"); used[map[i]] = 1;
        }
        return;
    }
    require(!state.final_bytes && !state.layout_blocks && !state.layout_done &&
        !state.summary.consumed && !state.output.consumed && !state.output.ordinary,
        "invalid unfinished merge state");
    if (state.phase == Phase::stage)
        require(!state.merge_count && !state.chunk && !state.output.bytes && !state.output.count,
                "staging checkpoint has an active merge output");
    auto visit = [&](Run const& run)
    {
        for (uint32_t block : blocks(run))
        {
            require(!used[block] && (block == 0 || map[block] != reserved_block), "overlapping merge runs"); used[block] = 1;
        }
    };
    uint32_t member = 0;
    for (size_t i = 0; i < state.run_count; ++i)
    {
        auto const& run = state.runs[i];
        require(run.first == member && run.count && run.count <= UINT16_MAX - member &&
            (!run.ordinary || (!i && run.count == state.original.group_size && run.bytes == state.original_bytes)),
            "invalid merge member coverage");
        bool const consuming = state.phase == Phase::pack ||
            (state.phase == Phase::merge && i >= state.merge_first && i - state.merge_first < state.merge_count);
        if (!consuming)
            require(run.consumed == (run.ordinary ? sizeof(Header) + state.original.memo_length : 0) && !run.released,
                    "inactive merge input has consumed blocks");
        member += run.count; visit(run);
    }
    require(member == state.members, "incomplete merge member coverage");
    visit(state.summary); visit(state.output);
    if (state.phase == Phase::merge)
    {
        require(state.merge_count >= 2 && state.merge_count <= state.merge_inputs &&
            state.merge_first < state.run_count && state.merge_count <= state.run_count - state.merge_first,
            "invalid active merge input count");
        uint32_t members = 0;
        for (size_t i = state.merge_first; i < state.merge_first + state.merge_count; ++i) members += state.runs[i].count;
        require(state.output.first == state.runs[state.merge_first].first && state.output.count == members &&
            (state.chunk || !state.output.bytes), "invalid active merge coverage");
    }
    if (state.phase == Phase::pack)
        require(!state.merge_count && state.finishing && state.run_count && !state.summary.bytes &&
            !state.output.first && state.output.count == state.members &&
            state.output.bytes >= sizeof(Header) + state.original.memo_length, "invalid final merge pack");
    uint64_t const first_meta = state.anchor / state.block_bytes;
    uint64_t const end_meta = align(state.anchor + 2 * alignment_ + 2 * state.slot_bytes, state.block_bytes) / state.block_bytes;
    for (size_t i = 0; i < map.size(); ++i)
    {
        if (inspection_limits_) inspection_limits_->check();
        bool const reserved = (i == 0 && !used[i]) || (i >= first_meta && i < end_meta);
        require(reserved ? map[i] == reserved_block : (used[i] || map[i] == free_block),
                "unreachable or unprotected merge block");
    }
}
inline void MergeState::relocate_checkpoint()
{
    require(state.phase == Phase::layout && !tail_records, "invalid merge checkpoint relocation");
    tail_records = true; checkpoint("merge-layout-ready");
    locators(state.storage_bytes - 2 * alignment_, footer_slots()); sync(); grow_failpoint("merge-layout-located");
}
inline void MergeState::publish()
{
    require(confirmed_ && tail_records && state.layout_done == state.layout_blocks, "incomplete merge publication");
    Header marker = state.target; marker.version = 0xf1;
    auto bytes = std::span<uint8_t const>(reinterpret_cast<uint8_t const*>(&marker), sizeof(marker));
    write(0, bytes.first(sizeof(Header) / 2)); grow_failpoint("merge-final-header-partial");
    write(sizeof(Header) / 2, bytes.subspan(sizeof(Header) / 2)); sync(); grow_failpoint("merge-finalizing");
    std::array<uint8_t, 16> retired{};
    write(state.storage_bytes - 2 * alignment_, retired); grow_failpoint("merge-retire-partial");
    write(state.storage_bytes - alignment_, retired); sync(); grow_failpoint("merge-retired");
    trim(state.final_bytes); sync(); grow_failpoint("merge-truncated");
    uint8_t const version = PlotGroupFile::FORMAT_VERSION; write(offsetof(Header, version), {&version, 1}); sync();
    grow_failpoint("merge-published"); active = false; tail_records = false;
    identity = {state.target, identity.memo, state.final_bytes, false, state.members, state.members, 0};
}
} // namespace group_grow
