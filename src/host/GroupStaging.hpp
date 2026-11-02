// Optional same-file compressed contributions followed by one canonical pack.
// This trades explicit extra disk workspace for avoiding repeated group rewrites.
#pragma once
#include "host/GroupEngine.hpp"

namespace group_grow
{
class StagingEngine
{
    static constexpr size_t record_bytes = 1024;
    static constexpr uint8_t construction_version = 0xf2, finalizing_version = 0xf1;
    enum class Phase : uint8_t { stage, pack, compact };
    struct Record
    {
        std::array<char, 16> magic{};
        uint64_t sequence = 0;
        Header original{}, target{};
        uint64_t original_bytes = 0, alignment = 0, anchor = 0;
        uint64_t staged_end = 0, symbols = 0, canonical_bound = 0, storage_bytes = 0;
        uint64_t packed_chunks = 0, packed_bytes = 0, final_bytes = 0, compacted = 0;
        uint16_t staged = 0;
        Phase phase = Phase::stage;
        std::array<uint8_t, 255> memo{};
        std::array<char, 64> build{};
    };
    struct RunHeader
    {
        uint64_t bytes = 0;
        uint16_t first = 0, count = 0;
        uint32_t reserved = 0;
    };
    struct Run { uint64_t consumed, end, cursor; uint16_t first, count; };
    // Parsing cursors, durable consumption and cached read-ahead are separate.
    // All readers borrow disjoint slices of one aggregate input allocation.
    class BufferedReader
    {
    public:
        BufferedReader(StagingEngine& engine, Run& run, std::span<uint8_t> buffer)
            : engine_(engine), run_(run), buffer_(buffer.size() >= sizeof(uint64_t) ? buffer : std::span<uint8_t>{}) {}
        void read(std::span<uint8_t> bytes)
        {
            require(run_.cursor <= run_.end && bytes.size() <= run_.end - run_.cursor, "truncated staged record");
            while (!bytes.empty())
            {
                if (run_.cursor >= cached_begin_ && run_.cursor - cached_begin_ < cached_size_)
                {
                    size_t const offset = size_t(run_.cursor - cached_begin_);
                    size_t const count = std::min(bytes.size(), cached_size_ - offset);
                    std::memcpy(bytes.data(), buffer_.data() + offset, count);
                    run_.cursor += count; bytes = bytes.subspan(count);
                }
                else if (buffer_.empty() || bytes.size() >= buffer_.size())
                {
                    engine_.read_at(run_.cursor, bytes); run_.cursor += bytes.size(); return;
                }
                else
                {
                    cached_begin_ = run_.cursor;
                    cached_size_ = size_t(std::min<uint64_t>(buffer_.size(), run_.end - run_.cursor));
                    engine_.read_at(cached_begin_, buffer_.first(cached_size_));
                }
            }
        }
        uint64_t length()
        {
            uint64_t value = 0; read({reinterpret_cast<uint8_t*>(&value), sizeof(value)});
            require(value && value <= run_.end - run_.cursor, "invalid staged record bounds"); return value;
        }
        void skip(uint64_t bytes)
        { require(run_.cursor <= run_.end && bytes <= run_.end - run_.cursor, "invalid staged record bounds"); run_.cursor += bytes; }
        void copy(uint64_t offset, std::span<uint8_t> bytes)
        {
            require(offset >= run_.consumed && offset <= run_.end && bytes.size() <= run_.end - offset,
                    "invalid staged cached record bounds");
            if (offset >= cached_begin_ && offset - cached_begin_ < cached_size_)
            {
                size_t const first = size_t(offset - cached_begin_);
                size_t const count = std::min(bytes.size(), cached_size_ - first);
                std::memcpy(bytes.data(), buffer_.data() + first, count); offset += count; bytes = bytes.subspan(count);
            }
            if (!bytes.empty()) engine_.read_at(offset, bytes);
        }
    private:
        StagingEngine& engine_;
        Run& run_;
        std::span<uint8_t> buffer_;
        uint64_t cached_begin_ = 0;
        size_t cached_size_ = 0;
    };
    using DecodeTable = std::unique_ptr<FSE_DTable, decltype(&POS2_FSE_freeDTable)>;
    static DecodeTable decoding_table()
    {
        auto norm = group_tools::compression_counts();
        DecodeTable table(POS2_FSE_createDTable(11), POS2_FSE_freeDTable);
        require(table && !POS2_FSE_isError(POS2_FSE_buildDTable(table.get(), norm.data(), 177, 11)),
                "cannot build staging decode table");
        return table;
    }
    // Like Engine's coalesced codec blocks, keep sequential writes at window
    // granularity rather than issuing a syscall for every small group chunk.
    class BufferedWriter
    {
    public:
        BufferedWriter(StagingEngine& engine, uint64_t offset)
            : engine_(engine), offset_(offset), buffer_(size_t(engine.config_.window_bytes)) {}
        void write(std::span<uint8_t const> bytes)
        {
            while (!bytes.empty())
            {
                size_t n = std::min(bytes.size(), buffer_.size() - used_);
                std::memcpy(buffer_.data() + used_, bytes.data(), n); used_ += n; bytes = bytes.subspan(n);
                if (used_ == buffer_.size()) flush();
            }
        }
        void flush()
        {
            if (used_) { engine_.write_at(offset_, std::span(buffer_).first(used_)); offset_ += used_; used_ = 0; }
        }
        uint64_t position() const { return offset_ + used_; }
    private:
        StagingEngine& engine_;
        uint64_t offset_;
        std::vector<uint8_t> buffer_;
        size_t used_ = 0;
    };
    std::string path_;
    Config config_;
    Identity identity_;
    int fd_ = -1;
    uint64_t device_ = 0, inode_ = 0, alignment_ = record_bytes, expected_bytes_ = 0;
    bool created_ = false, confirmed_ = false, active_ = false, tail_records_ = false;
    bool cow_ = false;
    Record record_{};
    std::vector<uint64_t> old_sizes_;
    mutable Stats stats_{};
    unsigned packing_workers_ = 0;
    // Self-written committed sources need no repeated decode. Resume and an
    // observed source change rebuild this trust during read-only admission.
    std::optional<file_io::Stat> validated_source_;

    static uint64_t multiply(uint64_t a, uint64_t b)
    { require(!b || a <= uint64_t(INT64_MAX) / b, "staging size overflow"); return a * b; }
    uint64_t aligned(uint64_t value) const
    { return grow_add(value, value % alignment_ ? alignment_ - value % alignment_ : 0); }
    uint64_t slots_end(uint64_t first) const { return grow_add(grow_add(first, alignment_), record_bytes); }
    uint64_t tail_anchor(Record const& saved) const
    { return (saved.storage_bytes - record_bytes) / alignment_ * alignment_ - alignment_; }
    uint64_t size() const
    {
        file_io::Stat info{};
        require(file_io::inspect_fd(fd_, &info) == 0 && info.st_size >= 0, "cannot inspect staging target");
        return uint64_t(info.st_size);
    }
    void check_path() const
    {
        file_io::Stat info{};
        require(file_io::inspect_link(path_.c_str(), &info) == 0 && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
            uint64_t(info.st_dev) == device_ && uint64_t(info.st_ino) == inode_,
            "staging target filename or links changed");
        require(size() == expected_bytes_, "staging target size changed while open");
    }
    void truncate(uint64_t bytes)
    { require(!file_io::resize_file(fd_, file_io::Offset(bytes)), "cannot resize staging workspace"); expected_bytes_ = bytes; }
    void read_at(uint64_t offset, std::span<uint8_t> bytes) const
    {
        grow_add(offset, bytes.size());
        while (!bytes.empty())
        {
            auto const begin = std::chrono::steady_clock::now();
            auto n = file_io::read_at(fd_, bytes.data(), std::min(bytes.size(), size_t(file_io::max_transfer)), file_io::Offset(offset));
            stats_.read_ns += elapsed_ns(begin); ++stats_.read_calls;
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot read staging target");
            stats_.read_bytes += uint64_t(n);
            offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
        }
    }
    void write_at(uint64_t offset, std::span<uint8_t const> bytes)
    {
        grow_add(offset, bytes.size());
        while (!bytes.empty())
        {
            auto const begin = std::chrono::steady_clock::now();
            auto n = file_io::write_at(fd_, bytes.data(), std::min(bytes.size(), size_t(file_io::max_transfer)), file_io::Offset(offset));
            stats_.write_ns += elapsed_ns(begin);
            ++stats_.write_calls;
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot write staging target; retain this file for resume");
            stats_.written_bytes += uint64_t(n);
            offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
        }
    }
    template<class T> T read_value(uint64_t offset) const
    { T value{}; read_at(offset, {reinterpret_cast<uint8_t*>(&value), sizeof(value)}); return value; }
    template<class T> void write_value(uint64_t offset, T const& value)
    { write_at(offset, {reinterpret_cast<uint8_t const*>(&value), sizeof(value)}); }
    void sync()
    {
        int status;
        auto const begin = std::chrono::steady_clock::now();
        do { status = file_io::flush(fd_); } while (status && errno == EINTR);
        stats_.sync_ns += elapsed_ns(begin);
        require(!status, "cannot sync staging target; retain this file for resume"); ++stats_.syncs;
    }
    void evict(uint64_t begin, uint64_t bytes, bool completed = false) const
    {
        if (!completed && !config_.release_cache) return;
#ifdef __linux__
        // Only call after the durable checkpoint no longer needs these clean
        // pages. The file bytes remain available to recovery and later readers.
        if (bytes) (void)::posix_fadvise(fd_, file_io::Offset(begin), file_io::Offset(bytes), POSIX_FADV_DONTNEED);
#else
        (void)begin; (void)bytes;
#endif
    }
    void checkpoint()
    {
        check_path();
        require(record_.sequence < UINT64_MAX, "staging checkpoint sequence overflow");
        ++record_.sequence;
        std::array<uint8_t, record_bytes> bytes{};
        std::memcpy(bytes.data(), &record_, sizeof(record_));
        auto digest = grow_digest(std::span(bytes).first(record_bytes - 32));
        std::copy(digest.begin(), digest.end(), bytes.end() - 32);
        uint64_t const anchor = tail_records_ ? tail_anchor(record_) : record_.anchor;
        uint64_t const offset = anchor + record_.sequence % 2 * alignment_;
        write_at(offset, std::span(bytes).first(record_bytes / 2));
        grow_failpoint("stage-checkpoint-partial");
        write_at(offset + record_bytes / 2, std::span(bytes).subspan(record_bytes / 2));
        sync(); ++stats_.checkpoints;
    }
    std::optional<Record> load_record(uint64_t offset) const
    {
        if (offset > size() || size() - offset < record_bytes) return {};
        std::array<uint8_t, record_bytes> bytes{};
        read_at(offset, bytes);
        auto digest = grow_digest(std::span(bytes).first(record_bytes - 32));
        if (!std::equal(digest.begin(), digest.end(), bytes.end() - 32)) return {};
        Record saved{}; std::memcpy(&saved, bytes.data(), sizeof(saved));
        if (std::string_view(saved.magic.data(), saved.magic.size()) !=
            std::string_view("XCHPLOT2STAGEv1\0\0", 16)) return {};
        return saved;
    }
    void load_index(Header const& header, uint64_t bytes)
    {
        size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(header.k));
        require(header.chunk_index_offset >= sizeof(Header) + header.memo_length &&
            header.chunk_index_offset < bytes, "invalid original staging index bounds");
        std::vector<uint64_t> words((std::min<uint64_t>(bytes - header.chunk_index_offset, chunks * 8) + 7) / 8, 0);
        size_t const available = size_t(std::min<uint64_t>(bytes - header.chunk_index_offset, chunks * 8));
        read_at(header.chunk_index_offset, {reinterpret_cast<uint8_t*>(words.data()), available});
        BitReader bits(words, available * 8);
        old_sizes_.resize(chunks);
        uint64_t end = sizeof(Header) + header.memo_length;
        auto const expected = gsz_expected_for_g(header.group_size);
        unsigned const rice = unsigned(gsz_rice_k_for_g(header.group_size));
        for (auto& value : old_sizes_)
        {
            bool const negative = bits.read_u64(1);
            uint64_t const remainder = bits.read_u64(rice), quotient = bits.read_unary_64();
            require(quotient + rice + 2 <= 64, "invalid original staging size code");
            uint64_t const magnitude = (quotient << rice) | remainder;
            require(!negative || magnitude < expected, "invalid original staging size delta");
            value = negative ? expected - magnitude : grow_add(expected, magnitude);
            check_gsz_size(value, header.group_size); end = grow_add(end, value);
        }
        require(end == header.chunk_index_offset, "original staging index does not cover payload");
        BitWriter canonical; gsz_encode(old_sizes_, header.group_size, canonical);
        auto index = canonical.asBytes();
        require(index.size() <= available && std::equal(index.begin(), index.end(),
            reinterpret_cast<uint8_t const*>(words.data())), "noncanonical original staging index");
        identity_.bytes = grow_add(header.chunk_index_offset, index.size());
    }
    void validate_header(Header const& header) const
    {
        require(header.magic == PlotGroupFile::MAGIC && header.k >= 18 && header.k <= 28 && !(header.k & 1) &&
            header.strength >= 2 && header.strength <= 63, "invalid staging group identity");
        require(memory_bytes(header.k, config_) <= config_.memory_limit_bytes,
                "staging codec/index/windows exceed memory budget");
    }
    uint64_t canonical_bound(uint64_t symbols, uint16_t members) const
    {
        uint8_t const k = identity_.header.k;
        uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        // FSE_COMPRESSBOUND(n) is n + (n >> 7) + FSE_COMPRESSBOUND(0).
        // Sum the fixed overhead per chunk and use floor(sum) >= sum(floor).
        uint64_t const ans = grow_add(grow_add(symbols, symbols >> 7),
            multiply(chunks, POS2_FSE_compressBound(0)));
        // Unary deltas telescope to less than members*range in each chunk.
        uint64_t const unary = (multiply(members, range) - 1) / (178ull << (k - 8));
        uint64_t const bits = grow_add(multiply(symbols, k - 7), multiply(chunks, unary));
        uint64_t const low = grow_add(bits / 8, chunks); // per-chunk byte rounding
        // At most ten LEB bytes and one machine-word GSZ code per chunk.
        return grow_add(sizeof(Header) + identity_.memo.size(),
            grow_add(grow_add(ans, low), multiply(chunks, 10 + sizeof(uint64_t))));
    }
    uint64_t chunk_bound(uint64_t symbols, uint16_t members) const
    {
        uint8_t const k = identity_.header.k;
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        uint64_t const unary = (multiply(members, range) - 1) / (178ull << (k - 8));
        uint64_t const bits = grow_add(multiply(symbols, k - 7), unary);
        require(symbols <= SIZE_MAX, "staged chunk symbol count exceeds platform");
        return grow_add(grow_add(POS2_FSE_compressBound(size_t(symbols)), bits / 8 + (bits % 8 != 0)), 10);
    }
    static std::pair<uint64_t, size_t> ans_bounds(std::span<uint8_t const> prefix, uint64_t bytes)
    {
        uint64_t ans = 0; size_t leb = 0;
        for (unsigned shift = 0;; shift += 7)
        {
            require(leb < prefix.size() && shift <= 63, "truncated staged ANS size");
            auto byte = prefix[leb++]; require(shift != 63 || byte <= 1, "staged ANS size overflow");
            ans |= uint64_t(byte & 0x7f) << shift;
            if (!(byte & 0x80)) { require(leb == 1 || byte, "noncanonical staged ANS size"); break; }
        }
        require(leb <= bytes && ans && ans < bytes - leb, "invalid staged ANS bounds");
        return {ans, leb};
    }
    void reserve(uint64_t bytes)
    {
        check_path(); require(bytes <= config_.capacity_bytes, "staging workspace exceeds capacity");
#ifdef __linux__
        // Like Engine, reject reflinked old extents before overwrite admission.
        if (size())
        {
            alignas(struct fiemap) std::array<uint8_t, sizeof(struct fiemap) + sizeof(struct fiemap_extent) * 64> data{};
            uint64_t position = 0;
            while (position < size())
            {
                data.fill(0); auto* map = reinterpret_cast<struct fiemap*>(data.data());
                map->fm_start = position; map->fm_length = size() - position;
                map->fm_flags = FIEMAP_FLAG_SYNC; map->fm_extent_count = 64;
                int const status = ::ioctl(fd_, FS_IOC_FIEMAP, map);
                if (status)
                {
                    require(!cow_ && (errno == EOPNOTSUPP || errno == ENOTTY || errno == ENOSYS),
                        "cannot inspect staging target sharing"); break;
                }
                if (!map->fm_mapped_extents) break;
                auto* extents = reinterpret_cast<struct fiemap_extent*>(data.data() + sizeof(struct fiemap));
                for (uint32_t i = 0; i < map->fm_mapped_extents; ++i)
                    require(!(extents[i].fe_flags & FIEMAP_EXTENT_SHARED), "shared staging target requires unsharing");
                auto const& last = extents[map->fm_mapped_extents - 1];
                uint64_t const next = grow_add(last.fe_logical, last.fe_length);
                require(next > position, "invalid staging extent map"); position = next;
                if (last.fe_flags & FIEMAP_EXTENT_LAST) break;
            }
        }
        // CoW overwrites need additional currently-free blocks even after the
        // file's unwritten extents were reserved. Bound one checkpoint interval.
        if (cow_)
        {
            file_io::Space space{};
            require(!file_io::space_fd(fd_, &space) && space.f_frsize, "cannot inspect staging CoW headroom");
            uint64_t const extra = grow_add(grow_add(std::max<uint64_t>(config_.checkpoint_bytes, sizeof(Header)),
                grow_add(config_.codec_bytes, config_.window_bytes)), multiply(alignment_, 2));
            uint64_t const growth = bytes > size() ? bytes - size() : 0;
            uint64_t const needed = grow_add(growth, extra);
            require(needed / space.f_frsize + (needed % space.f_frsize != 0) <= space.f_bavail,
                    "staging physical CoW workspace exceeds available space");
        }
#endif
#if defined(__linux__) || defined(_WIN32)
        int status;
        do { status = file_io::allocate_file(fd_, file_io::keep_size, 0, file_io::Offset(bytes)); } while (status && errno == EINTR);
        require(!status, "staging physical allocation preflight failed without changing original bytes");
        sync(); grow_failpoint("stage-allocated");
#else
        throw std::runtime_error("same-file staging requires supported native physical allocation");
#endif
    }
    std::vector<Run> runs() const
    {
        std::vector<Run> result;
        uint64_t position = grow_add(record_.anchor, multiply(alignment_, 2));
        uint16_t member = record_.original.group_size;
        while (position < record_.staged_end)
        {
            auto header = read_value<RunHeader>(position);
            require(header.first == member && header.count && header.count <= UINT16_MAX - member &&
                !header.reserved && header.bytes >= sizeof(header) && header.bytes <= record_.staged_end - position,
                "invalid staged run coverage or bounds");
            uint64_t const end = position + header.bytes;
            result.push_back({position + sizeof(header), end, position + sizeof(header), header.first, header.count});
            member = uint16_t(member + header.count); position = end;
        }
        require(position == record_.staged_end && member == record_.staged, "incomplete staged run coverage");
        return result;
    }
    std::vector<BufferedReader> buffered_readers(std::vector<Run>& inputs, std::unique_ptr<uint8_t[]>& arena)
    {
        std::vector<BufferedReader> result; result.reserve(inputs.size());
        size_t const bytes = inputs.empty() ? 0 : size_t(config_.window_bytes / inputs.size());
        if (bytes >= sizeof(uint64_t)) arena = std::make_unique<uint8_t[]>(size_t(config_.window_bytes));
        for (size_t i = 0; i < inputs.size(); ++i)
            result.emplace_back(*this, inputs[i], arena ? std::span<uint8_t>(arena.get() + i * bytes, bytes) : std::span<uint8_t>{});
        return result;
    }
    void decode(std::span<uint8_t const> payload, uint64_t chunk, uint16_t members,
                FSE_DTable const* table, std::vector<uint64_t>& fragments, std::vector<size_t>& boundaries)
    {
        CodecScheduler::Lease compute(config_.cpu_scheduler.get());
        auto const [ans, leb] = ans_bounds(payload, payload.size());
        auto low = payload.subspan(leb + size_t(ans));
        std::vector<uint8_t> high(low.size() * 8 / (identity_.header.k - 7));
        require(!high.empty(), "empty staged chunk");
        size_t const count = POS2_FSE_decompress_usingDTable(high.data(), high.size(), payload.data() + leb, size_t(ans), table);
        require(count && !POS2_FSE_isError(count), "cannot decode staged chunk");
        std::vector<uint64_t> fields((low.size() + 7) / 8, 0);
        std::memcpy(fields.data(), low.data(), low.size()); BitReader bits(fields, low.size() * 8);
        uint8_t const k = identity_.header.k;
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS), start = chunk * range;
        uint64_t previous = start, region = 0, bit_count = 0;
        boundaries.push_back(fragments.size());
        for (size_t i = 0; i < count; ++i)
        {
            auto remainder = bits.read_u64(k - 8), quotient = bits.read_unary_64();
            require(high[i] < 178 && quotient + k - 7 <= 64, "invalid staged delta");
            bit_count += quotient + k - 7;
            uint64_t const delta = quotient * (178ull << (k - 8)) + (uint64_t(high[i]) << (k - 8)) + remainder;
            require(delta < start + uint64_t(members) * range - previous && (!i || delta), "staged fragment outside group");
            previous += delta;
            uint64_t const member = (previous - start) / range;
            require(member == region || (i && member == region + 1), "empty staged member region");
            if (member != region) { boundaries.push_back(fragments.size()); region = member; }
            require(fragments.size() < fragments.capacity(), "decoded staging chunk exceeds bounded workspace");
            fragments.push_back(previous - member * range);
        }
        require(region + 1 == members && (bit_count + 7) / 8 == low.size() &&
            (!(bit_count & 7) || !(low.back() >> (bit_count & 7))), "noncanonical staged chunk padding/coverage");
    }

public:
    static uint64_t memory_bytes(uint8_t k, Config const& config, uint16_t batch_members = 1)
    {
        require(k >= 18 && k <= 28 && !(k & 1), "invalid staging memory-admission k");
        uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
        static_assert(sizeof(Run) + sizeof(BufferedReader) + sizeof(uint64_t) * 2 +
            sizeof(size_t) + sizeof(FragmentRegion) <= 128);
        // With W=window_bytes, simultaneous packing storage is bounded by:
        // decoded uint64 fragments 64W/(k-7), decoder high symbols 8W/(k-7),
        // bit words W+7, payload W, aggregate read cache W, buffered output W,
        // and multi-chunk task/frontier metadata W. At k18 this is 127W/11+7,
        // below 12W for the supported header-sized minimum window. Codec
        // workers share codec_bytes, including their replacement allocations.
        // Fixed run/member descriptors cover one chunk when its metadata is
        // larger than W; size/index arrays also cover the decoder table and
        // small fixed objects. Every window vector reserves once while empty
        // and is filled only within that capacity, avoiding replacement peaks.
        return grow_add(grow_add(grow_add(config.codec_bytes, ParallelEncoder::memory_bytes(config.workers, !config.cpu_scheduler)), multiply(config.window_bytes, 12)),
            grow_add(multiply(chunks, sizeof(uint64_t) * 4),
                grow_add(multiply(UINT16_MAX, 128), multiply(batch_members, sizeof(size_t) * 2))));
    }
    StagingEngine(std::string path, Config config, bool resume = false, std::optional<Identity> initial = {});
    ~StagingEngine()
    {
        if (fd_ >= 0)
        {
            file_io::Stat opened{}, named{};
            if (!file_io::inspect_fd(fd_, &opened))
            {
                if (!active_ && stats_.syncs && opened.st_size > 0) evict(0, uint64_t(opened.st_size), true);
                if (created_ && !opened.st_size && !file_io::inspect_link(path_.c_str(), &named) &&
                    uint64_t(named.st_dev) == device_ && uint64_t(named.st_ino) == inode_) file_io::unlink_file(path_.c_str());
            }
            file_io::close_file(fd_);
        }
    }
    StagingEngine(StagingEngine const&) = delete;
    StagingEngine& operator=(StagingEngine const&) = delete;
    Identity const& identity() const { return identity_; }
    uint16_t staged_members() const { return active_ ? record_.staged : identity_.header.group_size; }
    uint16_t committed_member_count() const { return staged_members(); }
    // Staging uses serial codecs and the fixed aggregate codec/window budget.
    // Keep the runtime grant for an eventual ordinary-engine handoff.
    void set_workers(unsigned workers)
    {
        auto cfg = config_; cfg.workers = workers;
        require(workers && memory_bytes(identity_.header.k, cfg) <= config_.memory_limit_bytes,
            "staged worker grant exceeds memory admission");
        config_.workers = workers;
    }
    bool pending() const { return active_; }
    bool can_append() const { return !active_ || record_.phase == Phase::stage; }
    // Read-only dispatch. Private markers with damaged checkpoints still select
    // this protocol, whose constructor then rejects them without changing data.
    static bool pending(std::string const& path)
    {
        int fd = file_io::open_file(path.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow);
        if (fd < 0) return false;
        struct Close { int fd; ~Close() { file_io::close_file(fd); } } close{fd};
        file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
        if (file_io::inspect_fd(fd, &info) || !file_io::is_regular(info.st_mode) || info.st_size < file_io::Offset(sizeof(Header)) ||
            file_io::space_fd(fd, &allocation) || page <= 0 || info.st_blksize <= 0) return false;
        auto read = [&](uint64_t offset, std::span<uint8_t> bytes)
        {
            while (!bytes.empty())
            {
                auto n = file_io::read_at(fd, bytes.data(), bytes.size(), file_io::Offset(offset));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) return false;
                offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
            }
            return true;
        };
        Header header{};
        if (read(0, {reinterpret_cast<uint8_t*>(&header), sizeof(header)}) &&
            header.magic == PlotGroupFile::MAGIC && header.version == construction_version) return true;
        uint64_t const alignment = std::max({uint64_t(record_bytes), uint64_t(page), uint64_t(info.st_blksize),
            uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize)});
        if (uint64_t(info.st_size) < alignment + record_bytes) return false;
        uint64_t const last = (uint64_t(info.st_size) - record_bytes) / alignment * alignment;
        for (uint64_t offset : {last - alignment, last})
        {
            std::array<uint8_t, record_bytes> bytes{};
            if (!read(offset, bytes)) continue;
            auto digest = grow_digest(std::span(bytes).first(record_bytes - 32));
            if (std::equal(digest.begin(), digest.end(), bytes.end() - 32) &&
                !std::memcmp(bytes.data(), "XCHPLOT2STAGEv1\0\0", 16)) return true;
        }
        return false;
    }
    Stats const& stats() const { return stats_; }
    uint64_t read_calls() const { return stats_.read_calls; }
    unsigned packing_workers() const { return packing_workers_; }
    void confirm_identity()
    {
        check_path(); require(size() <= config_.capacity_bytes && identity_.bytes <= config_.capacity_bytes &&
            (!active_ || record_.storage_bytes <= config_.capacity_bytes), "saved staging workspace exceeds capacity");
        confirmed_ = true;
    }
    AppendResult append(Members members);
    Result finish();
};

inline StagingEngine::StagingEngine(std::string path, Config config, bool resume, std::optional<Identity> initial)
    : path_(std::move(path)), config_(config)
{
    static_assert(sizeof(Record) <= record_bytes - 32);
    require(config.window_bytes >= sizeof(Header) && config.window_bytes <= SIZE_MAX && config.codec_bytes && config.checkpoint_bytes && config.workers &&
        config.capacity_bytes <= uint64_t(INT64_MAX), "invalid staging budgets");
    if (initial && !resume)
    { fd_ = file_io::open_file(path_.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600); created_ = fd_ >= 0; }
    else fd_ = file_io::open_file(path_.c_str(), file_io::read_write | file_io::close_on_exec | file_io::no_follow);
    if (fd_ < 0 && errno == ENOENT && initial)
    { fd_ = file_io::open_file(path_.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600); created_ = fd_ >= 0; }
    require(fd_ >= 0, "cannot open staging target: " + path_);
    try
    {
        require(!file_io::lock_file(fd_, file_io::lock_exclusive | file_io::lock_nonblocking), "staging target is already in use");
        file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
        require(!file_io::inspect_fd(fd_, &info) && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
            !file_io::space_fd(fd_, &allocation) && allocation.f_bsize && info.st_blksize > 0 && page > 0,
            "cannot inspect staging target allocation");
        device_ = info.st_dev; inode_ = info.st_ino;
        alignment_ = std::max({uint64_t(record_bytes), uint64_t(page), uint64_t(info.st_blksize),
            uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize)});
#ifdef __linux__
        struct statfs filesystem{}; require(!::fstatfs(fd_, &filesystem), "cannot inspect staging filesystem");
        cow_ = unsigned(filesystem.f_type) == 0x9123683eu;
#endif
        std::optional<Record> saved;
        auto consider = [&](uint64_t offset)
        {
            auto candidate = load_record(offset);
            if (candidate && (!saved || candidate->sequence > saved->sequence)) saved = *candidate;
        };
        if (size() >= alignment_ + record_bytes)
        { uint64_t last = (size() - record_bytes) / alignment_ * alignment_; consider(last); consider(last - alignment_); }
        Header header{};
        if (!saved && size() >= sizeof(Header))
        {
            require(resume, "existing ordinary or staged groups require --resume");
            header = read_value<Header>(0); validate_header(header);
            require(header.version == PlotGroupFile::FORMAT_VERSION || header.version == construction_version,
                "different construction protocol; recover it with ordinary growth first");
            identity_.header = header;
            identity_.header.version = PlotGroupFile::FORMAT_VERSION;
            identity_.memo.resize(header.memo_length); read_at(sizeof(Header), identity_.memo);
            if (header.group_size) load_index(identity_.header, size());
            else
            {
                require(header.version == construction_version &&
                    header.chunk_index_offset == sizeof(Header) + header.memo_length, "invalid staging bootstrap");
                identity_.bytes = 0;
            }
            uint64_t const anchor = aligned(identity_.bytes ? identity_.bytes : sizeof(Header) + header.memo_length);
            consider(anchor); consider(anchor + alignment_);
            if (!saved)
            {
                require(resume || size() == identity_.bytes, "existing group requires --resume");
                require(header.version != construction_version || resume, "staging bootstrap requires --resume");
                require(header.version != construction_version || !header.group_size,
                    "damaged staged group requires a valid construction checkpoint");
                // A crash before the initial checkpoint leaves the original
                // canonical bytes intact. Defer orphan trimming until confirmed.
            }
        }
        if (saved)
        {
            require(resume, "in-progress staged group is not farmable; use --resume");
            record_ = *saved; validate_header(record_.original); validate_header(record_.target);
            std::array<char, 64> build{}; auto name = std::string_view(GROW_IDENTITY);
            std::copy_n(name.begin(), std::min(name.size(), build.size()), build.begin());
            require(record_.build == build, "staging checkpoint requires the same codec/build identity");
            require(record_.alignment == alignment_ && record_.sequence &&
                record_.storage_bytes >= slots_end(0) && record_.storage_bytes <= size() &&
                record_.anchor <= record_.storage_bytes && multiply(alignment_, 2) <= record_.storage_bytes - record_.anchor &&
                record_.original_bytes <= record_.anchor &&
                record_.original.version == PlotGroupFile::FORMAT_VERSION &&
                record_.target.version == PlotGroupFile::FORMAT_VERSION &&
                record_.original.group_id == record_.target.group_id && record_.original.k == record_.target.k &&
                record_.original.strength == record_.target.strength && record_.original.meta_group == record_.target.meta_group &&
                record_.original.memo_length == record_.target.memo_length &&
                (record_.original.group_size ? record_.original_bytes > record_.original.chunk_index_offset &&
                    record_.original.chunk_index_offset >= sizeof(Header) + record_.original.memo_length : !record_.original_bytes) &&
                record_.staged >= record_.original.group_size && record_.target.group_size == record_.staged &&
                record_.anchor == aligned(record_.original_bytes ? record_.original_bytes : sizeof(Header) + record_.original.memo_length) &&
                record_.staged_end >= record_.anchor + alignment_ * 2 &&
                record_.staged_end <= tail_anchor(record_) && record_.canonical_bound <= tail_anchor(record_) - record_.staged_end &&
                record_.packed_chunks <= PlotGroupFile::getChunkCountForK(record_.original.k) &&
                uint8_t(record_.phase) <= uint8_t(Phase::compact) &&
                record_.packed_bytes <= record_.canonical_bound && record_.final_bytes <= record_.canonical_bound &&
                record_.compacted <= record_.final_bytes &&
                (record_.phase != Phase::stage || (!record_.packed_chunks && !record_.packed_bytes && !record_.final_bytes && !record_.compacted)) &&
                (record_.phase != Phase::pack || (record_.packed_bytes >= sizeof(Header) + record_.original.memo_length &&
                    !record_.final_bytes && !record_.compacted)) &&
                (record_.phase != Phase::compact || (record_.storage_bytes == size() &&
                    record_.packed_chunks == PlotGroupFile::getChunkCountForK(record_.original.k) &&
                    record_.target.chunk_index_offset == record_.packed_bytes && record_.final_bytes > record_.packed_bytes &&
                    record_.packed_bytes >= sizeof(Header) + record_.original.memo_length)), "inconsistent staging checkpoint");
            identity_ = {record_.original, {record_.memo.begin(), record_.memo.begin() + record_.original.memo_length},
                record_.original_bytes, true, record_.original.group_size, record_.staged, record_.storage_bytes};
            active_ = true; tail_records_ = record_.phase == Phase::compact;
            if (record_.original.group_size && record_.phase != Phase::compact) load_index(record_.original, record_.original_bytes);
            if (record_.phase != Phase::compact) (void)runs();
        }
        else if (!size())
        {
            require(initial.has_value(), "empty staging target needs identity"); identity_ = *initial;
            validate_header(identity_.header);
            require(!identity_.header.group_size && identity_.memo.size() <= 255 &&
                identity_.header.memo_length == identity_.memo.size(), "invalid new staging identity");
            identity_.header.version = PlotGroupFile::FORMAT_VERSION;
            identity_.header.chunk_index_offset = sizeof(Header) + identity_.memo.size(); identity_.bytes = 0;
        }
        expected_bytes_ = size();
    }
    catch (...)
    {
        file_io::Stat opened{}, named{};
        if (created_ && !file_io::inspect_fd(fd_, &opened) && !opened.st_size && !file_io::inspect_link(path_.c_str(), &named) &&
            uint64_t(named.st_dev) == device_ && uint64_t(named.st_ino) == inode_) file_io::unlink_file(path_.c_str());
        file_io::close_file(fd_); fd_ = -1; throw;
    }
}

inline AppendResult StagingEngine::append(Members members)
{
    require(confirmed_ && (!active_ || record_.phase == Phase::stage), "staging append requires confirmed stage phase");
    check_path();
    if (!active_ && identity_.header.group_size && old_sizes_.empty()) load_index(identity_.header, identity_.bytes);
    uint16_t const before = staged_members();
    file_io::Stat source{};
    require(!file_io::inspect_fd(fd_, &source), "cannot inspect staged source stability");
    bool const validate_sources = before && (!validated_source_ ||
        source.st_mtim.tv_sec != validated_source_->st_mtim.tv_sec ||
        source.st_mtim.tv_nsec != validated_source_->st_mtim.tv_nsec ||
        source.st_ctim.tv_sec != validated_source_->st_ctim.tv_sec ||
        source.st_ctim.tv_nsec != validated_source_->st_ctim.tv_nsec);
    require(!members.empty() && members.size() <= size_t(UINT16_MAX - before), "invalid staged batch membership");
    require(memory_bytes(identity_.header.k, config_, uint16_t(members.size())) <= config_.memory_limit_bytes,
            "staging batch metadata exceeds memory budget");
    uint8_t const k = identity_.header.k;
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    uint64_t const old_symbols = active_ ? record_.symbols : (identity_.header.group_size ?
        multiply(identity_.header.chunk_index_offset - sizeof(Header) - identity_.memo.size(), 8) / (k - 7) : 0);
    uint64_t symbols = old_symbols;
    for (size_t i = 0; i < members.size(); ++i)
    {
        CodecScheduler::Lease compute(config_.cpu_scheduler.get());
        auto member = members[i]; require(!member.empty(), "empty staged incoming member");
        require(!member.compact() || member.k() == k, "staged compact member k mismatch");
        if (!member.compact()) require(member.raw().back() < (1ull << (k * 2)) &&
            std::is_sorted(member.raw().begin(), member.raw().end()), "unsorted/out-of-range staged incoming member");
        symbols = grow_add(symbols, member.size());
    }
    ChunkCodec codec(k, config_.codec_bytes, config_.cpu_scheduler);
    unsigned min_ans_bits = 11, max_ans_bits = 0;
    auto const counts = group_tools::compression_counts();
    for (size_t i = 0; i < 178; ++i)
    {
        require(counts[i] > 0 && counts[i] <= 1024, "unexpected staging FSE table");
        auto const frequency = uint32_t(counts[i]);
        min_ans_bits = std::min(min_ans_bits, 11u - std::bit_width(frequency - 1));
        max_ans_bits = std::max(max_ans_bits, 11u - (std::bit_width(frequency) - 1));
    }
    // Canonical FSE initializes two symbols without emitting their bits. Both
    // 11-bit final states and the end bit bound the canonical target's size.
    uint64_t const ans_overhead = 2 * 11 + 1;
    auto symbol_bounds = [&](uint64_t ans, uint16_t count)
    {
        uint64_t const least_bits = grow_add(multiply(ans - 1, 8), 1);
        uint64_t const emitted = least_bits > ans_overhead ? least_bits - ans_overhead : 0;
        // Persisted streams need not be canonical: the decoder's penultimate
        // transition overflows the remaining bits, then emits the other state.
        // Thus n-2 transitions fit, but n-1 exceed the available bits. Round
        // down for the lower bound, including an immediately overflowing pair.
        uint64_t const lower = std::max<uint64_t>(count,
            grow_add(2, emitted / max_ans_bits));
        uint64_t const bits = multiply(ans, 8);
        return std::pair{lower, grow_add(2, (bits > ans_overhead ? bits - ans_overhead : 0) / min_ans_bits)};
    };
    auto minimum_bytes = [&](uint64_t symbols)
    {
        uint64_t const low = multiply(symbols, k - 7);
        uint64_t const ans = grow_add(multiply(symbols > 2 ? symbols - 2 : 0, min_ans_bits), ans_overhead);
        return grow_add(1, grow_add(low / 8 + (low % 8 != 0), ans / 8 + (ans % 8 != 0)));
    };
    std::vector<uint64_t> sizes(size_t(chunks), 0);
    std::vector<size_t> cursors(members.size(), 0);
    std::vector<FragmentRegion> regions(members.size());
    auto inputs = active_ ? runs() : std::vector<Run>{};
    std::unique_ptr<uint8_t[]> read_arena;
    auto readers = buffered_readers(inputs, read_arena);
    uint16_t const after = uint16_t(before + members.size());
    std::vector<std::pair<uint64_t, uint64_t>> prior_chunks(inputs.size());
    DecodeTable table(nullptr, POS2_FSE_freeDTable);
    std::vector<uint64_t> fragments;
    std::vector<size_t> boundaries;
    std::vector<FragmentRegion> combined_regions;
    std::vector<uint8_t> payload;
    auto prepare = [&](uint64_t chunk)
    {
        uint64_t incoming = 0, end = (chunk + 1) << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        for (size_t i = 0; i < members.size(); ++i)
        {
            auto member = members[i];
            if (member.compact()) regions[i] = member.region(chunk);
            else
            {
                auto raw = member.raw(); size_t first = cursors[i];
                while (cursors[i] < raw.size() && raw[cursors[i]] < end) ++cursors[i];
                regions[i] = member.region(chunk, first, cursors[i] - first);
            }
            incoming = grow_add(incoming, regions[i].size());
        }
        codec.prepare(ChunkCodec::estimate_work(k, 0, incoming));
    };
    uint64_t run_bytes = sizeof(RunHeader);
    uint64_t old_position = sizeof(Header) + identity_.memo.size();
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        prepare(chunk); auto bytes = codec.encoded_size(chunk, {}, 0, regions); ++stats_.sized_chunks;
        uint64_t const incoming_symbols = codec.symbol_count();
        require(bytes <= config_.window_bytes, "staged contribution exceeds source window");
        sizes[size_t(chunk)] = bytes; run_bytes = grow_add(run_bytes, grow_add(sizeof(uint64_t), bytes));
        uint64_t const old_bytes = old_sizes_.empty() ? 0 : old_sizes_[size_t(chunk)];
        uint64_t combined = grow_add(bytes, old_bytes);
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto& run = inputs[i];
            auto prior_bytes = readers[i].length();
            prior_chunks[i] = {run.cursor, prior_bytes};
            combined = grow_add(combined, prior_bytes); readers[i].skip(prior_bytes);
        }
        // Every decoded symbol consumes at least k-7 low bits. Using all
        // compressed bytes is conservative and also covers all run boundaries.
        uint64_t const max_symbols = multiply(combined, 8) / (k - 7);
        bool const fits = combined <= config_.window_bytes &&
            ChunkCodec::estimate_work(k, 0, max_symbols) <= config_.codec_bytes;
        if (!fits) return {false, false, 0, 0, "staged final chunk exceeds codec/window admission", {}};
        bool decoded_all = false;
        auto decode_prior = [&](bool all = true)
        {
            if (decoded_all) return;
            if (!table)
            {
                table = decoding_table();
                fragments.reserve(size_t(multiply(config_.window_bytes, 8) / (k - 7)));
                boundaries.reserve(size_t(before) + 1); combined_regions.reserve(after);
                payload.reserve(size_t(config_.window_bytes));
            }
            fragments.clear(); boundaries.clear(); combined_regions.clear();
            auto append_payload = [&](uint16_t count)
            {
                size_t const first = fragments.size();
                decode(payload, chunk, count, table.get(), fragments, boundaries);
                if (first)
                    codec.validate_delta((1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS)) +
                        fragments[first] - fragments[first - 1]);
            };
            auto consume = [&](uint64_t offset, uint64_t length, uint16_t count)
            {
                payload.resize(size_t(length)); read_at(offset, payload);
                append_payload(count);
            };
            if (old_bytes && (all || inputs.empty())) consume(old_position, old_bytes, identity_.header.group_size);
            for (size_t i = all || inputs.empty() ? 0 : inputs.size() - 1; i < inputs.size(); ++i)
            {
                payload.resize(size_t(prior_chunks[i].second)); readers[i].copy(prior_chunks[i].first, payload);
                append_payload(inputs[i].count);
            }
            decoded_all = all;
        };
        // A resumed source has not been checked by this instance. Observed
        // timestamp changes also invalidate self-written-source trust. Keep
        // malformed input failures distinct from ordinary GSZ admission.
        if (before)
        {
            // Only the immediate predecessor is needed for a normal append.
            // Validate its rebased bridge even when the GSZ interval fits.
            decode_prior(validate_sources);
            codec.validate_delta((1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS)) +
                regions.front().front() - fragments.back());
        }
        uint64_t least_symbols = incoming_symbols, most_symbols = incoming_symbols;
        auto add_bounds = [&](std::span<uint8_t const> prefix, uint64_t length, uint16_t count)
        {
            auto const [ans, leb] = ans_bounds(prefix, length); (void)leb;
            auto const [least, most] = symbol_bounds(ans, count);
            least_symbols = grow_add(least_symbols, least); most_symbols = grow_add(most_symbols, most);
        };
        std::array<uint8_t, 10> prefix{};
        if (old_bytes)
        {
            auto bytes = std::span(prefix).first(size_t(std::min<uint64_t>(prefix.size(), old_bytes)));
            read_at(old_position, bytes); add_bounds(bytes, old_bytes, identity_.header.group_size);
        }
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto const [offset, length] = prior_chunks[i];
            auto bytes = std::span(prefix).first(size_t(std::min<uint64_t>(prefix.size(), length)));
            readers[i].copy(offset, bytes); add_bounds(bytes, length, inputs[i].count);
        }
        bool gsz_proven = true;
        try
        {
            check_gsz_size(minimum_bytes(least_symbols), after);
            check_gsz_size(chunk_bound(std::min(max_symbols, most_symbols), after), after);
        }
        catch (UnencodableChunkSize const&) { gsz_proven = false; }
        if (!gsz_proven)
        {
            // The proven FSE/low-bit interval usually settles admission using
            // only existing length prefixes. Sparse and boundary chunks still
            // use exact combined sizing before any persistent mutation.
            decode_prior();
            boundaries.push_back(fragments.size());
            require(boundaries.size() == size_t(before) + 1, "staged exact sizing member coverage mismatch");
            for (size_t i = 0; i < before; ++i)
                combined_regions.emplace_back(std::span<uint64_t const>(fragments).subspan(
                    boundaries[i], boundaries[i + 1] - boundaries[i]));
            combined_regions.insert(combined_regions.end(), regions.begin(), regions.end());
            // The compressed-size bound covers deduplicated symbols, including
            // incoming regions with many duplicates. Those immutable regions
            // are borrowed directly; the codec allocates only unique deltas.
            codec.prepare(ChunkCodec::estimate_work(k, 0, max_symbols));
            try { (void)codec.encoded_size(chunk, {}, 0, combined_regions); ++stats_.sized_chunks; }
            catch (UnencodableChunkSize const&)
            {
                // Persisted-input decoding above and other codec failures must
                // propagate. Only the codec's GSZ admission diagnostic is a
                // normal rejection eligible for ordinary-growth fallback.
                return {false, false, 0, 0, "staged final chunk exceeds GSZ admission", {}};
            }
        }
        old_position = grow_add(old_position, old_bytes);
    }
    for (auto const& run : inputs) require(run.cursor == run.end, "staged preflight trailing data");
    uint64_t const bound = canonical_bound(symbols, after);
    uint64_t const anchor = active_ ? record_.anchor : aligned(identity_.bytes ? identity_.bytes : sizeof(Header) + identity_.memo.size());
    uint64_t const begin = active_ ? record_.staged_end : grow_add(anchor, multiply(alignment_, 2));
    uint64_t const end = grow_add(begin, run_bytes), storage = slots_end(aligned(grow_add(end, bound)));
    AppendResult result{storage <= config_.capacity_bytes, false, bound, storage, {}, {}};
    if (!result.accepted) { result.reason = "compressed staging and canonical packing workspace exceed capacity"; return result; }
    reserve(storage);
    if (!active_)
    {
        tail_records_ = false;
        record_ = {}; std::memcpy(record_.magic.data(), "XCHPLOT2STAGEv1\0\0", 16);
        record_.original = identity_.header; record_.target = identity_.header;
        record_.original_bytes = identity_.bytes; record_.alignment = alignment_; record_.anchor = anchor;
        record_.staged_end = begin; record_.staged = before;
        record_.symbols = old_symbols;
        record_.canonical_bound = bound; record_.storage_bytes = storage;
        std::copy(identity_.memo.begin(), identity_.memo.end(), record_.memo.begin());
        auto build = std::string_view(GROW_IDENTITY);
        std::copy_n(build.begin(), std::min(build.size(), record_.build.size()), record_.build.begin());
        if (!identity_.bytes)
        {
            Header bootstrap = identity_.header; bootstrap.version = construction_version;
            write_value(0, bootstrap); write_at(sizeof(Header), identity_.memo); sync();
            auto parent = file_io::native_path(path_).parent_path(); if (parent.empty()) parent = ".";
            int directory = file_io::open_file(parent.c_str(), file_io::read_only | file_io::close_on_exec | file_io::directory);
            require(directory >= 0, "cannot open staging parent directory"); int status = file_io::flush(directory); file_io::close_file(directory);
            require(!status, "cannot sync staging parent directory"); grow_failpoint("stage-bootstrap");
        }
        truncate(storage);
        checkpoint(); active_ = true;
        uint8_t marker = construction_version; write_at(offsetof(Header, version), {&marker, 1}); sync();
        grow_failpoint("stage-initialized");
    }
    else truncate(storage);
    std::fill(cursors.begin(), cursors.end(), 0);
    BufferedWriter output(*this, begin + sizeof(RunHeader));
    uint64_t flushed = begin + sizeof(RunHeader);
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        prepare(chunk); auto bytes = codec.encode(chunk, {}, 0, regions); ++stats_.encode_chunks;
        require(bytes.size() == sizes[size_t(chunk)], "staged member changed after measurement");
        uint64_t const length = bytes.size();
        output.write({reinterpret_cast<uint8_t const*>(&length), sizeof(length)}); output.write(bytes);
        if (output.position() - flushed >= config_.checkpoint_bytes)
        {
            output.flush(); sync(); grow_failpoint("stage-run-partial-flushed");
            evict(flushed, output.position() - flushed); flushed = output.position();
        }
    }
    output.flush(); require(output.position() == end, "staged run length mismatch");
    write_value(begin, RunHeader{run_bytes, before, uint16_t(members.size()), 0}); sync(); grow_failpoint("stage-run-flushed");
    record_.staged_end = end; record_.staged = after; record_.target.group_size = after;
    record_.symbols = symbols; record_.canonical_bound = bound; record_.storage_bytes = storage;
    checkpoint(); grow_failpoint("stage-run-committed"); evict(begin, run_bytes);
    file_io::Stat committed{};
    require(!file_io::inspect_fd(fd_, &committed), "cannot inspect committed staged source");
    validated_source_ = committed;
    identity_.pending = true; identity_.old_members = record_.original.group_size;
    identity_.target_members = after; identity_.pending_peak_bytes = storage;
    result.result = {bound, storage, storage - bound, after}; return result;
}

inline Result StagingEngine::finish()
{
    require(confirmed_, "staging finalization requires confirmed identity");
    check_path();
    validated_source_.reset();
    if (!active_)
    {
        if (size() != identity_.bytes) { truncate(identity_.bytes); sync(); }
        return {identity_.bytes, identity_.bytes, 0, identity_.header.group_size};
    }
    if (record_.staged == record_.original.group_size)
    {
        require(record_.phase == Phase::stage, "empty staged pack phase");
        if (record_.original_bytes) { write_value(0, record_.original); sync(); }
        truncate(record_.original_bytes); sync();
        active_ = false; identity_.pending = false; identity_.pending_peak_bytes = 0;
        return {identity_.bytes, record_.storage_bytes, record_.storage_bytes - identity_.bytes, record_.staged};
    }
    if (record_.phase != Phase::compact)
    {
        auto inputs = runs();
        if (record_.phase == Phase::stage)
        {
            record_.phase = Phase::pack; record_.packed_bytes = sizeof(Header) + identity_.memo.size();
            Header target = record_.target; target.chunk_index_offset = 0;
            write_value(record_.staged_end, target); write_at(record_.staged_end + sizeof(Header), identity_.memo);
            sync(); checkpoint(); grow_failpoint("stage-pack-initialized");
        }
        auto table = decoding_table();
        uint8_t const k = identity_.header.k; uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
        std::unique_ptr<uint8_t[]> read_arena;
        auto readers = buffered_readers(inputs, read_arena);
        uint64_t const metadata_per_task = grow_add(sizeof(ChunkInput) + sizeof(uint64_t) * 8,
            grow_add(multiply(record_.staged, sizeof(FragmentRegion)), multiply(inputs.size(), sizeof(uint64_t))));
        size_t const task_limit = size_t(std::max<uint64_t>(1, std::min({chunks, uint64_t(65536),
            config_.window_bytes / metadata_per_task})));
        std::vector<uint64_t> sizes(size_t(chunks), 0), fragments;
        fragments.reserve(size_t(multiply(config_.window_bytes, 8) / (k - 7)));
        std::vector<size_t> boundaries; boundaries.reserve(size_t(record_.staged) + 1);
        std::vector<FragmentRegion> regions; regions.reserve(size_t(multiply(task_limit, record_.staged)));
        std::vector<ChunkInput> tasks; tasks.reserve(task_limit);
        std::vector<uint64_t> costs; costs.reserve(task_limit + 1);
        std::vector<uint64_t> frontiers; frontiers.reserve(size_t(multiply(task_limit, inputs.size())));
        std::vector<uint64_t> old_frontiers; old_frontiers.reserve(task_limit);
        std::vector<uint64_t> offsets(inputs.size());
        std::vector<uint8_t> payload; payload.reserve(size_t(config_.window_bytes));
        uint64_t old_position = sizeof(Header) + identity_.memo.size(), packed = old_position;
        uint64_t committed = record_.packed_bytes, old_consumed = old_position;
        BufferedWriter output(*this, record_.staged_end + record_.packed_bytes);
        std::unique_ptr<ParallelEncoder> encoder;
        uint64_t const baseline = ChunkCodec::estimate_base();
        require(baseline <= config_.codec_bytes, "staging codec budget cannot hold one worker");
        for (uint64_t first = 0; first < chunks;)
        {
            fragments.clear(); regions.clear(); tasks.clear(); costs.clear(); frontiers.clear(); old_frontiers.clear();
            costs.push_back(0);
            uint64_t source_bytes = 0, estimated_work = baseline, next = first;
            while (next < chunks && tasks.size() < task_limit)
            {
                uint64_t const old_bytes = old_sizes_.empty() ? 0 : old_sizes_[size_t(next)];
                uint64_t combined = old_bytes;
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    uint64_t const length = readers[i].length(); offsets[i] = inputs[i].cursor;
                    readers[i].skip(length); combined = grow_add(combined, length);
                }
                uint64_t const maximum_symbols = multiply(combined, 8) / (k - 7);
                uint64_t const work = ChunkCodec::estimate_work(k, 0, maximum_symbols) - baseline;
                if (combined > config_.window_bytes - source_bytes || work > config_.codec_bytes - estimated_work)
                {
                    for (size_t i = 0; i < inputs.size(); ++i) inputs[i].cursor = offsets[i] - sizeof(uint64_t);
                    require(!tasks.empty(), "combined staged chunk exceeds codec/window admission"); break;
                }
                source_bytes += combined; estimated_work += work;
                size_t const fragment_begin = fragments.size(); boundaries.clear();
                if (old_bytes)
                {
                    payload.resize(size_t(old_bytes)); read_at(old_position, payload);
                    decode(payload, next, record_.original.group_size, table.get(), fragments, boundaries);
                }
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    payload.resize(size_t(inputs[i].cursor - offsets[i])); readers[i].copy(offsets[i], payload);
                    decode(payload, next, inputs[i].count, table.get(), fragments, boundaries);
                    frontiers.push_back(inputs[i].cursor);
                }
                boundaries.push_back(fragments.size());
                require(boundaries.size() == size_t(record_.staged) + 1, "staged decoded member coverage mismatch");
                size_t const region_begin = regions.size();
                for (size_t i = 0; i < record_.staged; ++i)
                    regions.emplace_back(std::span<uint64_t const>(fragments).subspan(boundaries[i], boundaries[i + 1] - boundaries[i]));
                tasks.push_back({next, {}, 0, std::span<FragmentRegion const>(regions).subspan(region_begin, record_.staged)});
                costs.push_back(grow_add(costs.back(), ChunkCodec::estimate_work(k, 0, fragments.size() - fragment_begin) - baseline));
                old_position = grow_add(old_position, old_bytes); old_frontiers.push_back(old_position); ++next;
            }
            // Match ParallelEncoder's contiguous ceil-partitions exactly. A
            // serial partition always fits the conservative gather bound.
            unsigned workers = unsigned(std::min({uint64_t(config_.workers), uint64_t(tasks.size()), config_.codec_bytes / baseline}));
            for (;;)
            {
                size_t const per_worker = tasks.size() / workers + (tasks.size() % workers != 0);
                bool fits = true;
                for (size_t begin = 0; begin < tasks.size(); begin += per_worker)
                {
                    size_t const end = std::min(tasks.size(), begin + per_worker);
                    if (baseline + costs[end] - costs[begin] > config_.codec_bytes / workers) { fits = false; break; }
                }
                if (fits) break;
                require(workers > 1, "staged gathered codec window exceeds admission"); --workers;
            }
            if (!encoder || encoder->workers() != workers)
            {
                encoder.reset();
                encoder = std::make_unique<ParallelEncoder>(k, workers, config_.codec_bytes, config_.cpu_scheduler);
            }
            packing_workers_ = std::max(packing_workers_, workers);
            auto blocks = encoder->encode(tasks, std::span(sizes).subspan(size_t(first), tasks.size()));
            stats_.encode_chunks += tasks.size();
            size_t ordered = 0;
            for (auto const& block : blocks)
            {
                require(block.begin == ordered, "unordered staged codec output"); size_t offset = 0;
                for (size_t task = block.begin; task < block.end; ++task)
                {
                    uint64_t const chunk = first + task, bytes = sizes[size_t(chunk)];
                    require(offset <= block.bytes.size() && bytes <= block.bytes.size() - offset, "truncated staged codec output");
                    if (chunk >= record_.packed_chunks)
                    {
                        require(grow_add(packed, bytes) <= record_.canonical_bound, "canonical staging bound violated");
                        output.write(block.bytes.subspan(offset, size_t(bytes)));
                    }
                    offset += size_t(bytes); packed = grow_add(packed, bytes);
                    if (chunk + 1 == record_.packed_chunks) require(packed == record_.packed_bytes, "staged packed prefix changed");
                    if (chunk + 1 > record_.packed_chunks &&
                        (packed - committed >= config_.checkpoint_bytes || chunk + 1 == chunks))
                    {
                        output.flush(); sync(); grow_failpoint("stage-pack-flushed"); record_.packed_chunks = chunk + 1; record_.packed_bytes = packed;
                        checkpoint(); grow_failpoint("stage-pack-committed");
                        for (size_t i = 0; i < inputs.size(); ++i)
                        {
                            uint64_t const end = frontiers[task * inputs.size() + i];
                            evict(inputs[i].consumed, end - inputs[i].consumed); inputs[i].consumed = end;
                        }
                        evict(old_consumed, old_frontiers[task] - old_consumed); old_consumed = old_frontiers[task];
                        evict(record_.staged_end + committed, packed - committed); committed = packed;
                    }
                }
                require(offset == block.bytes.size(), "unexpected staged codec trailing output"); ordered = block.end;
            }
            require(ordered == tasks.size(), "incomplete staged codec output"); first = next;
        }
        for (auto const& run : inputs) require(run.cursor == run.end, "unexpected staged run trailing data");
        record_.target.chunk_index_offset = packed;
        BitWriter index; gsz_encode(sizes, record_.staged, index); auto bytes = index.asBytes();
        record_.final_bytes = grow_add(packed, bytes.size());
        require(record_.final_bytes <= record_.canonical_bound, "canonical staged index exceeds bound");
        write_at(record_.staged_end + packed, bytes); write_value(record_.staged_end, record_.target); sync();
        grow_failpoint("stage-tail-flushed");
        record_.phase = Phase::compact; record_.compacted = 0; tail_records_ = true; checkpoint();
        grow_failpoint("stage-tail-committed");
    }
    // Destination is always below source. Limit each transaction to the gap so
    // its writes cannot destroy its own replay source before its checkpoint.
    uint64_t const window = std::min(config_.window_bytes, record_.staged_end);
    uint64_t const transaction = std::min(std::max<uint64_t>(config_.checkpoint_bytes, sizeof(Header)), record_.staged_end);
    std::vector<uint8_t> buffer(size_t(std::min(window, record_.final_bytes)), 0);
    while (record_.compacted < record_.final_bytes)
    {
        uint64_t const begin = record_.compacted;
        uint64_t const end = grow_add(begin, std::min(transaction, record_.final_bytes - begin));
        uint64_t position = begin;
        while (position < end)
        {
            uint64_t const bytes = std::min<uint64_t>(buffer.size(), end - position);
            read_at(record_.staged_end + position, std::span(buffer).first(size_t(bytes)));
            if (!position)
            {
                // The complete tail remains authoritative until this front copy
                // is durable. Keep the front recognizable as construction.
                require(bytes >= sizeof(Header), "staging compaction transaction cannot hold header");
                buffer[offsetof(Header, version)] = construction_version;
            }
            size_t const partial = size_t(bytes / 2);
            write_at(position, std::span(buffer).first(partial)); grow_failpoint("stage-compact-partial");
            write_at(position + partial, std::span(buffer).subspan(partial, size_t(bytes) - partial)); position += bytes;
        }
        sync(); grow_failpoint("stage-compact-flushed"); record_.compacted = end; checkpoint();
        grow_failpoint("stage-compact-committed"); evict(begin, end - begin); evict(record_.staged_end + begin, end - begin);
    }
    Header marker = record_.target; marker.version = finalizing_version;
    auto header = std::span<uint8_t const>(reinterpret_cast<uint8_t const*>(&marker), sizeof(marker));
    write_at(0, header.first(sizeof(Header) / 2)); grow_failpoint("stage-final-header-partial");
    write_at(sizeof(Header) / 2, header.subspan(sizeof(Header) / 2)); sync(); grow_failpoint("stage-finalizing");
    std::array<uint8_t, 16> retired{}; uint64_t const anchor = tail_anchor(record_);
    unsigned const latest = unsigned(record_.sequence % 2);
    write_at(anchor + (1 - latest) * alignment_, retired); grow_failpoint("stage-retire-partial");
    write_at(anchor + latest * alignment_, retired); grow_failpoint("stage-retire-partial");
    sync(); grow_failpoint("stage-retired");
    truncate(record_.final_bytes); sync(); grow_failpoint("stage-truncated");
    uint8_t const version = PlotGroupFile::FORMAT_VERSION; write_at(offsetof(Header, version), {&version, 1}); sync();
    grow_failpoint("stage-published");
    identity_.header = record_.target; identity_.bytes = record_.final_bytes; identity_.pending = false;
    identity_.old_members = identity_.target_members = record_.staged; identity_.pending_peak_bytes = 0; active_ = false;
    tail_records_ = false;
    old_sizes_.clear();
    return {record_.final_bytes, record_.storage_bytes, record_.storage_bytes - record_.final_bytes, record_.staged};
}
} // namespace group_grow
