// Bounded compressed-run merges with durable block recycling. Intermediate
// passes copy the stream assembler's member-range packets; only final packing
// encodes the ordinary group, whose bytes remain identical to the reference.
#pragma once
#include "host/GroupMergeState.hpp"

namespace group_grow
{
inline uint64_t merge_placement_bytes(uint64_t block_bytes, uint64_t boundary, uint64_t end, uint64_t minimum)
{ return std::min(block_bytes, std::max(end, minimum) - boundary); }

// Use only proven output bytes and completed checkpoint observations. The
// usual interval remains available when placement cannot justify extra work.
inline bool prefer_merge_placement(Stats const& stats, uint64_t bytes, uint64_t slot_bytes,
                                   long double measured_checkpoint_ns)
{
    if (!bytes || !stats.read_bytes || !stats.written_bytes || !stats.syncs || !measured_checkpoint_ns) return false;
    long double const read = static_cast<long double>(stats.read_ns) / stats.read_bytes;
    long double const write = static_cast<long double>(stats.write_ns) / stats.written_bytes;
    long double const copy = bytes * (read + write);
    // Include the complete slot write and both syncs. The measured pack
    // checkpoint also covers map copying, hashing and fixed syscall costs.
    long double const checkpoint = std::max(measured_checkpoint_ns,
        slot_bytes * write + 2 * static_cast<long double>(stats.sync_ns) / stats.syncs);
    return checkpoint < copy * (1 - 0.15L);
}

class MergeEngine
{
    using Run = MergeState::Run;
    using Phase = MergeState::Phase;
    using Packet = MergeState::Packet;
    using Summary = MergeState::Summary;
    struct Admission
    {
        uint64_t live = 0, growth = 0, largest = 0, canonical = 0, minimum = 0, peak = 0, codec_required = 0;
        bool gsz = true, buffers = true;
    };
    MergeState arena_;
    bool confirmed_ = false;
    uint64_t contribution_bytes_ = 0;
    std::optional<file_io::Stat> trusted_;
    uint64_t minimum_final_bytes_ = 0;
    struct ReadOnly {};
    MergeEngine(std::string path, Config cfg, ReadOnly, ConstructionInspectionLimits const& limits, std::optional<Identity> const& initial, ConstructionStamp& target,
                std::array<uint8_t, 32>& authority)
        : arena_(std::move(path), cfg, MergeState::ReadOnly{}, limits, initial, target, authority)
    {
        require(cfg.merge_inputs == 2 || cfg.merge_inputs == 4, "merged input count must be two or four");
        require(required_memory() <= cfg.memory_limit_bytes, "merge workspace exceeds memory budget");
        require(!arena_.active || (arena_.state.chunk_bound <= cfg.window_bytes && arena_.state.codec_required <= cfg.codec_bytes),
            "saved merge chunks exceed codec/window budgets; restore the original budgets");
    }

    uint8_t k() const { return arena_.identity.header.k; }
    uint64_t chunks() const { return PlotGroupFile::getChunkCountForK(k()); }
    Config const& config() const { return arena_.config; }
    uint64_t read_budget() const { return MergeState::multiply(config().window_bytes, 8); }
    uint64_t transaction_bytes() const { return std::min(config().checkpoint_bytes, arena_.state.transaction_bytes); }
    uint64_t summary_bytes() const { return MergeState::multiply(chunks(), sizeof(Summary)); }
    uint64_t header_bytes() const { return sizeof(Header) + arena_.identity.memo.size(); }
    void remember()
    {
        file_io::Stat current{}; require(!publication::inspect(arena_.path().c_str(), &current), "cannot inspect merge source stability"); trusted_ = current;
    }
    bool trusted() const
    {
        if (!trusted_) return false;
        file_io::Stat now{}; require(!publication::inspect(arena_.path().c_str(), &now), "cannot inspect merge source stability");
        return now.st_size == trusted_->st_size && now.st_mtim.tv_sec == trusted_->st_mtim.tv_sec &&
            now.st_mtim.tv_nsec == trusted_->st_mtim.tv_nsec && now.st_ctim.tv_sec == trusted_->st_ctim.tv_sec &&
            now.st_ctim.tv_nsec == trusted_->st_ctim.tv_nsec;
    }
    std::vector<uint64_t> ordinary_sizes(Run const& run)
    {
        require(run.ordinary, "expected ordinary merge input");
        auto const& original = arena_.active ? arena_.state.original : arena_.identity.header;
        uint64_t const available = run.bytes - original.chunk_index_offset;
        require(available <= chunks() * 8 && original.chunk_index_offset >= run.released,
                "invalid remaining ordinary index");
        MergeState::Reader input(arena_, run, config().window_bytes, original.chunk_index_offset);
        std::vector<uint64_t> words(size_t((available + 7) / 8), 0), sizes(size_t(chunks()), 0);
        input.read({reinterpret_cast<uint8_t*>(words.data()), size_t(available)});
        input.evict(input.position());
        BitReader bits(words, available * 8);
        uint64_t const expected = gsz_expected_for_g(run.count); unsigned const rice = unsigned(gsz_rice_k_for_g(run.count));
        uint64_t total = header_bytes();
        for (auto& size : sizes)
        {
            bool const negative = bits.read_u64(1);
            uint64_t const remainder = bits.read_u64(rice), quotient = bits.read_unary_64();
            require(quotient + rice + 2 <= 64, "invalid ordinary merge size code");
            uint64_t const magnitude = (quotient << rice) | remainder;
            require(!negative || magnitude < expected, "invalid ordinary merge size delta");
            size = negative ? expected - magnitude : grow_add(expected, magnitude); check_gsz_size(size, run.count);
            total = grow_add(total, size);
        }
        require(total == original.chunk_index_offset, "ordinary merge index coverage changed");
        BitWriter canonical; gsz_encode(sizes, run.count, canonical); auto encoded = canonical.asBytes();
        require(encoded.size() == available && std::equal(encoded.begin(), encoded.end(),
            reinterpret_cast<uint8_t const*>(words.data())), "noncanonical ordinary merge index");
        return sizes;
    }
    class Inputs
    {
    public:
        Inputs(MergeEngine& engine, std::span<Run const> runs, uint64_t first_chunk = 0)
            : engine_(engine), runs_(runs.begin(), runs.end())
        {
            require(!runs.empty() && runs.size() <= MergeState::max_runs, "invalid merge input count");
            engine_.arena_.stats.max_active_runs = std::max<uint64_t>(engine_.arena_.stats.max_active_runs, runs.size());
            uint64_t const buffer = std::max<uint64_t>(sizeof(Packet), engine.read_budget() / runs.size());
            for (auto const& run : runs)
            {
                readers_.push_back(std::make_unique<MergeState::Reader>(engine_.arena_, run, buffer));
                if (run.ordinary && first_chunk < engine_.chunks()) sizes_ = engine_.ordinary_sizes(run);
            }
        }
        size_t size() const { return readers_.size(); }
        Run const& run(size_t i) const { return runs_[i]; }
        MergeState::Reader& reader(size_t i) { return *readers_[i]; }
        Packet packet(size_t i, uint64_t chunk, uint16_t member)
        {
            auto const& run = runs_[i];
            Packet packet{};
            if (run.ordinary)
            {
                require(member == run.first && chunk < sizes_.size() && sizes_[size_t(chunk)] <= UINT32_MAX,
                        "invalid ordinary merge record");
                packet = {run.first, run.count, uint32_t(sizes_[size_t(chunk)])};
            }
            else packet = reader(i).value<Packet>();
            require(packet.first == member && packet.count && packet.count <= uint32_t(run.first) + run.count - member &&
                packet.bytes > 1 && packet.bytes <= reader(i).remaining(), "invalid merged packet coverage or size");
            return packet;
        }
        void end_chunk(size_t i, uint64_t chunk)
        {
            if (runs_[i].ordinary && chunk + 1 == engine_.chunks()) reader(i).skip(reader(i).remaining());
        }
        uint64_t chunk_bytes(uint64_t chunk)
        {
            uint64_t bytes = 0;
            for (size_t i = 0; i < size(); ++i)
            {
                auto const& source = run(i); uint64_t const position = reader(i).position();
                for (uint32_t member = source.first; member < uint32_t(source.first) + source.count;)
                {
                    auto const entry = packet(i, chunk, uint16_t(member));
                    bytes = grow_add(bytes, grow_add(sizeof(Packet), entry.bytes));
                    reader(i).skip(entry.bytes); member += entry.count;
                }
                reader(i).seek(position);
            }
            return bytes;
        }
        void finish()
        {
            for (auto& reader : readers_)
            {
                require(!reader->remaining(), "unexpected merged run trailing data");
                reader->evict(reader->position());
            }
        }
        void evict_read()
        { for (auto& reader : readers_) reader->evict_read(); }
    private:
        MergeEngine& engine_;
        std::vector<Run> runs_;
        std::vector<std::unique_ptr<MergeState::Reader>> readers_;
        std::vector<uint64_t> sizes_;
    };
    struct Gather
    {
        std::vector<uint8_t> payload;
        std::vector<EncodedChunk> contributions;
        explicit Gather(MergeEngine const& engine)
        {
            payload.reserve(size_t(engine.config().window_bytes));
            contributions.reserve(UINT16_MAX);
        }
        uint64_t read(MergeEngine& engine, Inputs& input, uint64_t chunk, bool append = false)
        {
            if (!append) { payload.clear(); contributions.clear(); }
            uint64_t bytes = 0; uint32_t members = 0;
            for (size_t i = 0; i < input.size(); ++i)
            {
                auto const& run = input.run(i); uint32_t const end = uint32_t(run.first) + run.count;
                for (uint32_t member = run.first; member < end;)
                {
                    auto const packet = input.packet(i, chunk, uint16_t(member));
                    require(packet.bytes <= engine.config().window_bytes &&
                        bytes <= engine.config().window_bytes - packet.bytes &&
                        packet.bytes <= payload.capacity() - payload.size(), "merged source chunk exceeds window budget");
                    size_t const offset = payload.size(); payload.resize(offset + packet.bytes);
                    auto part = std::span(payload).subspan(offset, packet.bytes); input.reader(i).read(part);
                    require(contributions.size() < contributions.capacity(), "merged packet descriptors exceed window budget");
                    contributions.push_back({part, packet.count});
                    members += packet.count;
                    bytes = grow_add(bytes, grow_add(sizeof(Packet), packet.bytes)); member += packet.count;
                }
                input.end_chunk(i, chunk);
            }
            require(members == engine.arena_.state.members, "merged final member coverage changed");
            return bytes;
        }
    };
    std::vector<Summary> summaries();
    Admission bounds(std::span<Summary const> summaries, uint16_t members, size_t runs,
                     Members incoming = {}, std::span<uint64_t> sizes = {}) const;
    uint64_t workspace(Admission const& plan, size_t runs) const;
    Admission exact(std::span<Summary const> summaries, Members incoming, std::span<uint64_t const> sizes);
    Admission compact(Admission plan, std::span<Summary const> summary, Members incoming,
                      std::span<uint64_t const> sizes);
    void validate_remaining();
    unsigned encoding_workers(std::span<uint64_t const> costs, uint64_t budget) const;
    struct ContributionWindow
    {
        uint64_t first = 0, next = 0, codec_ns = 0;
        std::vector<FragmentRegion> regions;
        std::vector<ChunkInput> tasks;
        std::vector<ChunkSummary> details;
        std::vector<uint64_t> costs, lengths;
        std::vector<EncodedBlock> blocks;
        ContributionWindow(size_t limit, size_t members)
        {
            regions.reserve(limit * members); tasks.reserve(limit); details.reserve(limit);
            costs.reserve(limit + 1); lengths.reserve(limit); blocks.reserve(limit);
        }
    };
    // Match the ordinary grower's bounded producer. Joining on unwind drains
    // the codec before either window or its borrowed member spans can die.
    class WindowWorker
    {
    public:
        WindowWorker() : thread_([this] { run(); }) {}
        ~WindowWorker()
        {
            {
                std::lock_guard lock(mutex_); stopping_ = true;
            }
            ready_.notify_one(); thread_.join();
        }
        std::future<void> encode(std::function<void()> encode)
        {
            std::packaged_task<void()> task(std::move(encode)); auto future = task.get_future();
            {
                std::lock_guard lock(mutex_);
                require(!task_.valid(), "merged window encoder queue is full"); task_ = std::move(task);
            }
            ready_.notify_one(); return future;
        }
    private:
        void run()
        {
            for (;;)
            {
                std::packaged_task<void()> task;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [this] { return stopping_ || task_.valid(); });
                    if (stopping_ && !task_.valid()) return;
                    task = std::move(task_);
                }
                task();
            }
        }
        std::mutex mutex_;
        std::condition_variable ready_;
        std::packaged_task<void()> task_;
        bool stopping_ = false;
        std::thread thread_;
    };
    static uint64_t contribution_metadata(size_t members)
    {
        return sizeof(ChunkInput) + sizeof(ChunkSummary) + sizeof(EncodedBlock) +
            3 * sizeof(uint64_t) + members * sizeof(FragmentRegion);
    }
    size_t contribution_limit(size_t members) const
    {
        return size_t(std::max<uint64_t>(1, std::min({chunks(), uint64_t(65536),
            config().window_bytes / contribution_metadata(members)})));
    }
    uint64_t contribution(Members members, std::span<uint64_t> sizes,
                          std::span<Summary> summary, MergeState::Writer* output = nullptr);
    void merge(size_t first, size_t count = 0);
    void balance(size_t limit, bool all = true);
    void pack();
    void pack_chunks(Inputs& input, MergeState::Writer& output, std::span<uint64_t> sizes);
    void arrange();

    void incoming_regions(Members members, uint64_t chunk, std::vector<size_t>& cursors,
                          std::vector<FragmentRegion>& regions, bool advance = true) const
    {
        uint64_t const end = (chunk + 1) << (k() + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        for (size_t i = 0; i < members.size(); ++i)
        {
            auto member = members[i];
            if (member.compact()) regions[i] = member.region(chunk);
            else
            {
                auto raw = member.raw(); size_t const first = cursors[i]; size_t next = first;
                while (next < raw.size() && raw[next] < end) ++next;
                regions[i] = member.region(chunk, first, next - first);
                if (advance) cursors[i] = next;
            }
        }
    }
    static uint64_t symbol_count(MemberRegions regions)
    { uint64_t result = 0; for (size_t i = 0; i < regions.size(); ++i) result = grow_add(result, regions[i].size()); return result; }

public:
    static uint64_t block_bytes(Config const& cfg)
    {
        // Keep recycling and placement granular when a longer transaction is
        // worthwhile. Independent moves can then share a checkpoint while
        // copying through the existing bounded window.
        return std::max(cfg.window_bytes,
            std::min({cfg.checkpoint_bytes, cfg.capacity_bytes / 64, uint64_t(256) * 1024 * 1024}));
    }
    static size_t packing_inputs(Config const& cfg)
    {
        // The four-input policy's eight-window preread budget can retain
        // eight full-window readers. The two-input policy keeps two readers.
        return cfg.merge_inputs == 4 ? 8 : 2;
    }
    static uint64_t memory_bytes(uint8_t k, Config const& cfg, uint16_t batch_members = 1)
    {
        require(k >= 18 && k <= 28 && !(k & 1) && cfg.window_bytes, "invalid merge memory admission");
        uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
        // One summary table, incoming/exact/original size tables, decoder and
        // gathered codec windows, eight I/O windows shared by active readers,
        // bounded member descriptors, simultaneous checkpoint/map copies,
        // and a per-run frontier table in each live packing window. Incoming
        // descriptors have their own bounded windows; retained encoded output
        // remains charged to the aggregate codec budget.
        uint64_t const maps = std::min<uint64_t>(cfg.capacity_bytes / cfg.window_bytes, MergeState::reserved_block - 1);
        uint64_t const windows = cfg.io_overlap && cfg.workers > 1 ? 28 : 20;
        uint64_t const frontiers = MergeState::multiply(cfg.window_bytes, cfg.io_overlap && cfg.workers > 1 ? 2 : 1);
        uint64_t const contributions = MergeState::multiply(std::max(cfg.window_bytes, contribution_metadata(batch_members)),
            cfg.io_overlap && cfg.workers > 1 ? 2 : 1);
        return grow_add(grow_add(frontiers, contributions), grow_add(grow_add(grow_add(cfg.codec_bytes, ParallelEncoder::memory_bytes(cfg.workers, !cfg.cpu_scheduler)), MergeState::multiply(cfg.window_bytes, windows)),
            grow_add(MergeState::multiply(chunks, sizeof(Summary) + 32),
                grow_add(MergeState::multiply(maps, 32), grow_add(MergeState::multiply(UINT16_MAX, 128),
                    grow_add(1024 * 1024, MergeState::multiply(batch_members, sizeof(size_t) + sizeof(FragmentRegion))))))));
    }
    MergeEngine(std::string path, Config cfg, bool resume = false, std::optional<Identity> initial = {})
        : arena_(std::move(path), cfg, resume, std::move(initial))
    {
        require(cfg.merge_inputs == 2 || cfg.merge_inputs == 4, "merged input count must be two or four");
        require(required_memory() <= cfg.memory_limit_bytes, "merge workspace exceeds memory budget");
        if (!arena_.active)
        {
            arena_.state = arena_.layout(block_bytes(cfg));
        }
        else require(arena_.state.chunk_bound <= cfg.window_bytes && arena_.state.codec_required <= cfg.codec_bytes,
                     "saved merge chunks exceed codec/window budgets; restore the original budgets");
    }
    static bool pending(std::string const& path) { return MergeState::pending(path); }
    static ConstructionInspection inspect(std::string const& path, Config cfg, ConstructionInspectionLimits const& limits,
        std::optional<Identity> const& initial = {})
    {
        ConstructionInspection result; std::array<uint8_t, 32> authority{};
        MergeEngine engine(path, cfg, ReadOnly{}, limits, initial, result.target, authority);
        result.identity = engine.identity(); result.checkpoint = engine.arena_.active;
        result.checkpoint_identity = engine.arena_.checkpoint_identity();
        result.committed_members = engine.committed_member_count(); result.can_append = engine.can_append();
        result.inspection_memory_bytes = engine.arena_.active ? grow_add(MergeState::multiply(engine.arena_.state.slot_bytes, 4),
            grow_add(MergeState::multiply(engine.arena_.map.size(), 32), 65536)) : Engine::inspection_memory_bytes(engine.k());
        limits.check(result.inspection_memory_bytes); result.recovery_memory_bytes = engine.required_memory();
        result.authority_digest = construction_digest(result, authority, "merged");
        require(construction_stamp(engine.arena_.fd_, path) == result.target, "merged destination changed during inspection"); return result;
    }
    uint64_t required_memory(uint16_t batch_members = 1, unsigned workers = 0) const
    {
        Config cfg = config();
        if (workers) cfg.workers = workers;
        if (arena_.active)
            cfg.capacity_bytes = std::max(MergeState::multiply(arena_.state.map_count, cfg.window_bytes),
                std::min(cfg.capacity_bytes, MergeState::multiply(arena_.state.map_count, arena_.state.block_bytes)));
        return memory_bytes(k(), cfg, batch_members);
    }
    Identity const& identity() const { return arena_.identity; }
    uint16_t committed_member_count() const { return arena_.active ? arena_.state.members : arena_.identity.header.group_size; }
    bool can_append() const { return !arena_.active || !arena_.state.finishing; }
    Stats const& stats() const { return arena_.stats; }
    uint64_t contribution_bytes() const { return contribution_bytes_; }
    // Contribution/packing encoders are local to each completed operation.
    // Moving from one worker may enable another bounded I/O window.
    void set_workers(unsigned workers)
    {
        require(workers && required_memory(1, workers) <= config().memory_limit_bytes,
            "merged worker grant exceeds memory admission");
        arena_.config.workers = workers;
    }
    void confirm_identity() { arena_.confirm(); confirmed_ = true; }
    AppendResult append(Members members);
    Result finish();
};

inline std::vector<MergeEngine::Summary> MergeEngine::summaries()
{
    config().check_running();
    std::vector<Summary> result(size_t(chunks()), Summary{});
    if (arena_.active && arena_.state.summary.bytes)
    {
        require(arena_.state.summary.bytes == summary_bytes(), "invalid merged summary length");
        MergeState::Reader input(arena_, arena_.state.summary, read_budget());
        input.read({reinterpret_cast<uint8_t*>(result.data()), result.size() * sizeof(Summary)});
        input.evict(input.position());
        require(grow_digest({reinterpret_cast<uint8_t const*>(result.data()), result.size() * sizeof(Summary)}, config().cpu_scheduler.get()) ==
            arena_.state.summary_digest, "merged summary checksum mismatch");
        CodecScheduler::Lease compute(config().cpu_scheduler.get());
        uint64_t const range = 1ull << (k() + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        for (size_t chunk = 0; chunk < result.size(); ++chunk)
        {
            auto const& entry = result[chunk]; auto const& summary = entry.codec;
            require(summary.symbols && summary.symbols <= config().window_bytes * 8 / (k() - 7) &&
                summary.low_bits >= summary.symbols * (k() - 7) && summary.low_bits <= summary.symbols * 64 &&
                summary.min_ans_bits >= summary.symbols && summary.min_ans_bits <= summary.max_ans_bits &&
                summary.max_ans_bits <= summary.symbols * 11 && summary.last >= chunk * range &&
                summary.last - chunk * range < range && entry.bytes && entry.bytes <= config().window_bytes,
                "invalid merged summary values");
        }
    }
    else if (arena_.state.members)
    {
        require(arena_.state.run_count == 1 && arena_.state.runs[0].ordinary,
                "committed merged group is missing its summary");
        Inputs input(*this, std::span(arena_.state.runs).first(1));
        ChunkCodec codec(k(), config().codec_bytes, config().cpu_scheduler); std::vector<uint8_t> payload; payload.reserve(size_t(config().window_bytes));
        for (uint64_t chunk = 0; chunk < chunks(); ++chunk)
        {
            auto const packet = input.packet(0, chunk, 0);
            require(packet.bytes <= config().window_bytes, "ordinary merge chunk exceeds source window");
            codec.prepare(ChunkCodec::estimate_work(k(), packet.bytes, 0));
            payload.resize(packet.bytes); input.reader(0).read(payload);
            auto const begin = std::chrono::steady_clock::now();
            require(codec.scan(chunk, payload, packet.count) == packet.bytes, "ordinary merge chunk has trailing bytes");
            arena_.stats.decode_ns += elapsed_ns(begin);
            result[size_t(chunk)] = {codec.summary(chunk, packet.count), grow_add(sizeof(Packet), packet.bytes)};
            input.end_chunk(0, chunk);
            input.evict_read();
        }
        input.finish(); remember();
    }
    return result;
}
inline uint64_t MergeEngine::workspace(Admission const& plan, size_t runs) const
{
    // At a checkpoint, each run wastes less than one segment at either end.
    // The next output transaction may hold T + largest_chunk bytes before
    // any inputs can be released. Two summary tables coexist during append;
    // the final index is at most one word per chunk. One extra segment lets
    // final block permutation break cycles without overwriting a live source.
    uint64_t const transaction = std::min(transaction_bytes(), plan.live);
    uint64_t const slack = MergeState::multiply(runs + 8, arena_.state.block_bytes);
    uint64_t const working = grow_add(grow_add(transaction, plan.largest),
        grow_add(slack, grow_add(MergeState::multiply(summary_bytes(), 2), chunks() * 8)));
    return arena_.required_bytes(arena_.state, grow_add(grow_add(plan.live, plan.growth), working));
}
inline MergeEngine::Admission MergeEngine::bounds(std::span<Summary const> summaries, uint16_t members, size_t runs,
                                                  Members incoming, std::span<uint64_t> sizes) const
{
    Admission result; result.minimum = result.growth = header_bytes(); uint64_t output = header_bytes(), consumed = 0;
    require(incoming.empty() || sizes.size() == chunks(), "invalid merged contribution bounds");
    auto reduce = [&](size_t chunk, MemberRegions regions, ChunkSummary const& detail)
    {
        auto entry = summaries[chunk];
        if (!incoming.empty())
        {
            auto const [least, most] = ChunkCodec::size_bounds(detail);
            sizes[chunk] = most;
            try { check_gsz_size(least, uint16_t(incoming.size())); check_gsz_size(most, uint16_t(incoming.size())); }
            catch (UnencodableChunkSize const&) { result.gsz = false; }
            if (most > config().window_bytes || most > UINT32_MAX ||
                ChunkCodec::estimate_work(k(), 0, symbol_count(regions)) > config().codec_bytes)
                result.buffers = false;
            entry.codec = ChunkCodec::combine(k(), entry.codec, detail, regions[0].front(), chunk);
            entry.bytes = grow_add(entry.bytes, grow_add(sizeof(Packet), most));
        }
        auto const [least, most] = ChunkCodec::size_bounds(entry.codec);
        uint64_t const source_work = std::max(ChunkCodec::estimate_work(k(), 0, entry.codec.symbols),
            ChunkCodec::estimate_work(k(), entry.bytes, 0));
        uint64_t const work = std::max(source_work, ChunkCodec::estimate_work(k(), most, 0));
        result.codec_required = std::max(result.codec_required, work);
        if (entry.bytes > config().window_bytes || source_work > config().codec_bytes) result.buffers = false;
        if (most > config().window_bytes || work > config().codec_bytes) result.gsz = false;
        try { check_gsz_size(least, members); check_gsz_size(most, members); }
        catch (UnencodableChunkSize const&) { result.gsz = false; }
        output = grow_add(output, most); consumed = grow_add(consumed, entry.bytes);
        result.minimum = grow_add(result.minimum, least);
        result.growth = std::max(result.growth, output > consumed ? output - consumed : 0);
        result.largest = std::max(result.largest, std::max(entry.bytes, most));
    };
    bool immutable = !incoming.empty();
    for (size_t member = 0; member < incoming.size(); ++member)
        immutable = immutable && incoming[member].compact() && incoming[member].immutable_id();
    size_t const limit = contribution_limit(incoming.size());
    unsigned const workers = unsigned(std::min({uint64_t(config().workers), uint64_t(limit),
        config().codec_bytes / ChunkCodec::estimate_base()}));
    if (immutable && workers > 1)
    {
        // Only independent immutable chunk scans run on the codec pool. Reduce
        // the bounded results in order so prefix-sensitive capacity admission
        // is identical to the serial path, without a per-member summary table.
        ParallelEncoder encoder(k(), workers, config().codec_bytes, config().cpu_scheduler);
        std::vector<FragmentRegion> regions; regions.reserve(limit * incoming.size());
        std::vector<ChunkInput> tasks; tasks.reserve(limit);
        std::vector<ChunkSummary> details; details.reserve(limit);
        for (size_t first = 0; first < summaries.size();)
        {
            config().check_running();
            size_t const count = std::min(limit, summaries.size() - first);
            tasks.clear(); regions.clear(); details.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                size_t const begin = regions.size();
                for (size_t member = 0; member < incoming.size(); ++member)
                    regions.push_back(incoming[member].region(first + i));
                tasks.push_back({first + i, {}, 0,
                    std::span<FragmentRegion const>(regions).subspan(begin, incoming.size())});
            }
            encoder.summarize(tasks, details);
            for (size_t i = 0; i < count; ++i) reduce(first + i, tasks[i].new_members, details[i]);
            first += count;
        }
    }
    else
    {
        std::vector<FragmentRegion> regions(incoming.size());
        std::vector<size_t> cursors(incoming.size(), 0);
        for (size_t chunk = 0; chunk < summaries.size(); ++chunk)
        {
            config().check_running();
            ChunkSummary detail;
            if (!incoming.empty())
            {
                incoming_regions(incoming, chunk, cursors, regions);
                detail = ChunkCodec::fresh_summary(k(), chunk, regions, config().cpu_scheduler.get());
            }
            reduce(chunk, regions, detail);
        }
    }
    result.canonical = grow_add(output, chunks() * 8); result.live = consumed;
    result.growth = std::max(result.growth, result.canonical > consumed ? result.canonical - consumed : 0);
    // live + growth is max(output prefix + source suffix). Bounding each
    // source packet above therefore also bounds the whole working reserve,
    // even though growth alone can decrease when source bytes increase.
    result.peak = workspace(result, runs); return result;
}
inline MergeEngine::Admission MergeEngine::exact(std::span<Summary const> summaries, Members incoming,
                                                std::span<uint64_t const> sizes)
{
    std::unique_ptr<Inputs> input;
    if (arena_.state.run_count) input = std::make_unique<Inputs>(*this, std::span(arena_.state.runs).first(arena_.state.run_count));
    ChunkCodec codec(k(), config().codec_bytes, config().cpu_scheduler); Gather gathered(*this);
    std::vector<size_t> cursors(incoming.size(), 0); std::vector<FragmentRegion> fresh(incoming.size());
    std::vector<uint64_t> canonical(size_t(chunks()), 0);
    uint16_t const members = uint16_t(arena_.state.members + incoming.size());
    Admission result; result.growth = header_bytes(); uint64_t output = header_bytes(), consumed = 0;
    for (uint64_t chunk = 0; chunk < chunks(); ++chunk)
    {
        config().check_running();
        uint64_t source = 0;
        if (input) source = gathered.read(*this, *input, chunk);
        else { gathered.payload.clear(); gathered.contributions.clear(); }
        if (!incoming.empty())
        {
            incoming_regions(incoming, chunk, cursors, fresh);
            source = grow_add(source, grow_add(sizeof(Packet), sizes[size_t(chunk)]));
        }
        require(source == summaries[size_t(chunk)].bytes, "merged summary does not cover its packets");
        codec.prepare(ChunkCodec::estimate_work(k(), gathered.payload.size(), symbol_count(fresh)));
        auto const begin = std::chrono::steady_clock::now();
        uint64_t const bytes = input ? codec.joined_size(chunk, gathered.contributions, fresh)
            : codec.encoded_size(chunk, {}, 0, fresh);
        arena_.stats.codec_ns += elapsed_ns(begin); ++arena_.stats.sized_chunks;
        require(codec.summary(chunk, members) == summaries[size_t(chunk)].codec,
                "merged summary differs from its compressed data");
        canonical[size_t(chunk)] = bytes; output = grow_add(output, bytes); consumed = grow_add(consumed, source);
        result.growth = std::max(result.growth, output > consumed ? output - consumed : 0);
        result.largest = std::max(result.largest, std::max(source, bytes));
        result.codec_required = std::max(result.codec_required,
            std::max(ChunkCodec::estimate_work(k(), 0, summaries[size_t(chunk)].codec.symbols),
                ChunkCodec::estimate_work(k(), std::max(source, bytes), 0)));
        if (bytes > config().window_bytes || result.codec_required > config().codec_bytes) result.buffers = false;
        if (input) input->evict_read();
    }
    if (input) input->finish();
    BitWriter index; gsz_encode(canonical, members, index);
    result.live = consumed; result.minimum = output; result.canonical = grow_add(output, index.asBytes().size());
    result.growth = std::max(result.growth, result.canonical > consumed ? result.canonical - consumed : 0);
    result.peak = workspace(result, arena_.state.run_count + (incoming.empty() ? 0 : 1)); return result;
}
inline MergeEngine::Admission MergeEngine::compact(Admission plan, std::span<Summary const> summary,
                                                  Members incoming, std::span<uint64_t const> sizes)
{
    while (plan.buffers && plan.peak > config().capacity_bytes && arena_.state.run_count > 1)
    {
        arena_.restore_extent(); balance(arena_.state.run_count - 1);
        plan = bounds(summary, uint16_t(arena_.state.members + incoming.size()), arena_.state.run_count + 1);
        if (!plan.gsz || plan.peak > config().capacity_bytes) plan = exact(summary, incoming, sizes);
    }
    return plan;
}
inline void MergeEngine::validate_remaining()
{
    config().check_running();
    if (!arena_.active) { minimum_final_bytes_ = 0; return; }
    if (trusted()) return;
    minimum_final_bytes_ = 0;
    auto const& state = arena_.state;
    if (state.phase == Phase::layout) return; // the authenticated map is the recovery authority
    ChunkCodec codec(k(), config().codec_bytes, config().cpu_scheduler); std::vector<uint8_t> payload; payload.reserve(size_t(config().window_bytes));
    auto packets = [&](Run const& run, uint64_t first, uint64_t end)
    {
        if (first == end) return;
        Inputs input(*this, std::span(&run, 1), first);
        for (uint64_t chunk = first; chunk < end; ++chunk)
        {
            for (uint32_t member = run.first; member < uint32_t(run.first) + run.count;)
            {
                auto const packet = input.packet(0, chunk, uint16_t(member));
                require(packet.bytes <= config().window_bytes, "saved merged packet exceeds source window");
                codec.prepare(ChunkCodec::estimate_work(k(), packet.bytes, 0));
                payload.resize(packet.bytes); input.reader(0).read(payload);
                auto const begin = std::chrono::steady_clock::now();
                require(codec.scan(chunk, payload, packet.count) == packet.bytes, "saved merged packet has trailing data");
                arena_.stats.decode_ns += elapsed_ns(begin);
                member += packet.count;
            }
            input.end_chunk(0, chunk);
            input.evict_read();
        }
        input.finish();
    };
    for (size_t i = 0; i < state.run_count; ++i)
    {
        bool const consumed = state.phase == Phase::pack ||
            (state.phase == Phase::merge && i >= state.merge_first && i - state.merge_first < state.merge_count);
        packets(state.runs[i], consumed ? state.chunk : 0, chunks());
    }
    if (state.phase == Phase::merge) packets(state.output, 0, state.chunk);
    // Canonical packed prefixes are scanned before their next write in pack().
    remember();
}
inline unsigned MergeEngine::encoding_workers(std::span<uint64_t const> costs, uint64_t budget) const
{
    uint64_t const baseline = ChunkCodec::estimate_base();
    require(costs.size() > 1 && budget >= baseline, "empty merged codec window");
    size_t const count = costs.size() - 1;
    unsigned workers = unsigned(std::min({uint64_t(config().workers), uint64_t(count), budget / baseline}));
    for (;;)
    {
        size_t const per_worker = count / workers + (count % workers != 0);
        bool fits = true;
        for (size_t begin = 0; begin < count; begin += per_worker)
        {
            size_t const end = std::min(count, begin + per_worker);
            if (baseline + costs[end] - costs[begin] > budget / workers) { fits = false; break; }
        }
        if (fits) return workers;
        require(workers > 1, "merged codec window exceeds admission"); --workers;
    }
}

inline uint64_t MergeEngine::contribution(Members members, std::span<uint64_t> sizes,
                                          std::span<Summary> summary, MergeState::Writer* output)
{
    config().check_running();
    uint64_t const baseline = ChunkCodec::estimate_base();
    require(config().codec_bytes >= baseline && sizes.size() == chunks() &&
        (output || summary.size() == chunks()) && (summary.empty() || summary.size() == chunks()),
        "invalid merged contribution workspace");
    size_t const limit = contribution_limit(members.size());
    std::vector<size_t> cursors(members.size(), 0);
    std::vector<FragmentRegion> regions(members.size());
    bool immutable = true;
    for (size_t member = 0; member < members.size(); ++member)
        immutable = immutable && members[member].compact() && members[member].immutable_id();
    uint64_t required = baseline;
    // Discover the largest chunk from boundaries/counts before any output is
    // borrowed. Keep the encoder and its worker budgets fixed across windows.
    // Raw/borrowed input keeps the original serial per-window path rather
    // than paying another full fragment scan just to discover its boundaries.
    if (immutable)
        for (uint64_t chunk = 0; chunk < chunks(); ++chunk)
        {
            config().check_running();
            uint64_t symbols = 0;
            for (size_t member = 0; member < members.size(); ++member)
            {
                auto const view = members[member];
                symbols = grow_add(symbols, view.boundary(size_t(chunk + 1)) - view.boundary(size_t(chunk)));
            }
            required = std::max(required, ChunkCodec::estimate_work(k(), 0, symbols));
        }
    require(required <= config().codec_bytes, "merged contribution chunk exceeds codec budget");
    bool const overlap = immutable && output && config().io_overlap && config().workers > 1 &&
        required <= config().codec_bytes / 2;
    uint64_t const budget = overlap ? config().codec_bytes / 2 : config().codec_bytes;
    unsigned workers = unsigned(std::min<uint64_t>(config().workers, budget / required));
    require(workers, "merged contribution requires a codec worker");
    auto prepare = [&](ContributionWindow& window, uint64_t first)
    {
        config().check_running();
        auto& tasks = window.tasks; auto& costs = window.costs;
        tasks.clear(); window.regions.clear(); costs.clear(); costs.push_back(0);
        window.first = first;
        uint64_t const available = budget - (immutable ? uint64_t(workers) : 1) * baseline;
        while (tasks.size() < limit && first + tasks.size() < chunks())
        {
            uint64_t const chunk = first + tasks.size();
            incoming_regions(members, chunk, cursors, regions, false);
            uint64_t const cost = ChunkCodec::estimate_work(k(), 0, symbol_count(regions)) - baseline;
            if (cost > available - costs.back()) break;
            size_t const begin = window.regions.size();
            window.regions.insert(window.regions.end(), regions.begin(), regions.end());
            tasks.push_back({chunk, {}, 0,
                std::span<FragmentRegion const>(window.regions).subspan(begin, members.size())});
            costs.push_back(grow_add(costs.back(), cost));
            for (size_t i = 0; i < members.size(); ++i) if (!members[i].compact()) cursors[i] += regions[i].size();
        }
        require(!tasks.empty(), "merged contribution chunk exceeds codec budget");
        size_t count = tasks.size();
        if (!immutable) workers = encoding_workers(costs, budget);
        else
        {
            for (;;)
            {
                size_t const partitions = std::min<size_t>(workers, count);
                size_t const per_worker = count / partitions + (count % partitions != 0);
                bool fits = true;
                for (size_t begin = 0; begin < count; begin += per_worker)
                    if (baseline + costs[std::min(count, begin + per_worker)] - costs[begin] > budget / workers)
                    { fits = false; break; }
                if (fits) break;
                count = std::max<size_t>(1, count / 2);
            }
        }
        tasks.resize(count); window.regions.resize(count * members.size()); costs.resize(count + 1);
        window.lengths.resize(count); window.details.resize(!output || !summary.empty() ? count : 0);
        window.next = first + count;
        arena_.stats.contribution_workers = std::max<uint64_t>(arena_.stats.contribution_workers, workers);
    };
    auto current = std::make_unique<ContributionWindow>(limit, members.size());
    auto next = overlap ? std::make_unique<ContributionWindow>(limit, members.size()) : nullptr;
    std::unique_ptr<ParallelEncoder> encoder;
    auto encode = [&](ContributionWindow& window)
    {
        if (!encoder || encoder->workers() != workers)
        { encoder.reset(); encoder = std::make_unique<ParallelEncoder>(k(), workers, config().codec_bytes, config().cpu_scheduler); }
        auto const begin = std::chrono::steady_clock::now();
        if (!output) encoder->measure(window.tasks, window.lengths, window.details);
        else
        {
            auto const blocks = encoder->encode(window.tasks, window.lengths, overlap, window.details);
            window.blocks.assign(blocks.begin(), blocks.end());
        }
        window.codec_ns = elapsed_ns(begin);
    };
    // The worker dies before the encoder and both windows, including when a
    // writer failure leaves an asynchronous codec exception still pending.
    auto worker = overlap ? std::make_unique<WindowWorker>() : nullptr;
    std::future<void> pending;
    uint64_t run_bytes = 0;
    prepare(*current, 0); encode(*current);
    for (;;)
    {
        config().check_running();
        if (next && current->next < chunks())
        {
            prepare(*next, current->next);
            pending = worker->encode([&, following = next.get()] { encode(*following); });
            ++arena_.stats.overlap_windows;
        }
        auto const first = current->first;
        auto const& tasks = current->tasks; auto const& lengths = current->lengths;
        auto const& details = current->details;
        arena_.stats.codec_ns += current->codec_ns;
        if (!output)
        {
            arena_.stats.sized_chunks += tasks.size();
            for (size_t i = 0; i < tasks.size(); ++i)
            {
                uint64_t const bytes = lengths[i], chunk = first + i;
                require(bytes <= config().window_bytes && bytes <= UINT32_MAX, "merged contribution exceeds source window");
                sizes[size_t(chunk)] = bytes;
                auto& entry = summary[size_t(chunk)];
                entry.codec = ChunkCodec::combine(k(), entry.codec, details[i], tasks[i].new_members[0].front(), chunk);
                entry.bytes = grow_add(entry.bytes, grow_add(sizeof(Packet), bytes));
                run_bytes = grow_add(run_bytes, grow_add(sizeof(Packet), bytes));
            }
        }
        else
        {
            arena_.stats.encode_chunks += tasks.size();
            size_t ordered = 0;
            for (auto const& block : current->blocks)
            {
                require(block.begin == ordered, "unordered merged contribution output"); size_t offset = 0;
                for (size_t i = block.begin; i < block.end; ++i)
                {
                    uint64_t const bytes = lengths[i];
                    size_t const chunk = size_t(first) + i;
                    require((summary.empty() ? bytes == sizes[chunk] : bytes <= sizes[chunk]) &&
                        bytes <= config().window_bytes && bytes <= UINT32_MAX &&
                        offset <= block.bytes.size() && bytes <= block.bytes.size() - offset,
                            "merged contribution changed after admission");
                    if (!summary.empty())
                    {
                        sizes[chunk] = bytes;
                        auto& entry = summary[chunk];
                        entry.codec = ChunkCodec::combine(k(), entry.codec, details[i], tasks[i].new_members[0].front(), chunk);
                        entry.bytes = grow_add(entry.bytes, grow_add(sizeof(Packet), bytes));
                    }
                    output->value(Packet{committed_member_count(), uint16_t(members.size()), uint32_t(bytes)});
                    output->write(block.bytes.subspan(offset, size_t(bytes))); offset += size_t(bytes);
                    run_bytes = grow_add(run_bytes, grow_add(sizeof(Packet), bytes));
                }
                require(offset == block.bytes.size(), "unexpected merged contribution trailing bytes"); ordered = block.end;
            }
            require(ordered == tasks.size(), "incomplete merged contribution output");
            // Full writer buffers have already reached the file. Exercise
            // provisional recovery at window boundaries without per-chunk
            // failpoint overhead or changing the writer's I/O grouping.
            if (output->position() >= config().window_bytes) grow_failpoint("merge-contribution-partial");
        }
        if (current->next == chunks()) break;
        if (next) { pending.get(); current.swap(next); }
        else { prepare(*current, current->next); encode(*current); }
    }
    return run_bytes;
}

inline AppendResult MergeEngine::append(Members members)
{
    config().check_running();
    require(confirmed_ && can_append(), "merge append requires confirmed staging phase"); arena_.check_path();
    if (!arena_.active)
    {
        auto record = arena_.layout(block_bytes(config())); uint64_t const minimum = arena_.required_bytes(record, 0);
        if (minimum > config().capacity_bytes)
            return {false, false, arena_.identity.bytes, minimum, "merge metadata and recovery reserve exceed capacity", {}};
        arena_.prepare(record);
    }
    bool const recheck = arena_.active && !trusted();
    validate_remaining();
    if (arena_.active && arena_.state.phase == Phase::merge)
    { arena_.restore_extent(); merge(arena_.state.merge_first); }
    PhaseTimer planning(arena_.stats, WorkPhase::plan, config().progress);
    uint16_t const before = committed_member_count();
    require(!members.empty() && members.size() <= size_t(UINT16_MAX - before), "invalid merged batch membership");
    require(required_memory(uint16_t(members.size())) <= config().memory_limit_bytes,
            "merge batch metadata exceeds memory budget");
    bool single_pass = !(before && recheck);
    for (size_t i = 0; i < members.size(); ++i)
    {
        CodecScheduler::Lease compute(config().cpu_scheduler.get());
        auto const member = members[i];
        require(!member.empty() && (!member.compact() || member.k() == k()), "invalid merged incoming member");
        single_pass = single_pass && member.compact() && member.immutable_id();
        if (!member.compact()) require(member.raw().back() < (1ull << (k() * 2)) &&
            std::is_sorted(member.raw().begin(), member.raw().end()), "unsorted/out-of-range merged member");
    }
    // Metadata runs may exceed the final reader set while the balanced tail
    // develops. Always leave a table slot before admitting another full run.
    if (arena_.active && arena_.state.run_count == MergeState::max_runs)
    { arena_.restore_extent(); balance(MergeState::max_runs - 1); }
    auto summary = summaries(); std::vector<uint64_t> sizes(size_t(chunks()), 0);
    uint16_t const after = uint16_t(before + members.size());
    Admission plan;
    uint64_t run_bytes = 0;
    if (single_pass)
    {
        plan = bounds(summary, after, arena_.state.run_count + 1, members, sizes);
        single_pass = plan.buffers && plan.gsz && plan.peak <= config().capacity_bytes;
    }
    if (!single_pass)
    {
        run_bytes = contribution(members, sizes, summary);
        plan = bounds(summary, after, arena_.state.run_count + 1);
        if (!plan.buffers) return {false, false, plan.canonical, plan.peak, "merged final chunk exceeds codec/window admission", {}};
        if (!plan.gsz || plan.peak > config().capacity_bytes || (before && recheck))
            plan = exact(summary, members, sizes);
        plan = compact(plan, summary, members, sizes);
    }
    AppendResult result{plan.buffers && plan.peak <= config().capacity_bytes, false, plan.canonical, plan.peak, {}, {}};
    if (!result.accepted)
    {
        result.reason = plan.buffers ? "merged run and recoverable working reserve exceed capacity" :
            "merged final chunk exceeds codec/window admission"; return result;
    }
    planning.finish(); PhaseTimer encoding(arena_.stats, WorkPhase::encode, config().progress);
    if (!arena_.active) arena_.initialize(arena_.state, plan.peak);
    else { arena_.restore_extent(); arena_.reserve(std::max(plan.peak, arena_.state.storage_bytes)); }
    auto& state = arena_.state;
    require(state.run_count < MergeState::max_runs, "balanced merge run limit exceeded");
    // Provisional packets use only blocks free in the committed map. Until
    // merge-run publishes both this run and its summary, recovery discards
    // every provisional write and retains the previous committed members.
    Run incoming; incoming.first = before; incoming.count = uint16_t(members.size());
    MergeState::Writer output(arena_, incoming, config().window_bytes);
    uint64_t const written = contribution(members, sizes, single_pass ? std::span(summary) : std::span<Summary>{}, &output);
    if (single_pass)
    {
        run_bytes = written;
        auto const actual = bounds(summary, after, state.run_count + 1);
        require(actual.buffers && actual.gsz && actual.peak <= plan.peak,
                "merged contribution exceeded admitted bounds");
        plan = actual;
    }
    else require(written == run_bytes, "merged contribution total changed after admission");
    output.flush(); require(incoming.bytes == run_bytes, "merged contribution length changed");
    grow_failpoint("merge-contribution-flushed");
    Run table;
    MergeState::Writer table_output(arena_, table, config().window_bytes);
    table_output.write({reinterpret_cast<uint8_t const*>(summary.data()), summary.size() * sizeof(Summary)}); table_output.flush();
    arena_.discard(state.summary); state.summary = table;
    state.summary_digest = grow_digest({reinterpret_cast<uint8_t const*>(summary.data()), summary.size() * sizeof(Summary)}, config().cpu_scheduler.get());
    state.runs[state.run_count++] = incoming; state.members = after; state.target.group_size = after;
    state.live_bytes = plan.live; state.growth_bound = plan.growth; state.chunk_bound = plan.largest;
    state.codec_required = plan.codec_required;
    arena_.checkpoint("merge-run"); minimum_final_bytes_ = plan.minimum;
    config().check_running();
    output.evict(true); table_output.evict(true);
    encoding.finish();
    balance(packing_inputs(config()), false); remember();
    contribution_bytes_ = run_bytes;
    config().check_running();
    result.result = {plan.canonical, state.storage_bytes, state.storage_bytes > plan.canonical ? state.storage_bytes - plan.canonical : 0, after};
    return result;
}
inline void MergeEngine::merge(size_t first, size_t count)
{
    config().check_running();
    PhaseTimer timing(arena_.stats, WorkPhase::merge, config().progress);
    auto& state = arena_.state;
    if (state.phase != Phase::merge)
    {
        require(state.phase == Phase::stage && count >= 2 && count <= state.merge_inputs &&
            first < state.run_count && count <= state.run_count - first, "invalid balanced merge selection");
        state.phase = Phase::merge; state.merge_first = uint32_t(first); state.chunk = 0;
        state.merge_count = uint8_t(count);
        state.output = {}; state.output.first = state.runs[first].first;
        uint32_t members = 0;
        for (size_t i = first; i < first + count; ++i) members += state.runs[i].count;
        require(members <= UINT16_MAX, "merged membership overflow"); state.output.count = uint16_t(members);
        arena_.checkpoint("merge-start");
    }
    require(state.merge_first == first && (!count || count == state.merge_count), "different active merge selected");
    count = state.merge_count;
    arena_.stats.merge_inputs = std::max<uint64_t>(arena_.stats.merge_inputs, count);
    Inputs input(*this, std::span(state.runs).subspan(first, count), state.chunk);
    MergeState::Writer output(arena_, state.output, config().window_bytes);
    uint64_t committed = state.output.bytes;
    for (uint64_t chunk = state.chunk; chunk < chunks(); ++chunk)
    {
        config().check_running();
        for (size_t i = 0; i < input.size(); ++i)
        {
            auto const& run = input.run(i); uint32_t const end = uint32_t(run.first) + run.count;
            for (uint32_t member = run.first; member < end;)
            {
                auto const packet = input.packet(i, chunk, uint16_t(member)); output.value(packet);
                uint64_t remaining = packet.bytes;
                while (remaining)
                {
                    auto bytes = input.reader(i).peek(); size_t const count = size_t(std::min<uint64_t>(bytes.size(), remaining));
                    output.write(bytes.first(count)); input.reader(i).skip(count); remaining -= count;
                }
                member += packet.count;
            }
            input.end_chunk(i, chunk);
        }
        if (output.position() - committed >= transaction_bytes() || chunk + 1 == chunks())
        {
            output.flush();
            for (size_t i = 0; i < count; ++i) arena_.release(state.runs[first + i], input.reader(i).position());
            state.chunk = chunk + 1; arena_.checkpoint("merge-copy");
            for (size_t i = 0; i < count; ++i) input.reader(i).evict(input.reader(i).position());
            output.evict(chunk + 1 == chunks()); committed = state.output.bytes;
        }
    }
    input.finish();
    state.runs[first] = state.output;
    for (size_t i = first + 1; i + count - 1 < state.run_count; ++i) state.runs[i] = state.runs[i + count - 1];
    size_t const previous = state.run_count; state.run_count -= uint16_t(count - 1);
    for (size_t i = state.run_count; i < previous; ++i) state.runs[i] = {};
    state.output = {}; state.chunk = 0; state.merge_count = 0; state.phase = Phase::stage;
    arena_.checkpoint("merge-complete"); ++arena_.stats.merge_passes;
}
inline void MergeEngine::balance(size_t limit, bool all)
{
    auto& state = arena_.state;
    if (state.phase == Phase::merge) merge(state.merge_first);
    require(limit, "empty merged input limit");
    while (state.run_count > limit)
    {
        size_t const count = std::min<size_t>(state.merge_inputs, state.run_count);
        size_t first = state.run_count - count;
        if (!all)
        {
            // Defer small final sets without repeatedly folding a growing
            // suffix into each incoming batch. The base rank gate bounds
            // copy levels; forced metadata/space/final-reader pressure can
            // select an unbalanced adjacent set only when it is necessary.
            auto rank = [&](uint16_t members)
            {
                unsigned const bits = std::bit_width(uint32_t(members - 1));
                return state.merge_inputs == 4 ? (bits + 1) / 2 : bits;
            };
            bool ready = true;
            for (size_t i = first; i + 1 < state.run_count; ++i)
                if (rank(state.runs[i].count) > rank(state.runs[state.run_count - 1].count)) ready = false;
            if (!ready) break;
        }
        else
        {
            uint64_t least = UINT64_MAX;
            for (size_t begin = 0; begin + count <= state.run_count; ++begin)
            {
                uint64_t bytes = 0;
                for (size_t i = begin; i < begin + count; ++i) bytes = grow_add(bytes, state.runs[i].bytes);
                if (bytes <= least) { least = bytes; first = begin; }
            }
        }
        merge(first, count);
    }
}
inline void MergeEngine::pack_chunks(Inputs& input, MergeState::Writer& output, std::span<uint64_t> sizes)
{
    config().check_running();
    auto& state = arena_.state; uint64_t committed = state.output.bytes;
    uint64_t checkpoint_ns = 0, checkpoints = 0;
    std::array<uint64_t, MergeState::max_runs> completed{};
    for (size_t i = 0; i < input.size(); ++i) completed[i] = state.runs[i].consumed;
    auto checkpoint = [&](uint64_t next, std::span<uint64_t const> frontiers, std::string const& label)
    {
        output.flush();
        for (size_t i = 0; i < input.size(); ++i) arena_.release(state.runs[i], frontiers[i]);
        state.chunk = next;
        auto const begin = std::chrono::steady_clock::now(); arena_.checkpoint(label);
        checkpoint_ns = grow_add(checkpoint_ns, elapsed_ns(begin)); ++checkpoints;
        for (size_t i = 0; i < input.size(); ++i) input.reader(i).evict(frontiers[i]);
        output.evict(); committed = state.output.bytes;
        config().check_running();
    };
    auto emit = [&](uint64_t chunk, std::span<uint8_t const> bytes, std::span<uint64_t const> frontiers)
    {
        if (chunk > state.chunk)
        {
            uint64_t const end = grow_add(output.position(), bytes.size());
            uint64_t placement_bytes = 0;
            for (uint64_t boundary = MergeState::align(output.position(), state.block_bytes);
                 boundary < end; boundary += state.block_bytes)
            {
                uint32_t const block = uint32_t(boundary / state.block_bytes);
                for (size_t i = 0; i < input.size(); ++i)
                    if (arena_.releasing(state.runs[i], completed[i], block))
                    { placement_bytes += merge_placement_bytes(state.block_bytes, boundary, end, minimum_final_bytes_); break; }
            }
            // Only already emitted chunks may free a destination. Publish
            // their output and consumed frontiers before the next allocation;
            // preread or encoded-but-unwritten input stays protected.
            if (checkpoints && prefer_merge_placement(arena_.stats, placement_bytes, state.slot_bytes,
                    static_cast<long double>(checkpoint_ns) / checkpoints))
                checkpoint(chunk, std::span(completed).first(input.size()), "merge-pack-placement");
        }
        sizes[size_t(chunk)] = bytes.size(); output.write(bytes);
        std::copy(frontiers.begin(), frontiers.end(), completed.begin());
        if (output.position() - committed >= transaction_bytes() || chunk + 1 == chunks())
            checkpoint(chunk + 1, frontiers, "merge-pack");
    };
    if (config().workers == 1)
    {
        arena_.stats.packing_workers = std::max<uint64_t>(arena_.stats.packing_workers, 1);
        ChunkCodec codec(k(), config().codec_bytes, config().cpu_scheduler); Gather gathered(*this);
        std::array<uint64_t, MergeState::max_runs> frontiers{};
        for (uint64_t chunk = state.chunk; chunk < chunks(); ++chunk)
        {
            config().check_running();
            (void)gathered.read(*this, input, chunk);
            codec.prepare(ChunkCodec::estimate_work(k(), gathered.payload.size(), 0));
            auto const begin = std::chrono::steady_clock::now();
            auto bytes = codec.join(chunk, gathered.contributions);
            arena_.stats.codec_ns += elapsed_ns(begin); ++arena_.stats.encode_chunks;
            for (size_t i = 0; i < input.size(); ++i) frontiers[i] = input.reader(i).position();
            emit(chunk, bytes, std::span(frontiers).first(input.size()));
        }
        return;
    }
    // Workers validate and join the encoded contributions in contiguous
    // partitions. Ordered reads can prepare the next bounded encoded window.
    uint64_t const encoding_budget = config().codec_bytes;
    uint64_t const baseline = ChunkCodec::estimate_base();
    uint64_t const metadata_per_task = sizeof(ChunkInput) + 32 + uint64_t(state.members) * sizeof(EncodedChunk) +
        input.size() * sizeof(uint64_t);
    size_t const limit = size_t(std::max<uint64_t>(1, std::min({chunks(), uint64_t(65536), config().window_bytes / metadata_per_task})));
    struct Window
    {
        Gather gathered;
        std::vector<ChunkInput> tasks;
        std::vector<uint64_t> costs, frontiers;
        uint64_t first = 0, next = 0;
        Window(MergeEngine const& engine, size_t limit, uint16_t members, size_t runs) : gathered(engine)
        {
            gathered.contributions.reserve(size_t(MergeState::multiply(limit, members)));
            tasks.reserve(limit); costs.reserve(limit + 1); frontiers.reserve(limit * runs);
        }
    };
    auto prepare = [&](Window& window, uint64_t first)
    {
        config().check_running();
        auto& gathered = window.gathered; auto& tasks = window.tasks; auto& costs = window.costs;
        tasks.clear(); costs.clear(); window.frontiers.clear(); gathered.payload.clear(); gathered.contributions.clear(); costs.push_back(0);
        window.first = first; window.next = first; uint64_t source_bytes = 0;
        while (window.next < chunks() && tasks.size() < limit)
        {
            uint64_t const chunk = window.next;
            uint64_t const bytes = input.chunk_bytes(chunk);
            uint64_t const cost = ChunkCodec::estimate_work(k(), bytes, 0) - baseline;
            if (bytes > config().window_bytes - source_bytes ||
                cost > encoding_budget - baseline - costs.back())
            { require(!tasks.empty(), "merged packing chunk exceeds window admission"); break; }
            size_t const contribution = gathered.contributions.size();
            require(gathered.read(*this, input, chunk, true) == bytes, "merged packing window coverage changed");
            costs.push_back(grow_add(costs.back(), cost));
            for (size_t i = 0; i < input.size(); ++i) window.frontiers.push_back(input.reader(i).position());
            tasks.push_back({chunk, {}, 0, {}, std::span<EncodedChunk const>(gathered.contributions).subspan(contribution)});
            source_bytes += bytes; ++window.next;
        }
    };
    auto current = std::make_unique<Window>(*this, limit, state.members, input.size());
    auto next = config().io_overlap ? std::make_unique<Window>(*this, limit, state.members, input.size()) : nullptr;
    std::unique_ptr<ParallelEncoder> encoder;
    prepare(*current, state.chunk);
    while (!current->tasks.empty())
    {
        config().check_running();
        auto const first = current->first; auto const& tasks = current->tasks;
        unsigned const workers = encoding_workers(current->costs, encoding_budget);
        arena_.stats.packing_workers = std::max<uint64_t>(arena_.stats.packing_workers, workers);
        if (!encoder || encoder->workers() != workers)
        { encoder.reset(); encoder = std::make_unique<ParallelEncoder>(k(), workers, encoding_budget, config().cpu_scheduler); }
        auto encode = [&]
        {
            auto const begin = std::chrono::steady_clock::now();
            auto blocks = encoder->encode(tasks, sizes.subspan(size_t(first), tasks.size()));
            arena_.stats.codec_ns += elapsed_ns(begin); return blocks;
        };
        std::span<EncodedBlock const> blocks;
        if (next && current->next < chunks())
        {
            // The future drains before either window can be reused, including
            // exception unwinding. Only this writer reads or reclaims inputs.
            auto pending = std::async(std::launch::async, encode);
            if (pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                ++arena_.stats.packing_overlap_windows;
            prepare(*next, current->next); blocks = pending.get();
        }
        else blocks = encode();
        arena_.stats.encode_chunks += tasks.size(); size_t ordered = 0;
        for (auto const& block : blocks)
        {
            require(block.begin == ordered, "unordered merged codec output"); size_t offset = 0;
            for (size_t task = block.begin; task < block.end; ++task)
            {
                size_t const bytes = size_t(sizes[size_t(first) + task]);
                require(offset <= block.bytes.size() && bytes <= block.bytes.size() - offset, "truncated merged codec output");
                emit(first + task, block.bytes.subspan(offset, bytes),
                    std::span(current->frontiers).subspan(task * input.size(), input.size())); offset += bytes;
            }
            require(offset == block.bytes.size(), "unexpected merged codec trailing output"); ordered = block.end;
        }
        require(ordered == tasks.size(), "incomplete merged codec output");
        uint64_t const following = current->next;
        if (following == chunks()) break;
        if (next) current.swap(next);
        else prepare(*current, following);
    }
}
inline void MergeEngine::pack()
{
    config().check_running();
    PhaseTimer timing(arena_.stats, WorkPhase::pack, config().progress);
    auto& state = arena_.state;
    if (state.phase == Phase::stage)
    {
        require(state.run_count && state.run_count <= MergeState::max_runs, "incomplete merged packing input");
        state.phase = Phase::pack; state.chunk = 0; state.output = {};
        state.output.count = state.members;
        MergeState::Writer output(arena_, state.output, config().window_bytes, true);
        Header header = state.target; header.chunk_index_offset = 0;
        output.value(header); output.write(arena_.identity.memo); output.flush();
        arena_.discard(state.summary); arena_.checkpoint("merge-pack-start");
    }
    require(state.phase == Phase::pack, "invalid merged packing phase");
    std::vector<uint64_t> sizes(size_t(chunks()), 0);
    if (state.chunk)
    {
        ChunkCodec codec(k(), config().codec_bytes, config().cpu_scheduler);
        // Only restart pays for rescanning its already durable canonical
        // prefix. Its consumed input blocks may have been reused long ago.
        MergeState::Reader prefix(arena_, state.output, read_budget(), header_bytes());
        std::vector<uint8_t> window(size_t(config().window_bytes));
        uint64_t begin = UINT64_MAX; size_t available = 0;
        for (uint64_t chunk = 0; chunk < state.chunk; ++chunk)
        {
            if (begin == UINT64_MAX || prefix.position() - begin > available ||
                available - size_t(prefix.position() - begin) < std::min(state.chunk_bound, prefix.remaining()))
            {
                begin = prefix.position(); available = size_t(std::min<uint64_t>(window.size(), prefix.remaining()));
                prefix.copy(begin, std::span(window).first(available));
            }
            size_t const offset = size_t(prefix.position() - begin);
            codec.prepare(ChunkCodec::estimate_work(k(), std::min<uint64_t>(state.chunk_bound, available - offset), 0));
            size_t const bytes = codec.scan(chunk, std::span(window).subspan(offset, available - offset), state.members);
            sizes[size_t(chunk)] = bytes; prefix.skip(bytes); prefix.evict_read();
        }
        require(!prefix.remaining(), "saved canonical merge prefix changed");
    }
    Inputs input(*this, std::span(state.runs).first(state.run_count), state.chunk);
    MergeState::Writer output(arena_, state.output, config().window_bytes, true);
    pack_chunks(input, output, sizes);
    input.finish(); state.target.chunk_index_offset = state.output.bytes;
    BitWriter index; gsz_encode(sizes, state.members, index); output.write(index.asBytes()); output.flush();
    state.final_bytes = state.output.bytes;
    require(state.output.head < state.block_limit && sizeof(Header) <= state.block_bytes, "missing final merged header");
    arena_.value(uint64_t(state.output.head) * state.block_bytes, state.target);
    arena_.sync(); output.evict(true); grow_failpoint("merge-index-flushed");
    auto blocks = arena_.blocks(state.output); state.layout_blocks = uint32_t(blocks.size()); state.layout_done = 0;
    std::fill(arena_.map.begin(), arena_.map.end(), MergeState::free_block);
    std::copy(blocks.begin(), blocks.end(), arena_.map.begin());
    state.phase = Phase::layout; arena_.relocate_checkpoint();
}
inline void MergeEngine::arrange()
{
    config().check_running();
    PhaseTimer timing(arena_.stats, WorkPhase::layout, config().progress);
    auto& state = arena_.state;
    require(state.phase == Phase::layout, "invalid merged block arrangement");
    std::vector<uint32_t> owner(state.block_limit, MergeState::free_block);
    for (uint32_t i = 0; i < state.layout_blocks; ++i)
    { require(arena_.map[i] < owner.size() && owner[arena_.map[i]] == MergeState::free_block, "duplicate final merge block"); owner[arena_.map[i]] = i; }
    std::vector<uint8_t> buffer(size_t(std::min(config().window_bytes, state.block_bytes)));
    std::vector<uint32_t> ready, cycles, spares;
    ready.reserve(state.layout_blocks); cycles.reserve(state.layout_blocks / 2);
    spares.reserve(state.block_limit - state.layout_blocks);
    std::vector<uint8_t> visited(state.layout_blocks, 0);
    // The injective map consists of disjoint open chains and cycles. Collect
    // each chain's vacant destination once, rather than rescanning the map
    // after every commit. Placed blocks form cycles of length one.
    auto visit = [&](uint32_t block)
    {
        while (block < state.layout_blocks && !visited[block])
        { visited[block] = 1; block = arena_.map[block]; }
    };
    for (uint32_t block = 0; block < state.layout_blocks; ++block)
        if (owner[block] == MergeState::free_block) { ready.push_back(block); visit(block); }
    for (uint32_t block = 0; block < state.layout_blocks; ++block)
        if (!visited[block])
        {
            if (arena_.map[block] != block) cycles.push_back(block);
            visit(block);
        }
    for (uint32_t block = state.block_limit; block > state.layout_blocks;)
    { --block; if (owner[block] == MergeState::free_block) spares.push_back(block); }
    std::vector<uint32_t> pending;
    pending.reserve(size_t(std::min<uint64_t>(state.layout_blocks, transaction_bytes() / state.block_bytes + 1)));
    uint64_t pending_bytes = 0;
    auto commit = [&]
    {
        if (pending.empty()) return;
        while (state.layout_done < state.layout_blocks && arena_.map[size_t(state.layout_done)] == state.layout_done)
            ++state.layout_done;
        arena_.checkpoint("merge-layout-move"); ++arena_.stats.layout_batches;
        // No destination in this batch can reuse one of these sources. They
        // remain occupied until both their replacement and the new map are
        // durable, including when several independent moves share a commit.
        for (uint32_t source : pending)
        {
            require(owner[source] != MergeState::free_block && arena_.map[owner[source]] != source,
                    "layout batch lost its protected source");
            owner[source] = MergeState::free_block;
            if (source < state.layout_blocks) ready.push_back(source);
            else spares.push_back(source);
            arena_.evict(uint64_t(source) * state.block_bytes, state.block_bytes);
        }
        pending.clear(); pending_bytes = 0;
    };
    auto move = [&](uint32_t logical, uint32_t target)
    {
        config().check_running();
        uint32_t const source = arena_.map[logical];
        require(source != target && target < state.block_limit && owner[target] == MergeState::free_block,
                "final merge move would overwrite a live block");
        uint64_t const bytes = std::min(state.block_bytes, state.final_bytes - uint64_t(logical) * state.block_bytes);
        if (!pending.empty() && bytes > transaction_bytes() - std::min(pending_bytes, transaction_bytes())) commit();
        for (uint64_t offset = 0; offset < bytes;)
        {
            size_t const count = size_t(std::min<uint64_t>(buffer.size(), bytes - offset));
            auto part = std::span(buffer).first(count); arena_.read(uint64_t(source) * state.block_bytes + offset, part);
            if (!logical && !offset)
            {
                require(count >= sizeof(Header), "merge layout window cannot hold header");
                Header marker = state.target; marker.version = MergeState::construction_version; marker.chunk_index_offset = state.anchor;
                std::memcpy(part.data(), &marker, sizeof(marker));
            }
            size_t const half = count / 2;
            arena_.write(uint64_t(target) * state.block_bytes + offset, part.first(half)); grow_failpoint("merge-layout-partial");
            arena_.write(uint64_t(target) * state.block_bytes + offset + half, part.subspan(half)); grow_failpoint("merge-layout-written");
            offset += count;
        }
        arena_.map[logical] = target;
        owner[target] = logical; pending.push_back(source); pending_bytes += bytes;
        ++arena_.stats.layout_moves; arena_.stats.layout_bytes += bytes;
    };
    size_t next = 0, cycle = 0;
    while (next < ready.size() || cycle < cycles.size())
    {
        // Independent chains advance together. Their next destinations only
        // enter ready after the checkpoint releases the protected sources.
        while (next < ready.size()) { uint32_t const target = ready[next++]; move(target, target); }
        // Use only existing tail vacancies as durable cycle spares. If all
        // tail slots hold sources, open chains release them first. No extra
        // capacity is reserved and each cycle still needs one extra copy.
        while (cycle < cycles.size() && !spares.empty())
        {
            uint32_t const spare = spares.back(); spares.pop_back();
            move(cycles[cycle++], spare);
        }
        require(!pending.empty(), "missing reserved merge layout block");
        commit();
    }
    while (state.layout_done < state.layout_blocks && arena_.map[size_t(state.layout_done)] == state.layout_done)
        ++state.layout_done;
    require(state.layout_done == state.layout_blocks, "incomplete merged block arrangement");
    arena_.checkpoint("merge-layout-complete"); arena_.publish();
}
inline Result MergeEngine::finish()
{
    config().check_running();
    require(confirmed_, "merge finalization requires confirmed identity"); arena_.check_path();
    validate_remaining(); arena_.restore_extent();
    auto& state = arena_.state;
    if (!arena_.active)
    {
        config().check_running();
        return {arena_.identity.bytes, arena_.identity.bytes, 0, arena_.identity.header.group_size};
    }
    if (state.members == state.original.group_size && state.phase == Phase::stage)
    {
        if (state.original_bytes) { arena_.value(0, state.original); arena_.sync(); }
        arena_.trim(state.original_bytes); arena_.sync(); arena_.active = false; arena_.identity.pending = false;
        arena_.identity.pending_peak_bytes = 0;
        config().check_running();
        return {state.original_bytes, state.storage_bytes, state.storage_bytes - state.original_bytes, state.members};
    }
    if (!state.finishing) { state.finishing = true; arena_.checkpoint("merge-finishing"); }
    if (state.phase == Phase::stage || state.phase == Phase::merge) balance(packing_inputs(config()));
    if (state.phase == Phase::stage || state.phase == Phase::pack) pack();
    if (state.phase == Phase::layout) arrange();
    remember();
    config().check_running();
    return {state.final_bytes, state.storage_bytes, state.storage_bytes - state.final_bytes, state.members};
}
} // namespace group_grow
