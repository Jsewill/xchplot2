// Bounded-batch external assembly with durable progress and optional raw reclamation.
#include "host/GroupCheckpoint.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#ifdef __linux__
#include <sys/resource.h>
#endif

namespace
{
using namespace group_tools;
using namespace stream_state;
namespace fs = std::filesystem;
constexpr uint64_t mib = 1024 * 1024;

struct Budget
{
    uint64_t limit, fixed;
    Stats& stats;
    void room(uint64_t live) const
    {
        require(fixed <= limit && live <= limit - fixed, "stream assembly RAM budget exceeded");
        stats.peak_modeled_bytes = std::max(stats.peak_modeled_bytes, fixed + live);
    }
};
void create_output(std::ofstream& out, fs::path const& path)
{
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path, std::ios::binary);
    fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write);
}
void append_output(std::ofstream& out, fs::path const& path, uint64_t bytes)
{
    if (!fs::exists(path))
    {
        require(bytes == 0, "committed output is missing");
        create_output(out, path);
    }
    else
    {
        require(stamp(path).bytes == bytes, "output was not rolled back to its checkpoint");
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.open(path, std::ios::binary | std::ios::app);
    }
}
struct Job
{
    PlotGroupParams params;
    std::vector<uint8_t> memo;
    std::vector<std::string> members;
};
Job load_job(std::string const& path, Budget& budget)
{
    uint64_t const bytes = fs::file_size(path);
    require(bytes < budget.limit / 16, "group input manifest exceeds RAM budget");
    auto in = input(path);
    int k, strength, meta;
    std::string group, memo, tail, base, extra;
    in >> k >> strength >> meta >> group >> memo;
    std::getline(in, tail);
    std::istringstream range(tail);
    if (range >> base)
        require(number(base.c_str()) == 0 && !(range >> extra),
                "upstream groups must start at member index zero");
    require(k >= 18 && k <= 28 && !(k & 1) && meta >= 0 && meta <= 255 &&
                strength >= 2 && strength <= k - 3, "invalid group parameters");
    Job job{PlotGroupParams(PlotGroupId(group), uint8_t(k), uint8_t(strength), uint8_t(meta)),
            hex(memo), {}};
    require(job.memo.size() <= 255, "memo too long");
    in.exceptions(std::ios::badbit);
    std::string name;
    while (in >> std::quoted(name))
    {
        require(job.members.size() < 65535 && !name.empty() && name.size() <= 4096,
                "too many members or invalid path");
        job.members.push_back(fs::absolute(name).lexically_normal().string());
    }
    require(in.eof() && !job.members.empty(), "invalid or empty member list");
    // Manifest strings/run descriptors, small stream buffers, FSE tables and
    // streamed-index workspace. This is a buffer model, not an RSS guarantee.
    uint64_t paths = 0;
    for (auto const& path : job.members) paths += path.size();
    budget.fixed = bytes * 4 + paths * 4 + job.members.size() * 1024 + mib;
    budget.room(0);
    return job;
}

struct Raw
{
    std::ifstream in;
    std::vector<uint64_t> offsets, fragments;
    uint8_t k;

    Raw(Job const& job, size_t member) : in(input(job.members[member])), k(job.params.get_k())
    {
        auto const size = fs::file_size(job.members[member]);
        require(read<std::array<char, 4>>(in) == std::array<char, 4>{'p', 'o', 's', '2'} &&
                    read<uint8_t>(in) == PlotFile::FORMAT_VERSION, "expected PoS2 1.0 raw v2 member");
        require(PlotGroupId(read<std::array<uint8_t, 32>>(in)) == job.params.get_plot_group_id(),
                "raw member group ID mismatch");
        require(read<uint8_t>(in) == k && read<uint8_t>(in) == job.params.get_strength() &&
                    read<uint16_t>(in) == member && read<uint8_t>(in) == job.params.get_meta_group(),
                "raw member parameters/index mismatch");
        require(read<uint8_t>(in) == job.memo.size(), "raw memo length mismatch");
        std::vector<uint8_t> memo(job.memo.size());
        in.read(reinterpret_cast<char*>(memo.data()), memo.size());
        require(memo == job.memo, "raw member memo mismatch");
        uint64_t const count = read<uint64_t>(in);
        require(count == (1ull << (k - PlotFile::CHUNK_SPAN_RANGE_BITS)),
                "raw member does not cover every group range");
        require(size >= 51 + memo.size() + count * 8, "truncated raw index");
        offsets.resize(count);
        in.read(reinterpret_cast<char*>(offsets.data()), count * 8);
        uint64_t end = 51 + memo.size() + count * 8;
        for (uint64_t offset : offsets)
        {
            require(offset == end && offset <= size && size - offset >= 8, "invalid raw chunk offset");
            in.seekg(offset);
            uint64_t const bytes = read<uint64_t>(in);
            require(bytes >= 12 && bytes <= size - offset - 8, "invalid raw chunk size");
            end = offset + 8 + bytes;
        }
        require(end == size, "unexpected raw trailing data");
    }
    void load(uint64_t chunk, Budget const& budget, uint64_t& live, Stats& stats)
    {
        live -= fragments.capacity() * 8;
        std::vector<uint64_t>().swap(fragments);
        in.seekg(offsets.at(chunk));
        uint64_t const bytes = read<uint64_t>(in);
        budget.room(live + bytes);
        uint32_t const count = read<uint32_t>(in);
        uint32_t const ans = read<uint32_t>(in);
        uint32_t const stubs = read<uint32_t>(in);
        require(count > 0 && uint64_t(ans) + stubs + 12 == bytes &&
                    stubs == (uint64_t(count) * (k - 2) + 7) / 8, "invalid raw chunk lengths");
        budget.room(live + bytes + uint64_t(count) * 32);
        in.seekg(offsets[chunk] + 8);
        std::vector<uint8_t> compressed(bytes);
        in.read(reinterpret_cast<char*>(compressed.data()), bytes);
        stats.raw_payload_read_bytes += bytes;
        uint64_t const range = 1ull << (k + PlotFile::CHUNK_SPAN_RANGE_BITS);
        fragments = ChunkCompressor::decompressProofFragments(compressed, chunk * range, k - 2);
        require(fragments.size() == count && std::is_sorted(fragments.begin(), fragments.end()) &&
                    fragments.front() >= chunk * range && fragments.back() < (chunk + 1) * range,
                "invalid raw fragment order/range");
        live += fragments.capacity() * 8;
        budget.room(live);
    }
};

std::string job_identity(Job const& job)
{
    std::ostringstream out;
    out << job.params.get_k() << ' ' << int(job.params.get_strength()) << ' '
        << int(job.params.get_meta_group()) << ' ' << job.params.get_plot_group_id().to_string()
        << ' ' << hex(job.memo) << '\n';
    for (auto const& path : job.members) out << std::quoted(path) << '\n';
    return digest(out.str());
}
struct Checkpoints
{
    fs::path work;
    State& state;
    Job const& job;
    std::string identity;
    size_t raw_cleaned = 0;
    uint32_t cleanup_id = UINT32_MAX;
    std::vector<uint64_t> cleaned_segments;

    void reclaim()
    {
        auto& stats = state.stats;
        std::set<fs::path> directories;
        if (state.config.consume)
            while (raw_cleaned < state.staged)
            {
                auto const& path = job.members[raw_cleaned];
                if (fs::exists(path))
                {
                    require(same_output(state.raw[raw_cleaned], stamp(path)),
                            "raw member changed before reclamation");
                    require(fs::remove(path), "cannot reclaim raw member");
                    stats.remaining_raw_bytes -= state.raw[raw_cleaned].bytes;
                    stats.reclaimed_raw_bytes += state.raw[raw_cleaned].bytes;
                    directories.insert(fs::path(path).parent_path());
                    failpoint("raw-after-unlink");
                }
                ++raw_cleaned;
            }
        if (state.active && !state.config.keep)
        {
            auto const& active = state.operation;
            if (cleanup_id != active.output.id)
            {
                cleanup_id = active.output.id;
                cleaned_segments.assign(active.inputs.size(), 0);
            }
            for (size_t i = 0; i < active.inputs.size(); ++i)
            {
                auto const& run = active.inputs[i];
                uint64_t const position = active.positions[i];
                uint64_t const end = position / state.config.segment +
                    (position == run.bytes && position % state.config.segment != 0);
                while (cleaned_segments[i] < end)
                {
                    auto const path = segment_path(work, run, cleaned_segments[i]);
                    if (fs::exists(path))
                    {
                        stats.remove(path);
                        failpoint("segment-after-unlink");
                    }
                    ++cleaned_segments[i];
                }
            }
            directories.insert(work);
        }
        if (state.phase >= Phase::ready && fs::exists(work / "index.tmp"))
        {
            stats.remove(work / "index.tmp");
            directories.insert(work);
        }
        for (auto const& path : directories) sync_path(path, true);
    }
    void commit(std::string const& label)
    {
        save_state(work, state, identity, label);
        reclaim();
    }
    void recover()
    {
        auto const segment = state.config.segment;
        auto& stats = state.stats;
        stats.remaining_raw_bytes = 0;
        for (size_t i = 0; i < job.members.size(); ++i)
        {
            if (fs::exists(job.members[i]))
            {
                require(same_output(state.raw[i], stamp(job.members[i])),
                        "raw member changed since checkpoint");
                stats.remaining_raw_bytes += state.raw[i].bytes;
            }
            else require(state.config.consume && i < state.staged, "required raw member is missing");
        }
        stats.reclaimed_raw_bytes = stats.input_bytes - stats.remaining_raw_bytes;
        auto validate_complete = [&](Run const& run) { validate_run(work, run, 0, segment); };
        if (state.phase == Phase::stage)
            for (auto run : state.runs) validate_complete(run);
        if (state.phase == Phase::merge)
        {
            for (auto run : state.next_runs) validate_complete(run);
            size_t const first = state.merge_first + (state.active ? state.operation.inputs.size() : 0);
            require(first <= state.runs.size(), "checkpoint merge inputs exceed run list");
            for (size_t i = first; i < state.runs.size(); ++i) validate_complete(state.runs[i]);
        }
        if (state.active && state.phase < Phase::ready)
        {
            auto const& active = state.operation;
            require(active.output.id < state.next_id && active.output.count > 0 &&
                        active.output.first + uint64_t(active.output.count) <= job.members.size(),
                    "invalid checkpoint output range");
            size_t const first = state.phase == Phase::merge ? state.merge_first : 0;
            require(first + active.inputs.size() <= state.runs.size() &&
                std::equal(active.inputs.begin(), active.inputs.end(), state.runs.begin() + first),
                "checkpoint inputs differ from run list");
            for (size_t i = 0; i < active.inputs.size(); ++i)
            {
                require(active.inputs[i].id != active.output.id, "checkpoint output aliases input");
                validate_run(work, active.inputs[i], active.positions[i], segment);
            }
            if (state.phase == Phase::merge) validate_run_prefix(work, active.output, segment);
            else
            {
                validate_prefix(work / "group.gplot.partial", active.output.bytes);
                validate_prefix(work / "index.tmp", active.index_bytes);
                require(!fs::exists(work / "group.gplot"), "unexpected published group");
            }
        }
        if (state.phase >= Phase::ready)
        {
            bool found = false;
            for (auto const* name : {"group.gplot.partial", "group.gplot"})
                if (fs::exists(work / name))
                {
                    require(same_output(state.completed, stamp(work / name)), "completed group was changed");
                    auto group = PlotGroupFile::open((work / name).string());
                    auto const& info = group.getInfo();
                    require(info.group_id == job.params.get_plot_group_id() && info.k == job.params.get_k() &&
                        info.strength == job.params.get_strength() && info.meta_group == job.params.get_meta_group() &&
                        info.group_size == job.members.size() && group.readMemo() == job.memo,
                        "completed group identity mismatch");
                    found = true;
                }
            require(found, "completed group is missing");
        }
        // Validate every required prefix before truncating tails or deleting files.
        stats.work_bytes = work_file_bytes(work);
        stats.peak();
        if (state.phase == Phase::stage)
            rollback_run(work, Run{state.next_id}, segment, stats);
        if (state.active && state.phase == Phase::merge)
            rollback_run(work, state.operation.output, segment, stats);
        if (state.phase == Phase::final)
        {
            truncate_prefix(work / "group.gplot.partial", state.operation.output.bytes);
            truncate_prefix(work / "index.tmp", state.operation.index_bytes);
            stats.work_bytes = work_file_bytes(work);
        }
        for (auto const& entry : fs::directory_iterator(work))
            if (entry.path().filename().string().starts_with("checkpoint.partial."))
            {
                (void)stamp(entry.path());
                fs::remove(entry.path());
            }
        reclaim();
    }
};

// Canonical chunk payloads are also useful as independently compressed batch
// contributions. They are construction records, not independently farmable groups.
struct Encoder
{
    std::vector<uint8_t> high;
    BitWriter bits;
    uint64_t previous;
    uint8_t k;
    Encoder(uint64_t start, uint8_t k) : previous(start), k(k) {}
    void append(uint64_t value)
    {
        require(value >= previous, "unordered group fragment");
        append_delta(value - previous, k, high, bits);
        previous = value;
    }
    std::vector<uint8_t> finish(FSE_CTable const* table)
    {
        std::vector<uint8_t> ans(POS2_FSE_compressBound(high.size()));
        size_t const size = POS2_FSE_compress_usingCTable(ans.data(), ans.size(), high.data(), high.size(), table);
        require(size > 0 && !POS2_FSE_isError(size), "cannot compress grouped region");
        std::vector<uint8_t> result;
        auto const low = bits.asBytes();
        result.reserve(10 + size + low.size());
        uint64_t leb = size;
        do
        {
            result.push_back(uint8_t((leb & 0x7f) | (leb > 0x7f ? 0x80 : 0)));
            leb >>= 7;
        } while (leb);
        result.insert(result.end(), ans.begin(), ans.begin() + size);
        result.insert(result.end(), low.begin(), low.end());
        return result;
    }
};

class RunWriter
{
    fs::path work;
    Run& run;
    uint64_t segment_bytes;
    Stats& stats;
    std::vector<char> buffer;
    std::ofstream out;
public:
    RunWriter(fs::path const& work, Run& run, uint64_t segment_bytes, uint64_t buffer_bytes, Stats& stats)
        : work(work), run(run), segment_bytes(segment_bytes), stats(stats), buffer(buffer_bytes) {}
    void write(void const* data, uint64_t bytes)
    {
        auto p = static_cast<char const*>(data);
        while (bytes)
        {
            if (!out.is_open())
            {
                out.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
                append_output(out, segment_path(work, run, run.bytes / segment_bytes), run.bytes % segment_bytes);
            }
            uint64_t const n = std::min(bytes, segment_bytes - run.bytes % segment_bytes);
            out.write(p, n);
            run.bytes += n;
            stats.add(n);
            p += n;
            bytes -= n;
            if (run.bytes % segment_bytes == 0)
                close();
        }
    }
    void sync()
    {
        if (out.is_open())
        {
            out.flush();
            sync_path(segment_path(work, run, (run.bytes - 1) / segment_bytes));
        }
    }
    void close()
    {
        if (out.is_open())
        {
            sync();
            out.close();
        }
    }
};
class RunReader
{
    fs::path work;
    Run run;
    uint64_t segment_bytes;
    Stats& stats;
    std::vector<char> buffer;
    std::ifstream in;
    uint64_t pos;
public:
    RunReader(fs::path const& work, Run run, uint64_t segment_bytes, uint64_t buffer_bytes,
              Stats& stats, uint64_t position)
        : work(work), run(run), segment_bytes(segment_bytes), stats(stats), buffer(buffer_bytes), pos(position) {}
    void read(void* data, uint64_t bytes)
    {
        require(bytes <= run.bytes - pos, "truncated temporary run");
        auto p = static_cast<char*>(data);
        while (bytes)
        {
            auto const path = segment_path(work, run, pos / segment_bytes);
            if (!in.is_open())
            {
                in.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
                in.open(path, std::ios::binary);
                in.exceptions(std::ios::badbit | std::ios::failbit);
                in.seekg(pos % segment_bytes);
            }
            uint64_t const n = std::min({bytes, segment_bytes - pos % segment_bytes, run.bytes - pos});
            in.read(p, n);
            p += n;
            bytes -= n;
            pos += n;
            stats.run_read_bytes += n;
            if (pos % segment_bytes == 0 || pos == run.bytes)
            {
                in.close();
            }
        }
    }
    void finish() { require(pos == run.bytes, "unexpected temporary trailing data"); }
    uint64_t position() const { return pos; }
};
// Private same-host scratch layout. Records may cross segment boundaries.
struct Record
{
    uint16_t first, count;
    uint32_t bytes;
};
static_assert(sizeof(Record) == 8);
Record next_record(RunReader& in, uint32_t first, uint32_t end)
{
    Record record{};
    in.read(&record, sizeof(record));
    require(record.first == first && record.count > 0 && first + record.count <= end && record.bytes > 1,
            "invalid temporary member range/record size");
    return record;
}

Run stage(Job const& job, Run run, fs::path const& work, uint64_t segment, uint64_t buffer,
          Budget const& budget, Stats& stats, FSE_CTable const* table)
{
    uint8_t const k = job.params.get_k();
    uint64_t const raw_chunks = 1ull << (k - PlotFile::CHUNK_SPAN_RANGE_BITS);
    uint64_t live = buffer + run.count * (raw_chunks * 8 + 16384);
    budget.room(live);
    std::vector<std::unique_ptr<Raw>> members;
    for (uint32_t member = run.first; member < run.first + run.count; ++member)
        members.push_back(std::make_unique<Raw>(job, member));
    stats.max_active_members = std::max<uint64_t>(stats.max_active_members, run.count);
    RunWriter out(work, run, segment, buffer, stats);
    uint64_t const per_raw = 1ull << (PlotFile::CHUNK_SPAN_RANGE_BITS - PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (uint64_t coarse = 0; coarse < raw_chunks; ++coarse)
    {
        for (auto& member : members)
            member->load(coarse, budget, live, stats);
        for (uint64_t fine = 0; fine < per_raw; ++fine)
        {
            uint64_t const start = (coarse * per_raw + fine) * range;
            Encoder encoder(start, k);
            for (size_t m = 0; m < members.size(); ++m)
            {
                auto const& fragments = members[m]->fragments;
                auto first = std::lower_bound(fragments.begin(), fragments.end(), start);
                auto last = std::lower_bound(first, fragments.end(), start + range);
                require(first != last, "upstream grouped format requires a nonempty region for every member");
                budget.room(live + (encoder.high.size() + (last - first)) * 64);
                for (auto it = first; it != last; ++it)
                    if (it == first || *it != *(it - 1))
                        encoder.append(*it + m * range);
            }
            budget.room(live + encoder.high.size() * 64);
            auto payload = encoder.finish(table);
            require(payload.size() <= UINT32_MAX, "temporary record too large");
            Record record{uint16_t(run.first), uint16_t(run.count), uint32_t(payload.size())};
            out.write(&record, sizeof(record));
            out.write(payload.data(), payload.size());
        }
    }
    out.close();
    stats.staged_bytes += run.bytes;
    return run;
}

std::vector<std::unique_ptr<RunReader>> readers(Checkpoints& checkpoint)
{
    auto& state = checkpoint.state;
    auto const& config = state.config;
    auto const& active = state.operation;
    std::vector<std::unique_ptr<RunReader>> result;
    for (size_t i = 0; i < active.inputs.size(); ++i)
        result.push_back(std::make_unique<RunReader>(checkpoint.work, active.inputs[i], config.segment,
            config.buffer, state.stats, active.positions[i]));
    state.stats.max_active_runs = std::max<uint64_t>(state.stats.max_active_runs, result.size());
    return result;
}
void capture_positions(Active& active, std::vector<std::unique_ptr<RunReader>> const& inputs)
{
    for (size_t i = 0; i < inputs.size(); ++i) active.positions[i] = inputs[i]->position();
}
void merge(Checkpoints& checkpoint, uint64_t chunks, Budget const& budget)
{
    auto& state = checkpoint.state;
    auto& active = state.operation;
    auto const& config = state.config;
    budget.room((active.inputs.size() + 2) * config.buffer);
    RunWriter out(checkpoint.work, active.output, config.segment, config.buffer, state.stats);
    auto inputs = readers(checkpoint);
    std::vector<char> copy(config.buffer);
    uint64_t committed = active.output.bytes;
    while (active.chunk < chunks)
    {
        uint64_t const begin = active.output.bytes;
        for (size_t r = 0; r < active.inputs.size(); ++r)
        {
            auto const& run = active.inputs[r];
            uint32_t const end = run.first + run.count;
            for (uint32_t member = run.first; member < end;)
            {
                auto record = next_record(*inputs[r], member, end);
                out.write(&record, sizeof(record));
                uint64_t remaining = record.bytes;
                while (remaining)
                {
                    uint64_t const n = std::min<uint64_t>(remaining, copy.size());
                    inputs[r]->read(copy.data(), n);
                    out.write(copy.data(), n);
                    remaining -= n;
                }
                member += record.count;
            }
        }
        state.stats.merge_written_bytes += active.output.bytes - begin;
        ++active.chunk;
        if (active.output.bytes - committed >= config.segment || active.chunk == chunks)
        {
            out.sync();
            failpoint("merge-after-flush");
            capture_positions(active, inputs);
            checkpoint.commit("merge");
            committed = active.output.bytes;
        }
    }
    for (auto& in : inputs) in->finish();
    out.close();
}

// Decode a batch contribution into the final encoder without building a
// full fragment array. FSE normalization matches compression_table().
void append_record(std::span<uint8_t const> payload, Record record, uint64_t start, uint64_t range,
                   Encoder& out, FSE_DTable const* table, Budget const& budget, uint64_t live)
{
    uint64_t ans = 0;
    bool terminated = false;
    for (unsigned i = 0; i < 10; ++i)
    {
        require(!payload.empty(), "truncated temporary ANS size");
        uint8_t const byte = payload.front();
        payload = payload.subspan(1);
        require(i != 9 || byte <= 1, "invalid temporary ANS size");
        ans |= uint64_t(byte & 0x7f) << (i * 7);
        if (!(byte & 0x80)) { terminated = true; break; }
    }
    require(terminated && ans > 0 && ans < payload.size(), "invalid temporary ANS bounds");
    uint64_t const low_size = payload.size() - ans;
    std::vector<uint8_t> high(low_size * 8 / (out.k - 7));
    require(!high.empty(), "empty temporary chunk");
    auto const count = POS2_FSE_decompress_usingDTable(high.data(), high.size(), payload.data(), ans, table);
    require(!POS2_FSE_isError(count) && count > 0, "cannot decode temporary chunk");
    std::vector<uint64_t> fields((low_size + 7) / 8);
    std::memcpy(fields.data(), payload.data() + ans, low_size);
    BitReader bits(fields, low_size * 8);
    uint64_t previous = start, last_member = 0;
    for (size_t i = 0; i < count; ++i)
    {
        uint64_t const low = bits.read_u64(out.k - 8);
        uint64_t const quotient = bits.read_unary_64();
        require(high[i] < 178, "invalid temporary high symbol");
        uint64_t const delta = quotient * (178ull << (out.k - 8)) + (uint64_t(high[i]) << (out.k - 8)) + low;
        require(delta <= start + uint64_t(record.count) * range - previous,
                "temporary fragment outside member range");
        previous += delta;
        uint64_t const member = (previous - start) / range;
        require(member < record.count && (i ? member <= last_member + 1 && delta > 0 : member == 0),
                "empty temporary member or duplicate fragment");
        last_member = member;
        budget.room(live + (out.high.size() + 1) * 64);
        out.append(previous + uint64_t(record.first) * range);
    }
    require(last_member + 1 == record.count, "missing temporary member");
}

// Flush whole bytes only; preserve the partial byte across GSZ entries.
void flush_index(std::ofstream& out, BitWriter& bits, Stats& stats, bool finish)
{
    auto const bytes = bits.asBytes();
    uint64_t const n = finish ? bytes.size() : bits.bitCount() / 8;
    uint32_t const carry = finish ? 0 : uint32_t(bits.bitCount() % 8);
    uint8_t const tail = carry ? bytes[n] : 0;
    out.write(reinterpret_cast<char const*>(bytes.data()), n);
    stats.index_bytes += n;
    stats.add(n);
    bits.clear();
    if (carry) bits.append(tail, carry);
}
void finalize(Checkpoints& checkpoint, Budget const& budget, FSE_CTable const* ctable)
{
    auto const& job = checkpoint.job;
    auto const& work = checkpoint.work;
    auto& state = checkpoint.state;
    auto& active = state.operation;
    auto& stats = state.stats;
    auto const& config = state.config;
    uint64_t const live = (active.inputs.size() + 2) * config.buffer;
    budget.room(live);
    auto inputs = readers(checkpoint);
    auto const index_path = work / "index.tmp";
    auto const group_path = work / "group.gplot.partial";
    std::ofstream index;
    append_output(index, index_path, active.index_bytes);
    std::vector<char> out_buffer(config.buffer);
    std::ofstream out;
    out.rdbuf()->pubsetbuf(out_buffer.data(), out_buffer.size());
    append_output(out, group_path, active.output.bytes);
    PlotGroupFile::Header header{PlotGroupFile::MAGIC, PlotGroupFile::FORMAT_VERSION,
        job.params.get_plot_group_id(), uint8_t(job.params.get_k()), uint8_t(job.params.get_strength()),
        uint16_t(job.members.size()), uint8_t(job.params.get_meta_group()), 0, uint8_t(job.memo.size())};
    auto write = [&](void const* data, uint64_t bytes)
    {
        out.write(static_cast<char const*>(data), bytes);
        stats.group_bytes += bytes;
        stats.add(bytes);
    };
    if (!active.output.bytes)
    {
        write(&header, sizeof(header));
        write(job.memo.data(), job.memo.size());
    }
    auto norm = compression_counts();
    std::unique_ptr<FSE_DTable, decltype(&POS2_FSE_freeDTable)> dtable(
        POS2_FSE_createDTable(11), POS2_FSE_freeDTable);
    require(bool(dtable) && !POS2_FSE_isError(POS2_FSE_buildDTable(dtable.get(), norm.data(), 177, 11)),
            "cannot build FSE decode table");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
    uint64_t const range = 1ull << (header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    BitWriter index_bits;
    if (active.carry_bits) index_bits.append(active.carry_value, active.carry_bits);
    uint64_t committed = active.output.bytes;
    while (active.chunk < chunks)
    {
        uint64_t const start = active.chunk * range;
        {
            Encoder encoder(start, header.k);
            for (size_t r = 0; r < active.inputs.size(); ++r)
            {
                auto const& run = active.inputs[r];
                uint32_t const end = run.first + run.count;
                for (uint32_t member = run.first; member < end;)
                {
                    auto record = next_record(*inputs[r], member, end);
                    budget.room(live + uint64_t(record.bytes) * 16 + encoder.high.size() * 64);
                    std::vector<uint8_t> payload(record.bytes);
                    inputs[r]->read(payload.data(), payload.size());
                    append_record(payload, record, start, range, encoder, dtable.get(),
                                  budget, live + uint64_t(record.bytes) * 16);
                    member += record.count;
                }
            }
            budget.room(live + encoder.high.size() * 64);
            auto payload = encoder.finish(ctable);
            uint64_t const size = payload.size();
            int64_t const delta = int64_t(size) - int64_t(gsz_expected_for_g(header.group_size));
            uint64_t const magnitude = delta < 0 ? uint64_t(-delta) : uint64_t(delta);
            uint32_t const rice = uint32_t(gsz_rice_k_for_g(header.group_size));
            require((magnitude >> rice) + rice + 2 <= 64, "group chunk index cannot encode this group size");
            gsz_encode(std::span<uint64_t const>(&size, 1), header.group_size, index_bits);
            if (index_bits.asBytes().size() >= 65536) flush_index(index, index_bits, stats, false);
            write(payload.data(), payload.size());
        }
        ++active.chunk;
        if (stats.group_bytes - committed >= config.segment || active.chunk == chunks)
        {
            flush_index(index, index_bits, stats, false);
            out.flush();
            index.flush();
            sync_path(group_path);
            sync_path(index_path);
            failpoint("final-after-flush");
            active.output.bytes = stats.group_bytes;
            active.index_bytes = stats.index_bytes;
            active.carry_bits = uint32_t(index_bits.bitCount());
            active.carry_value = active.carry_bits ? index_bits.asBytes()[0] : 0;
            capture_positions(active, inputs);
            checkpoint.commit("final");
            committed = active.output.bytes;
        }
    }
    for (auto& in : inputs) in->finish();
    flush_index(index, index_bits, stats, true);
    index.close();
    sync_path(index_path);
    header.chunk_index_offset = stats.group_bytes;
    auto source = input(index_path.string());
    std::vector<char> copy(config.buffer);
    for (uint64_t remaining = stats.index_bytes; remaining;)
    {
        uint64_t const n = std::min<uint64_t>(remaining, copy.size());
        source.read(copy.data(), n);
        write(copy.data(), n);
        remaining -= n;
    }
    source.close();
    out.close();
    std::fstream patch(group_path, std::ios::binary | std::ios::in | std::ios::out);
    patch.exceptions(std::ios::badbit | std::ios::failbit);
    patch.write(reinterpret_cast<char*>(&header), sizeof(header));
    patch.close();
    sync_path(group_path);
    failpoint("final-tail-after-flush");
    state.completed = stamp(group_path);
    active.output.bytes = stats.group_bytes;
    active.index_bytes = stats.index_bytes;
    active.carry_bits = active.carry_value = 0;
    state.phase = Phase::ready;
    checkpoint.commit("ready");
}
void publish(Checkpoints& checkpoint)
{
    auto const& work = checkpoint.work;
    auto& state = checkpoint.state;
    auto const partial = work / "group.gplot.partial", output = work / "group.gplot";
    if (fs::exists(output))
        require(same_output(state.completed, stamp(output)), "different group already occupies destination");
    else
    {
        require(same_output(state.completed, stamp(partial)), "completed temporary group was changed");
        fs::create_hard_link(partial, output);
    }
    sync_path(work, true);
    failpoint("publish-after-link");
    if (fs::exists(partial))
    {
        require(same_output(state.completed, stamp(partial)), "temporary group was replaced");
        fs::remove(partial);
    }
    sync_path(work, true);
    failpoint("publish-after-unlink");
    state.phase = Phase::done;
    state.active = false;
    state.runs.clear();
    state.next_runs.clear();
    state.merge_first = 0;
    checkpoint.commit("done");
}
} // namespace

int main(int argc, char** argv)
try
{
    require(argc >= 8,
        "Usage: pos2_pr118_stream INPUTS WORK_DIR RAM_MIB BATCH_MEMBERS FAN_IN BUFFER_KIB SEGMENT_MIB "
        "[--resume] [--consume-raw] [--keep-runs]");
    auto const started = std::chrono::steady_clock::now();
    uint64_t const ram = number(argv[3]), batch = number(argv[4]), fan = number(argv[5]);
    uint64_t const buffer_kib = number(argv[6]), segment_mib = number(argv[7]);
    require(ram >= 16 && ram <= 1048576 && batch >= 1 && batch <= 65535 && fan >= 2 && fan <= 256 &&
                buffer_kib >= 4 && buffer_kib <= 65536 && segment_mib >= 1 && segment_mib <= 1048576,
            "invalid RAM/batch/fan-in/buffer/segment setting");
    Config config{uint32_t(batch), uint32_t(fan), buffer_kib * 1024, segment_mib * mib};
    bool resume = false;
    for (int i = 8; i < argc; ++i)
    {
        std::string option(argv[i]);
        if (option == "--resume") resume = true;
        else if (option == "--consume-raw") config.consume = true;
        else if (option == "--keep-runs") config.keep = true;
        else throw std::runtime_error("unknown assembly option");
    }
    State state;
    Budget budget{ram * mib, 0, state.stats};
    auto job = load_job(argv[1], budget);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(job.params.get_k());
    budget.room((std::min<uint64_t>(fan, (job.members.size() + batch - 1) / batch) + 2) * config.buffer);
    fs::path const work = fs::absolute(argv[2]).lexically_normal();
    if (resume) require(fs::is_directory(work) && fs::exists(work / "checkpoint"),
                        "--resume requires the original work directory and checkpoint");
    else
    {
        require(fs::create_directory(work), "work directory must not already exist");
        fs::permissions(work, fs::perms::owner_all);
        sync_path(work.parent_path(), true);
    }
    WorkLock lock(work);
    auto const identity = job_identity(job);
    if (resume)
    {
        budget.room(stamp(work / "checkpoint").bytes * 4);
        uint64_t const resume_peak = state.stats.peak_modeled_bytes;
        state = load_state(work, config, identity, job.members.size(), budget.limit, chunks);
        state.stats.peak_modeled_bytes = std::max(state.stats.peak_modeled_bytes, resume_peak);
    }
    else
    {
        state.config = config;
        budget.room((1ull << (job.params.get_k() - PlotFile::CHUNK_SPAN_RANGE_BITS)) * 8 + 16384);
        for (size_t i = 0; i < job.members.size(); ++i)
        {
            auto const before = stamp(job.members[i]);
            Raw checked(job, i);
            require(before == stamp(job.members[i]), "raw member changed during preflight");
            state.raw.push_back(before);
            state.stats.input_bytes += before.bytes;
        }
        state.stats.remaining_raw_bytes = state.stats.input_bytes;
        state.stats.peak();
    }
    Checkpoints checkpoint{work, state, job, identity};
    if (resume) checkpoint.recover();
    else checkpoint.commit("init");
    auto& stats = state.stats;
    auto table = compression_table();
    if (state.phase == Phase::stage)
    {
        while (state.staged < job.members.size())
        {
            uint32_t const first = state.staged;
            uint32_t const count = uint32_t(std::min<uint64_t>(batch, job.members.size() - first));
            auto unchanged = [&]
            {
                for (uint32_t i = first; i < first + count; ++i)
                    require(same_output(state.raw[i], stamp(job.members[i])), "raw member changed during staging");
            };
            unchanged();
            auto run = stage(job, Run{state.next_id, first, count}, work,
                             config.segment, config.buffer, budget, stats, table.get());
            failpoint("stage-after-flush");
            unchanged();
            state.runs.push_back(run);
            state.staged += count;
            ++state.next_id;
            ++stats.initial_runs;
            checkpoint.commit("stage");
        }
        std::cout << "Staged " << state.runs.size() << " runs: " << stats.staged_bytes << " bytes\n";
        state.phase = Phase::merge;
        checkpoint.commit("plan");
    }
    while (state.phase == Phase::merge)
    {
        if (!state.active)
        {
            if (state.runs.size() <= fan)
            {
                state.operation = Active{Run{state.next_id++, 0, uint32_t(job.members.size())}};
                state.operation.inputs = state.runs;
                state.operation.positions.resize(state.runs.size());
                state.active = true;
                state.phase = Phase::final;
                checkpoint.commit("plan");
                break;
            }
            if (state.merge_first == state.runs.size())
            {
                state.runs = std::move(state.next_runs);
                state.next_runs.clear();
                state.merge_first = 0;
                ++stats.merge_passes;
                checkpoint.commit("plan");
                std::cout << "Merged to " << state.runs.size() << " runs\n";
                continue;
            }
            auto group = std::span<Run const>(state.runs).subspan(state.merge_first,
                std::min<size_t>(fan, state.runs.size() - state.merge_first));
            if (group.size() == 1)
            {
                state.next_runs.push_back(group.front());
                ++state.merge_first;
                checkpoint.commit("plan");
                continue;
            }
            state.operation = Active{Run{state.next_id++, group.front().first,
                group.back().first + group.back().count - group.front().first}};
            state.operation.inputs.assign(group.begin(), group.end());
            state.operation.positions.resize(group.size());
            state.active = true;
            checkpoint.commit("plan");
        }
        merge(checkpoint, chunks, budget);
        state.next_runs.push_back(state.operation.output);
        state.merge_first += state.operation.inputs.size();
        state.active = false;
        checkpoint.commit("plan");
    }
    if (state.phase == Phase::final) finalize(checkpoint, budget, table.get());
    if (state.phase == Phase::ready) publish(checkpoint);
    uint64_t peak_rss_kib = 0;
#ifdef __linux__
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) peak_rss_kib = usage.ru_maxrss;
#endif
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::ostringstream report;
    report << "{\n  \"members\": " << job.members.size()
           << ",\n  \"input_bytes\": " << stats.input_bytes
           << ",\n  \"raw_payload_read_bytes\": " << stats.raw_payload_read_bytes
           << ",\n  \"staged_bytes\": " << stats.staged_bytes
           << ",\n  \"merge_written_bytes\": " << stats.merge_written_bytes
           << ",\n  \"run_read_bytes\": " << stats.run_read_bytes
           << ",\n  \"group_bytes\": " << stats.group_bytes
           << ",\n  \"index_bytes\": " << stats.index_bytes
           << ",\n  \"peak_work_file_bytes\": " << stats.peak_work_bytes
           << ",\n  \"remaining_work_file_bytes\": " << stats.work_bytes
           << ",\n  \"reclaimed_bytes\": " << stats.reclaimed_bytes
           << ",\n  \"peak_modeled_bytes\": " << stats.peak_modeled_bytes
           << ",\n  \"peak_rss_kib\": " << peak_rss_kib
           << ",\n  \"initial_runs\": " << stats.initial_runs
           << ",\n  \"merge_passes\": " << stats.merge_passes
           << ",\n  \"max_active_members\": " << stats.max_active_members
           << ",\n  \"max_active_runs\": " << stats.max_active_runs
           << ",\n  \"remaining_raw_bytes\": " << stats.remaining_raw_bytes
           << ",\n  \"reclaimed_raw_bytes\": " << stats.reclaimed_raw_bytes
           << ",\n  \"peak_total_file_bytes\": " << stats.peak_total_file_bytes
           << ",\n  \"seconds_this_run\": " << seconds << "\n}\n";
    atomic_save(work / "metrics.json", report.str());
    std::cout << "PASS assembly: " << (work / "group.gplot") << "\n";
    return 0;
}
catch (std::exception const& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
