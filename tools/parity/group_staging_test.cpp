#include "host/GroupStaging.hpp"

#include <chrono>
#include <iostream>
#include <sys/wait.h>

using namespace group_grow;
using group_tools::require;
namespace fs = std::filesystem;

std::vector<uint8_t> contents(fs::path const& path)
{
    auto in = group_tools::input(path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}

Identity initial_identity()
{
    Identity identity;
    identity.header.magic = PlotGroupFile::MAGIC;
    identity.header.version = PlotGroupFile::FORMAT_VERSION;
    identity.header.group_id = PlotGroupId(std::string(64, 'a'));
    identity.header.k = 18; identity.header.strength = 2;
    identity.memo = {0, 17, 255}; identity.header.memo_length = uint8_t(identity.memo.size());
    identity.header.chunk_index_offset = sizeof(Header) + identity.memo.size();
    return identity;
}

Config config()
{
    Config config;
    config.capacity_bytes = 64ull * 1024 * 1024;
    config.window_bytes = 32768; config.codec_bytes = 8ull * 1024 * 1024;
    config.checkpoint_bytes = 65536; config.io_overlap = false; config.workers = 1;
    config.memory_limit_bytes = StagingEngine::memory_bytes(18, config, 3);
    return config;
}

std::vector<std::vector<uint64_t>> members(size_t count = 3, uint64_t per_chunk = 8)
{
    std::vector<std::vector<uint64_t>> result(count);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18);
    uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t member = 0; member < result.size(); ++member)
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
            for (uint64_t i = 0; i < per_chunk; ++i)
            {
                uint64_t value = chunk * range + i * (range / per_chunk) + member + 7;
                result[member].push_back(value);
                if (i == 3) result[member].push_back(value); // canonical deduplication
            }
    return result;
}

void direct(fs::path const& path, Members input, Config cfg = config())
{
    cfg.memory_limit_bytes = std::max(cfg.memory_limit_bytes, Engine::memory_bytes(18, cfg, uint16_t(input.size())));
    Engine engine(path.string(), cfg, false, initial_identity()); engine.confirm_identity();
    require(engine.append(input).accepted, "synthetic direct output did not fit");
}

void complete(fs::path const& path, Members input, bool resume, Config cfg = config())
{
    // Once durable footer retirement starts, the existing finalizing marker is
    // deliberately handled by the ordinary Engine's publication recovery.
    if (resume && fs::exists(path) && fs::file_size(path) >= sizeof(Header) && !StagingEngine::pending(path.string()))
    {
        auto in = group_tools::input(path.string()); auto header = group_tools::read<Header>(in);
        if (header.version == 0xf1)
        {
            cfg.memory_limit_bytes = std::max(cfg.memory_limit_bytes, Engine::memory_bytes(18, cfg));
            Engine engine(path.string(), cfg, true); engine.confirm_identity(); engine.resume({}); return;
        }
        if (header.version == PlotGroupFile::FORMAT_VERSION && header.group_size == input.size()) return;
    }
    StagingEngine engine(path.string(), cfg, resume, initial_identity()); engine.confirm_identity();
    while (engine.committed_member_count() < input.size())
    {
        auto result = engine.append(input.subspan(engine.committed_member_count(), 1));
        require(result.accepted, "synthetic staged run rejected");
    }
    auto result = engine.finish();
    require(result.members == input.size() && result.final_bytes == fs::file_size(path) &&
        result.peak_file_bytes <= cfg.capacity_bytes, "staging final result bounds differ");
}

void parity(fs::path const& root, std::vector<std::vector<uint64_t>> const& input)
{
    auto reference = root / "reference.gplot"; direct(reference, input);
    auto expected = contents(reference);
    auto staged = root / "staged.gplot"; complete(staged, input, false);
    require(contents(staged) == expected && !StagingEngine::pending(staged.string()), "staged output differs from direct bytes");
    auto mtime = fs::last_write_time(staged); complete(staged, input, true);
    require(fs::last_write_time(staged) == mtime && contents(staged) == expected, "completed staging resume changed output");

    auto existing = root / "existing.gplot"; direct(existing, Members(input).subspan(0, 1));
    complete(existing, input, true);
    require(contents(existing) == expected, "staged extension changed original contribution");

    std::vector<Member> compact;
    for (auto const& member : input)
    {
        compact.emplace_back(std::vector<uint64_t>(member)); compact.back().compact(18);
    }
    auto packed = root / "packed.gplot"; complete(packed, compact, false);
    require(contents(packed) == expected, "compact staged member parity differs");

    auto released = root / "released.gplot";
    {
        auto cfg = config(); cfg.release_cache = true;
        StagingEngine engine(released.string(), cfg, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted, "cache-release staged batch rejected"); engine.finish();
        require(engine.stats().write_calls < engine.stats().encode_chunks, "staging output was not coalesced");
    }
    require(contents(released) == expected, "phase cache release changed canonical staging bytes");

    auto reused = root / "reused.gplot";
    {
        StagingEngine engine(reused.string(), config(), false, initial_identity()); engine.confirm_identity();
        require(engine.append(Members(input).subspan(0, 1)).accepted, "first reusable staged batch rejected"); engine.finish();
        require(engine.can_append() && engine.append(Members(input).subspan(1)).accepted, "later reusable staged batch rejected"); engine.finish();
    }
    require(contents(reused) == expected, "reused staging object changed canonical bytes");

    auto group = PlotGroupFile::open(staged.string());
    uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(18); ++chunk)
    {
        std::vector<std::vector<ProofFragment>> decoded(input.size()); group.readChunk(chunk, decoded);
        for (size_t member = 0; member < input.size(); ++member)
        {
            auto first = std::lower_bound(input[member].begin(), input[member].end(), chunk * range);
            auto last = std::lower_bound(first, input[member].end(), (chunk + 1) * range);
            std::vector<uint64_t> expected_member(first, last);
            expected_member.erase(std::unique(expected_member.begin(), expected_member.end()), expected_member.end());
            require(decoded[member] == expected_member, "staged upstream reader returned a different member");
        }
    }
}

void capacity(fs::path const& root, std::vector<std::vector<uint64_t>> const& input)
{
    uint64_t peak = 0;
    {
        StagingEngine engine((root / "measure.gplot").string(), config(), false, initial_identity()); engine.confirm_identity();
        auto appended = engine.append(Members(input).subspan(0, 1)); require(appended.accepted, "staging peak measurement rejected");
        peak = appended.result.peak_file_bytes; engine.finish();
    }
    auto cfg = config(); cfg.capacity_bytes = peak;
    {
        StagingEngine engine((root / "exact.gplot").string(), cfg, false, initial_identity()); engine.confirm_identity();
        auto appended = engine.append(Members(input).subspan(0, 1));
        require(appended.accepted && appended.result.peak_file_bytes == peak, "exact staging reservation rejected"); engine.finish();
    }
    cfg.capacity_bytes = peak - 1;
    {
        auto path = root / "short.gplot";
        StagingEngine engine(path.string(), cfg, false, initial_identity()); engine.confirm_identity();
        require(!engine.append(Members(input).subspan(0, 1)).accepted && !fs::file_size(path), "short staging capacity changed destination");
    }
    auto existing = root / "reject-existing.gplot"; direct(existing, Members(input).subspan(0, 1));
    auto before = contents(existing); cfg.capacity_bytes = before.size();
    bool refused = false;
    try { StagingEngine engine(existing.string(), config(), false); }
    catch (std::runtime_error const&) { refused = true; }
    require(refused && contents(existing) == before, "existing ordinary staging target did not require resume");
    {
        StagingEngine engine(existing.string(), cfg, true); engine.confirm_identity();
        require(!engine.append(Members(input).subspan(1, 1)).accepted, "near-capacity staging extension accepted");
    }
    require(contents(existing) == before, "rejected staging extension changed existing completed bytes");

    // A window that fits each contribution can still be too small for their
    // combined final chunk. Admission must reject before committing that run.
    cfg = config(); cfg.window_bytes = sizeof(Header);
    auto bounded = root / "bounded.gplot";
    StagingEngine engine(bounded.string(), cfg, false, initial_identity()); engine.confirm_identity();
    auto first = engine.append(Members(input).subspan(0, 1)); require(first.accepted, "small first contribution did not fit");
    require(!engine.append(Members(input).subspan(1, 1)).accepted && engine.committed_member_count() == 1,
        "unfinishable combined staging chunk committed");
    engine.finish();

    auto short_checkpoint = root / "short-checkpoint.gplot"; cfg = config(); cfg.checkpoint_bytes = sizeof(Header) - 1;
    {
        StagingEngine short_engine(short_checkpoint.string(), cfg, false, initial_identity()); short_engine.confirm_identity();
        require(short_engine.append(Members(input).subspan(0, 1)).accepted, "short-checkpoint staged append rejected");
        short_engine.finish();
    }
    require(contents(short_checkpoint) == contents(root / "measure.gplot"), "short checkpoint changed staged publication");

    auto substitute = root / "substitute.gplot";
    {
        StagingEngine empty(substitute.string(), config(), false, initial_identity());
        fs::rename(substitute, root / "renamed-empty.gplot");
        std::ofstream out(substitute); out << "preserve replacement path";
    }
    require(fs::exists(substitute) && !contents(substitute).empty(), "staging cleanup unlinked replacement inode");

    auto damaged = root / "damaged.gplot";
    fs::copy_file(root / "measure.gplot", damaged);
    {
        std::fstream file(damaged, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(offsetof(Header, version)); file.put(char(0xf2));
    }
    auto damaged_before = contents(damaged);
    require(StagingEngine::pending(damaged.string()), "private staging marker was not recognized"); refused = false;
    try { StagingEngine damaged_engine(damaged.string(), config(), true); }
    catch (std::runtime_error const&) { refused = true; }
    require(refused && contents(damaged) == damaged_before, "damaged staged marker was accepted or mutated");
}

void recovery(fs::path const& root, std::vector<std::vector<uint64_t>> const& input)
{
    auto expected = contents(root / "reference.gplot");
    std::vector<std::string> points{
        "stage-allocated", "stage-bootstrap", "stage-checkpoint-partial", "stage-checkpoint-partial:2",
        "stage-initialized", "stage-run-partial-flushed", "stage-run-flushed", "stage-run-committed", "stage-run-flushed:2",
        "stage-run-committed:2", "stage-pack-initialized", "stage-pack-flushed", "stage-pack-committed",
        "stage-tail-flushed", "stage-tail-committed", "stage-compact-partial", "stage-compact-flushed",
        "stage-compact-committed", "stage-compact-partial:2", "stage-final-header-partial", "stage-finalizing",
        "stage-retire-partial", "stage-retire-partial:2", "stage-retired", "stage-truncated", "stage-published"};
    for (bool prefix : {false, true})
        for (auto const& point : points)
        {
            if (prefix && point == "stage-bootstrap") continue;
            auto const name = std::string(prefix ? "recovery-prefix-" : "recovery-") + point;
            auto directory = root / name; fs::create_directory(directory); auto path = directory / "group.gplot";
            if (prefix) direct(path, Members(input).subspan(0, 1));
            pid_t child = ::fork(); require(child >= 0, "cannot fork staging recovery check");
            if (!child)
            {
                ::setenv("XCHPLOT2_GROW_FAILPOINT", point.c_str(), 1);
                try { complete(path, input, prefix); }
                catch (...) { std::_Exit(1); }
                std::_Exit(0);
            }
            int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
                "staging failpoint was not reached: " + name);
            complete(path, input, true);
            require(contents(path) == expected, "staging recovery changed canonical bytes: " + name);
            size_t entries = 0; for (auto const& entry : fs::directory_iterator(directory)) { (void)entry; ++entries; }
            require(entries == 1, "staging recovery created an external construction file");
        }
}

void gsz_admission(fs::path const& root)
{
    auto dense = members(17, 64);
    // More raw values than the compressed-symbol bound remains safe: exact
    // sizing borrows these duplicates, and the codec allocates unique deltas.
    dense.back().insert(dense.back().begin(), 8192, dense.back().front());
    auto cfg = config(); cfg.memory_limit_bytes = StagingEngine::memory_bytes(18, cfg, 32);
    auto reference = root / "gsz-reference.gplot"; direct(reference, dense, cfg);
    auto expected = contents(reference);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18);
    auto finish_many = [&](fs::path const& path, bool resume)
    {
        StagingEngine engine(path.string(), cfg, resume, initial_identity()); engine.confirm_identity();
        if (resume) require(engine.committed_member_count() == 16, "exact-sizing recovery lost committed runs");
        while (engine.committed_member_count() < dense.size())
        {
            size_t const next = engine.committed_member_count();
            AppendResult result;
            if (next + 1 == dense.size())
            {
                std::vector<Member> compact;
                compact.emplace_back(std::vector<uint64_t>(dense[next])); compact.back().compact(18);
                result = engine.append(compact);
            }
            else result = engine.append(Members(dense).subspan(next, 1));
            require(result.accepted, "encodable larger staged group rejected GSZ admission");
        }
        engine.finish(); return engine.stats();
    };
    auto path = root / "gsz-staged.gplot";
    auto stats = finish_many(path, false);
    require(contents(path) == expected && stats.sized_chunks == dense.size() * chunks,
            "proven GSZ interval changed canonical bytes or repeated combined exact sizing");

    auto prefix = root / "gsz-prefix.gplot"; direct(prefix, Members(dense).subspan(0, 15), cfg);
    {
        StagingEngine engine(prefix.string(), cfg, true); engine.confirm_identity();
        require(engine.append(Members(dense).subspan(15)).accepted, "ordinary prefix extension rejected exact GSZ admission");
        engine.finish();
    }
    require(contents(prefix) == expected, "ordinary prefix exact GSZ sizing changed canonical bytes");

    auto interrupted = root / "gsz-recovery.gplot";
    pid_t child = ::fork(); require(child >= 0, "cannot fork exact GSZ recovery check");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "stage-run-committed:16", 1);
        try { finish_many(interrupted, false); }
        catch (...) { std::_Exit(1); }
        std::_Exit(0);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "exact GSZ committed-run failpoint was not reached");
    finish_many(interrupted, true);
    require(contents(interrupted) == expected, "exact GSZ committed-run recovery changed canonical bytes");

    auto chunk_regions = [](std::vector<std::vector<uint64_t>> const& input)
    {
        std::vector<FragmentRegion> regions;
        uint64_t const end = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        for (auto const& member : input)
        {
            auto last = std::lower_bound(member.begin(), member.end(), end);
            regions.emplace_back(std::span<uint64_t const>(member.data(), size_t(last - member.begin())));
        }
        return regions;
    };
    // Corrupt a committed compressed run, leaving its checkpoint and record
    // lengths intact. Its decoder failure must propagate before any writes.
    auto corrupt = root / "gsz-corrupt.gplot";
    {
        StagingEngine engine(corrupt.string(), cfg, false, initial_identity()); engine.confirm_identity();
        require(engine.append(Members(dense).subspan(0, 15)).accepted, "corruption fixture staging rejected");
    }
    ChunkCodec codec(18, cfg.codec_bytes); auto regions = chunk_regions(dense);
    auto encoded = codec.encode(0, {}, 0, std::span(regions).first(15));
    auto bytes = contents(corrupt);
    auto first = std::search(bytes.begin(), bytes.end(), encoded.begin(), encoded.end());
    require(first != bytes.end(), "cannot locate committed compressed fixture payload");
    {
        std::fstream file(corrupt, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(first - bytes.begin()); file.put(0);
    }
    auto before = contents(corrupt);
    {
        StagingEngine engine(corrupt.string(), cfg, true); engine.confirm_identity();
        bool threw = false;
        try { (void)engine.append(Members(dense).subspan(15, 1)); }
        catch (std::runtime_error const& error)
        { threw = std::string_view(error.what()) == "invalid staged ANS bounds"; }
        require(threw && !engine.stats().write_calls && engine.committed_member_count() == 15,
                "corrupt exact-sizing input became an admission fallback or changed state");
    }
    require(contents(corrupt) == before, "corrupt exact-sizing input changed persistent bytes");

    // A live instance may trust only its own unchanged committed sources.
    // An observed metadata change must restore read-only validation, even
    // when the new size interval would otherwise settle GSZ admission.
    auto changed = root / "gsz-changed.gplot";
    {
        StagingEngine engine(changed.string(), cfg, false, initial_identity()); engine.confirm_identity();
        require(engine.append(Members(dense).subspan(0, 15)).accepted, "changed-source fixture staging rejected");
        bytes = contents(changed);
        first = std::search(bytes.begin(), bytes.end(), encoded.begin(), encoded.end());
        require(first != bytes.end(), "cannot locate live committed fixture payload");
        auto const modified = fs::last_write_time(changed);
        {
            std::fstream file(changed, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(first - bytes.begin()); file.put(0);
        }
        fs::last_write_time(changed, modified - std::chrono::seconds(1));
        before = contents(changed); auto const writes = engine.stats().write_calls;
        bool threw = false;
        try { (void)engine.append(Members(dense).subspan(15, 1)); }
        catch (std::runtime_error const& error)
        { threw = std::string_view(error.what()) == "invalid staged ANS bounds"; }
        require(threw && engine.stats().write_calls == writes && engine.committed_member_count() == 15 &&
            contents(changed) == before, "changed committed source bypassed read-only validation");
    }

    // A genuinely sparse group can have an unrepresentable canonical GSZ
    // entry. Compare with exact codec sizing and preserve its prior run graph.
    auto sparse = members(32, 8); auto sparse_regions = chunk_regions(sparse);
    auto rejected = root / "gsz-rejected.gplot";
    StagingEngine engine(rejected.string(), cfg, false, initial_identity()); engine.confirm_identity();
    bool rejected_gsz = false;
    for (size_t member = 0; member < sparse.size(); ++member)
    {
        bool encodable = true;
        try { (void)codec.encoded_size(0, {}, 0, std::span(sparse_regions).first(member + 1)); }
        catch (UnencodableChunkSize const&) { encodable = false; }
        before = contents(rejected);
        auto result = engine.append(Members(sparse).subspan(member, 1));
        if (!encodable)
        {
            require(member + 1 > 16 && !result.accepted && engine.committed_member_count() == member &&
                contents(rejected) == before, "unencodable exact GSZ candidate changed committed state");
            engine.finish(); rejected_gsz = true; break;
        }
        require(result.accepted, "encodable sparse staged group was rejected before exact sizing");
    }
    require(rejected_gsz, "sparse exact GSZ rejection was not exercised");

    // Dense sources can prove a GSZ interval while their rebased member bridge
    // still exceeds the codec's delta width. Both the live fast path and resume
    // must reject this before writing the individually encodable contribution.
    uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t member : {15, 16})
    {
        dense[member].clear();
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
            for (uint64_t i = 0; i < 64; ++i) dense[member].push_back(chunk * range + member + i + 7);
    }
    auto clustered = root / "gsz-clustered.gplot";
    auto reject_bridge = [&](StagingEngine& target)
    {
        auto const before = contents(clustered); auto const writes = target.stats().write_calls;
        bool threw = false;
        try { (void)target.append(Members(dense).subspan(16, 1)); }
        catch (std::runtime_error const& error)
        { threw = std::string_view(error.what()) == "unencodable group fragment delta"; }
        require(threw && target.stats().write_calls == writes && target.committed_member_count() == 16 &&
            contents(clustered) == before, "unencodable rebased staging bridge changed committed bytes/state");
    };
    {
        StagingEngine target(clustered.string(), cfg, false, initial_identity()); target.confirm_identity();
        require(target.append(Members(dense).subspan(0, 16)).accepted, "clustered bridge prefix was not encodable");
        reject_bridge(target);
    }
    {
        StagingEngine target(clustered.string(), cfg, true); target.confirm_identity(); reject_bridge(target); target.finish();
    }
    auto clustered_reference = root / "gsz-clustered-reference.gplot"; direct(clustered_reference, Members(dense).subspan(0, 16), cfg);
    require(contents(clustered) == contents(clustered_reference), "rejected staging bridge changed final canonical prefix");
}

void fse_boundaries(fs::path const& root)
{
    // Exercise the first encodable FSE counts, both initialization parities,
    // and nearby byte-rounding transitions across the GSZ admission boundary.
    for (uint64_t count : {3, 4, 7, 8, 9})
    {
        auto input = members(32, count);
        auto const stem = std::string("fse-boundary-") + std::to_string(count);
        std::vector<FragmentRegion> regions;
        uint64_t const end = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        for (auto const& member : input)
        {
            auto last = std::lower_bound(member.begin(), member.end(), end);
            regions.emplace_back(std::span<uint64_t const>(member.data(), size_t(last - member.begin())));
        }
        auto cfg = config(); cfg.memory_limit_bytes = StagingEngine::memory_bytes(18, cfg, uint16_t(input.size()));
        auto staged = root / (stem + "-staged.gplot");
        size_t accepted = 0;
        {
            ChunkCodec codec(18, cfg.codec_bytes);
            StagingEngine engine(staged.string(), cfg, false, initial_identity()); engine.confirm_identity();
            for (; accepted < input.size(); ++accepted)
            {
                bool encodable = true;
                try { (void)codec.encoded_size(0, {}, 0, std::span(regions).first(accepted + 1)); }
                catch (UnencodableChunkSize const&) { encodable = false; }
                auto const before = contents(staged); auto const writes = engine.stats().write_calls;
                auto result = engine.append(Members(input).subspan(accepted, 1));
                require(result.accepted == encodable, "FSE boundary admission differs from exact codec sizing");
                if (!encodable)
                {
                    require(contents(staged) == before && engine.stats().write_calls == writes &&
                        engine.committed_member_count() == accepted, "FSE boundary rejection changed committed state");
                    require(!engine.append(Members(input).subspan(accepted, 1)).accepted &&
                        contents(staged) == before, "repeated FSE boundary rejection changed committed state");
                    break;
                }
            }
            require(accepted < input.size(), "FSE boundary GSZ rejection was not exercised"); engine.finish();
        }
        auto reference = root / (stem + "-reference.gplot"); direct(reference, Members(input).subspan(0, accepted), cfg);
        require(contents(staged) == contents(reference), "FSE boundary admission changed canonical bytes");
    }
}

void noncanonical_sources(fs::path const& root)
{
    auto cfg = config();
    cfg.memory_limit_bytes = std::max(StagingEngine::memory_bytes(18, cfg, 32), Engine::memory_bytes(18, cfg, 32));
    auto input = members(32, 8);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18);
    uint64_t const end = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<FragmentRegion> regions;
    for (auto const& member : input)
    {
        auto last = std::lower_bound(member.begin(), member.end(), end);
        regions.emplace_back(std::span<uint64_t const>(member.data(), size_t(last - member.begin())));
    }
    auto counts = group_tools::compression_counts();
    std::unique_ptr<FSE_DTable, decltype(&POS2_FSE_freeDTable)> table(POS2_FSE_createDTable(11), POS2_FSE_freeDTable);
    require(table && !POS2_FSE_isError(POS2_FSE_buildDTable(table.get(), counts.data(), 177, 11)),
        "cannot build noncanonical-source decode table");
    // The first stream has a noncanonical terminal transition and decodes fewer
    // symbols than the encoder-only length bound. The second overflows during
    // state initialization but is still accepted as the decoder's terminal pair.
    for (auto const& ans : {std::vector<uint8_t>{0x00, 0x80, 0x00, 0x04, 0x20}, std::vector<uint8_t>{0x01}})
    {
        std::array<uint8_t, 64> high{};
        size_t const symbols = POS2_FSE_decompress_usingDTable(high.data(), high.size(), ans.data(), ans.size(), table.get());
        require(symbols == (ans.size() == 1 ? 2 : 3), "noncanonical ANS fixture decoded a different symbol count");
        // Leave the source's last fragment beyond the region midpoint so its
        // bridge to the next member is within the ordinary delta-width limit.
        uint64_t const quotient = 48 / symbols;
        BitWriter low;
        for (size_t i = 0; i < symbols; ++i)
        {
            low.append(1, 10); low.append((1ull << quotient) - 1, uint32_t(quotient)); low.append(0, 1);
        }
        std::vector<uint8_t> payload{uint8_t(ans.size())}; payload.insert(payload.end(), ans.begin(), ans.end());
        auto bits = low.asBytes(); payload.insert(payload.end(), bits.begin(), bits.end());
        ChunkCodec codec(18, cfg.codec_bytes);
        require(codec.scan(0, payload, 1) == payload.size(), "noncanonical source chunk was not accepted by the codec");
        auto identity = initial_identity(); identity.header.group_size = 1;
        identity.header.chunk_index_offset += chunks * payload.size();
        BitWriter index; std::vector<uint64_t> sizes(size_t(chunks), payload.size()); gsz_encode(sizes, 1, index);
        auto const stem = std::string("noncanonical-") + std::to_string(ans.size());
        auto staged = root / (stem + "-staged.gplot");
        {
            std::ofstream out(staged, std::ios::binary); out.exceptions(std::ios::failbit | std::ios::badbit);
            out.write(reinterpret_cast<char const*>(&identity.header), sizeof(Header));
            out.write(reinterpret_cast<char const*>(identity.memo.data()), std::streamsize(identity.memo.size()));
            for (uint64_t chunk = 0; chunk < chunks; ++chunk)
                out.write(reinterpret_cast<char const*>(payload.data()), std::streamsize(payload.size()));
            auto bytes = index.asBytes(); out.write(reinterpret_cast<char const*>(bytes.data()), std::streamsize(bytes.size()));
        }
        auto reference = root / (stem + "-reference.gplot"); fs::copy_file(staged, reference);
        size_t accepted = 0;
        {
            StagingEngine engine(staged.string(), cfg, true); engine.confirm_identity();
            for (; accepted < input.size(); ++accepted)
            {
                bool encodable = true;
                try { (void)codec.encoded_size(0, payload, 1, std::span(regions).first(accepted + 1)); }
                catch (UnencodableChunkSize const&) { encodable = false; }
                auto const before = contents(staged); auto const writes = engine.stats().write_calls;
                auto result = engine.append(Members(input).subspan(accepted, 1));
                require(result.accepted == encodable, "noncanonical source admission differs from exact codec sizing");
                if (!encodable)
                {
                    require(contents(staged) == before && engine.stats().write_calls == writes &&
                        engine.committed_member_count() == accepted + 1, "noncanonical source GSZ rejection changed committed state");
                    break;
                }
            }
            require(accepted && accepted < input.size(), "noncanonical source acceptance and GSZ rejection were not exercised");
            engine.finish();
        }
        Engine engine(reference.string(), cfg, true); engine.confirm_identity();
        require(engine.append(Members(input).subspan(0, accepted)).accepted, "ordinary growth rejected noncanonical source fixture");
        require(contents(staged) == contents(reference), "noncanonical source staging differs from ordinary growth");
    }
}

void buffered_parallel(fs::path const& root, std::vector<std::vector<uint64_t>> const& input)
{
    auto expected = contents(root / "reference.gplot");
    auto cfg = config(); cfg.workers = 3;
    cfg.memory_limit_bytes = StagingEngine::memory_bytes(18, cfg, uint16_t(input.size()));
    auto parallel = root / "parallel.gplot";
    {
        StagingEngine engine(parallel.string(), cfg, false, initial_identity()); engine.confirm_identity();
        for (size_t member = 0; member < input.size(); ++member)
            require(engine.append(Members(input).subspan(member, 1)).accepted, "parallel fixture staging rejected");
        uint64_t const reads = engine.read_calls(); engine.finish();
        require(engine.packing_workers() > 1, "parallel staged packing did not use requested workers");
        require(engine.read_calls() - reads < input.size() * PlotGroupFile::getChunkCountForK(18),
                "staged run payload reads were not buffered");
    }
    require(contents(parallel) == expected, "buffered parallel packing differs from canonical bytes");

    // Reserve exactly one conservative chunk's codec work. The requested pool
    // must reduce to serial rather than change admission or exceed its budget.
    uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    ChunkCodec codec(18, cfg.codec_bytes); uint64_t source_bytes = 0;
    for (auto const& member : input)
    {
        auto last = std::lower_bound(member.begin(), member.end(), range);
        std::span<uint64_t const> region(member.data(), size_t(last - member.begin()));
        std::array<std::span<uint64_t const>, 1> regions{region};
        source_bytes += codec.encoded_size(0, {}, 0, regions);
    }
    cfg.codec_bytes = ChunkCodec::estimate_work(18, 0, source_bytes * 8 / (18 - 7));
    cfg.memory_limit_bytes = StagingEngine::memory_bytes(18, cfg, uint16_t(input.size()));
    auto low_codec = root / "parallel-low-codec.gplot";
    {
        StagingEngine engine(low_codec.string(), cfg, false, initial_identity()); engine.confirm_identity();
        for (size_t member = 0; member < input.size(); ++member)
            require(engine.append(Members(input).subspan(member, 1)).accepted, "low-codec staged admission changed");
        engine.finish(); require(engine.packing_workers() == 1, "low-codec packing did not reduce workers");
    }
    require(contents(low_codec) == expected, "low-codec worker reduction changed canonical bytes");

    // One-member runs make one chunk's frontier/region descriptors larger than
    // this input window. The fixed single-chunk allowance must remain viable.
    auto many = members(15, 8); cfg = config(); cfg.window_bytes = 512; cfg.workers = 4;
    cfg.memory_limit_bytes = StagingEngine::memory_bytes(18, cfg, uint16_t(many.size()));
    auto tiny = root / "parallel-tiny-window.gplot";
    {
        StagingEngine engine(tiny.string(), cfg, false, initial_identity()); engine.confirm_identity();
        for (size_t member = 0; member < many.size(); ++member)
            require(engine.append(Members(many).subspan(member, 1)).accepted, "tiny-window many-run staging rejected");
        engine.finish(); require(engine.packing_workers() == 1, "single-chunk metadata fallback was not exercised");
    }
    auto tiny_reference = root / "parallel-tiny-reference.gplot"; direct(tiny_reference, many);
    require(contents(tiny) == contents(tiny_reference), "tiny-window many-run packing changed canonical bytes");

    // Commit within the first gathered window, then resume with a different
    // worker partition. Cached/gathered source ahead of the checkpoint is live.
    cfg = config(); cfg.workers = 3; cfg.checkpoint_bytes = 256; cfg.release_cache = true;
    cfg.memory_limit_bytes = StagingEngine::memory_bytes(18, cfg, uint16_t(input.size()));
    auto interrupted = root / "parallel-window-recovery.gplot";
    pid_t child = ::fork(); require(child >= 0, "cannot fork gathered-window recovery check");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "stage-pack-committed:2", 1);
        try { complete(interrupted, input, false, cfg); }
        catch (...) { std::_Exit(1); }
        std::_Exit(0);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "gathered-window checkpoint failpoint was not reached");
    cfg.workers = 2; complete(interrupted, input, true, cfg);
    require(contents(interrupted) == expected, "gathered-window checkpoint recovery changed canonical bytes");
}

int main() try
{
    char name[] = "/tmp/xchplot2-staging-XXXXXX";
    require(::mkdtemp(name) != nullptr, "cannot create staging test directory"); fs::path root(name);
    struct Cleanup { fs::path root; ~Cleanup() { fs::remove_all(root); } } cleanup{root};
    auto input = members(); parity(root, input); capacity(root, input); recovery(root, input); gsz_admission(root); fse_boundaries(root); noncanonical_sources(root); buffered_parallel(root, input);
    std::cout << "same-file compressed staging parity, admission and recovery checks passed\n";
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
