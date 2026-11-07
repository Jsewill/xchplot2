#include "host/GroupEngine.hpp"
#include "host/GroupWriter.hpp"

#include <iostream>
#include <random>
#include <sys/wait.h>

using namespace group_grow;
using group_tools::require;

template <typename F> void rejects(F&& operation)
{
    bool rejected = false;
    try { operation(); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "invalid codec input was accepted");
}

std::vector<uint8_t> copy(std::span<uint8_t const> bytes)
{
    return {bytes.begin(), bytes.end()};
}

std::array<uint8_t, 32> framed_digest(Members members)
{
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < members.size(); ++i)
    {
        auto const member = members[i];
        uint64_t const size = member.size();
        auto const prefix = reinterpret_cast<uint8_t const*>(&size);
        bytes.insert(bytes.end(), prefix, prefix + sizeof(size));
        if (member.compact())
        {
            for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(member.k()); ++chunk)
                for (uint64_t value : member.region(chunk))
                {
                    auto const data = reinterpret_cast<uint8_t const*>(&value);
                    bytes.insert(bytes.end(), data, data + sizeof(value));
                }
        }
        else if (!member.empty())
        {
            auto const data = reinterpret_cast<uint8_t const*>(member.raw().data());
            bytes.insert(bytes.end(), data, data + member.size() * sizeof(uint64_t));
        }
    }
    return grow_digest(bytes);
}

void segmented_sha()
{
    auto check = [](std::span<std::span<uint8_t const> const> parts)
    {
        std::vector<pos2_sha256_segment> segments;
        std::vector<uint8_t> concatenated;
        for (auto const part : parts)
        {
            segments.push_back({part.empty() ? nullptr : part.data(), part.size()});
            if (!part.empty()) concatenated.insert(concatenated.end(), part.begin(), part.end());
        }
        std::array<uint8_t, 32> result{};
        require(pos2_sha256_segments(segments.data(), segments.size(), result.data()) == POS2_OK,
                "segmented SHA ABI rejected valid spans");
        require(result == grow_digest(concatenated), "segmented SHA ABI differs from upstream SHA");
    };
    check({});
    std::array<uint8_t, 257> data{};
    for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t(i);
    for (size_t bytes : {0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 255, 256, 257})
    {
        size_t const split = bytes / 2;
        std::array<std::span<uint8_t const>, 4> parts{};
        parts[1] = std::span(data).first(split);
        parts[2] = std::span(data).subspan(split, bytes - split);
        check(parts);
    }
    std::vector<std::vector<uint64_t>> members{{}, {0, 1, UINT64_MAX}, {3, 5, 8, 13, 21}};
    std::vector<uint64_t> sizes(members.size());
    std::vector<std::span<uint8_t const>> parts;
    for (size_t i = 0; i < members.size(); ++i)
    {
        sizes[i] = members[i].size();
        parts.emplace_back(reinterpret_cast<uint8_t const*>(&sizes[i]), sizeof(sizes[i]));
        parts.emplace_back(reinterpret_cast<uint8_t const*>(members[i].data()),
                           members[i].size() * sizeof(uint64_t));
    }
    check(parts);
}

void synthetic()
{
    constexpr uint64_t budget = 1ull << 24;
    for (uint8_t k = 18; k <= 28; k += 2)
    {
        ChunkCodec codec(k, budget);
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        uint64_t const ordinal = 3;
        std::vector<uint64_t> first, second, duplicate;
        for (uint64_t i = 0; i < 64; ++i)
        {
            first.push_back(ordinal * range + i * (range / 64) + 7);
            second.push_back(ordinal * range + i * (range / 64) + 11);
            duplicate.push_back(first.back());
            duplicate.push_back(first.back());
        }
        std::span<uint64_t const> a = first, b = second, repeated = duplicate;
        auto const one = copy(codec.encode(ordinal, {}, 0, {&a, 1}));
        require(codec.encoded_size(ordinal, {}, 0, {&repeated, 1}) == one.size(),
                "duplicate sizing differs from emission");
        require(copy(codec.encode(ordinal, {}, 0, {&repeated, 1})) == one,
                "duplicate handling differs");
        require(one.size() <= ChunkCodec::fresh_bound(k, ordinal, {&repeated, 1}),
                "duplicate chunk exceeded fresh reservation bound");
        std::array<std::span<uint64_t const>, 2> both{a, b};
        auto const full = copy(codec.encode(ordinal, {}, 0, both));
        require(codec.encoded_size(ordinal, {}, 0, both) == full.size() &&
                codec.encoded_size(ordinal, one, 1, {&b, 1}) == full.size(),
                "exact sizing differs from fresh or appended emission");
        require(full.size() <= ChunkCodec::fresh_bound(k, ordinal, both),
                "fresh reservation bound is shorter than encoded output");
        require(copy(codec.encode(ordinal, one, 1, {&b, 1})) == full,
                "appended chunk differs from batched chunk");
        require(codec.scan(ordinal, full, 2) == full.size(), "scanner length differs");
        auto prefix = full;
        prefix.insert(prefix.end(), full.begin(), full.end());
        require(codec.scan(ordinal, prefix, 2) == full.size(), "scanner consumed next chunk");

        // More than the mean entries per member remains valid: the compressed
        // stream and budget, rather than a mean-count guess, bound decoding.
        std::vector<uint64_t> dense;
        for (uint64_t i = 0; i < 256; ++i)
            dense.push_back(ordinal * range + i * (range / 256) + 7);
        std::span<uint64_t const> dense_span = dense;
        auto const dense_bytes = copy(codec.encode(ordinal, {}, 0, {&dense_span, 1}));
        require(codec.encoded_size(ordinal, {}, 0, {&dense_span, 1}) == dense_bytes.size(),
                "dense sizing differs from emission");
        require(dense_bytes.size() <= ChunkCodec::fresh_bound(k, ordinal, {&dense_span, 1}),
                "dense chunk exceeded fresh reservation bound");
        require(codec.scan(ordinal, dense_bytes, 1) == dense_bytes.size(), "dense chunk rejected");

        // Maximum-width deltas cross a loaded word at every bit alignment.
        // Compare their low-bit bytes with the pinned upstream BitWriter.
        for (uint64_t alignment = 0; alignment < 8; ++alignment)
        {
            uint64_t const threshold = 178ull << (k - 8);
            std::vector<uint64_t> wide_fields{ordinal * range + alignment * threshold + 3};
            wide_fields.push_back(wide_fields.back() + uint64_t(71 - k) * threshold + 1);
            while (wide_fields.size() < 64) wide_fields.push_back(wide_fields.back() + 1);
            std::span<uint64_t const> fields = wide_fields;
            auto const encoded = copy(codec.encode(ordinal, {}, 0, {&fields, 1}));
            BitWriter expected_low;
            uint64_t previous = ordinal * range;
            for (uint64_t fragment : wide_fields)
            {
                uint64_t const delta = fragment - previous, quotient = delta / threshold;
                expected_low.append((((1ull << quotient) - 1) << (k - 8)) |
                    (delta & ((1ull << (k - 8)) - 1)), uint32_t(quotient + 1 + k - 8));
                previous = fragment;
            }
            size_t prefix = 0;
            uint64_t ans = 0;
            for (unsigned shift = 0;; shift += 7)
            {
                uint8_t const byte = encoded[prefix++];
                ans |= uint64_t(byte & 0x7f) << shift;
                if (!(byte & 0x80)) break;
            }
            require(copy(std::span(encoded).subspan(prefix + size_t(ans))) == copy(expected_low.asBytes()),
                    "word-sized insertion differs from upstream BitWriter");
            require(codec.scan(ordinal, encoded, 1) == encoded.size() &&
                    codec.encoded_size(ordinal, encoded, 1, {}) == encoded.size(),
                    "wide field decoding or sizing differs");
            rejects([&] { codec.scan(ordinal, std::span(encoded).first(encoded.size() - 1), 1); });
        }

        rejects([&] { codec.scan(ordinal, {}, 1); });
        rejects([&] { codec.scan(ordinal, std::span(full).first(full.size() - 1), 2); });
        rejects([&] { codec.scan(ordinal, one, 2); });
        rejects([&] { codec.scan(ordinal, full, 1); });
        auto padding = one;
        padding.back() |= 0x80;
        rejects([&] { codec.scan(ordinal, padding, 1); });
        auto trailing = one;
        trailing.push_back(0);
        rejects([&] { codec.encode(ordinal, trailing, 1, {&b, 1}); });
        rejects([&] { codec.encoded_size(ordinal, trailing, 1, {&b, 1}); });
        rejects([&] { codec.encoded_size(ordinal, std::span(one).first(one.size() - 1), 1, {&b, 1}); });
        rejects([&] { codec.encoded_size(ordinal, padding, 1, {&b, 1}); });
        std::span<uint64_t const> empty;
        rejects([&] { codec.encode(ordinal, one, 1, {&empty, 1}); });
        rejects([&] { codec.encoded_size(ordinal, one, 1, {&empty, 1}); });
        auto unsorted = first;
        std::swap(unsorted[0], unsorted[1]);
        std::span<uint64_t const> bad = unsorted;
        rejects([&] { codec.encode(ordinal, {}, 0, {&bad, 1}); });
        rejects([&] { codec.encoded_size(ordinal, {}, 0, {&bad, 1}); });
        rejects([&] { ChunkCodec::fresh_bound(k, ordinal, {&bad, 1}); });
        bad = std::span(first).last(1);
        rejects([&] { codec.encode(ordinal + 1, {}, 0, {&bad, 1}); });
        rejects([&] { ChunkCodec::fresh_bound(k, ordinal + 1, {&bad, 1}); });
        auto wide = first;
        std::fill(wide.begin(), wide.end(), (ordinal + 1) * range - 1);
        wide.front() = ordinal * range;
        bad = wide;
        rejects([&] { ChunkCodec::fresh_bound(k, ordinal, {&bad, 1}); });
        std::fill(wide.begin(), wide.end(), ordinal * range + 7);
        std::vector<uint64_t> far(64, (ordinal + 1) * range - 1);
        std::array<std::span<uint64_t const>, 2> transition{wide, far};
        rejects([&] { ChunkCodec::fresh_bound(k, ordinal, transition); });
        rejects([&] { codec.encode(ordinal, one, UINT16_MAX, {&b, 1}); });
        rejects([&] { codec.encoded_size(ordinal, one, UINT16_MAX, {&b, 1}); });
        rejects([&] { codec.scan(PlotGroupFile::getChunkCountForK(k), one, 1); });
        require(codec.scratch_bytes() <= budget, "codec exceeded admitted scratch");

        ParallelEncoder pool(k, 2, budget);
        std::array<ChunkInput, 5> chunks;
        for (auto& chunk : chunks) chunk = {ordinal, one, 1, {&b, 1}};
        std::array<uint64_t, 5> sizes{};
        pool.measure(chunks, sizes);
        for (auto const size : sizes) require(size == full.size(), "parallel sizing differs");
        auto blocks = pool.encode(chunks, sizes);
        for (auto const& block : blocks)
        {
            size_t offset = 0;
            for (size_t i = block.begin; i < block.end; ++i)
            {
                require(sizes[i] == full.size() && offset <= block.bytes.size() &&
                    full.size() <= block.bytes.size() - offset &&
                    std::equal(full.begin(), full.end(), block.bytes.begin() + offset),
                    "parallel output differs");
                offset += sizes[i];
            }
            require(offset == block.bytes.size(), "coalesced block size differs");
        }
        chunks[0].new_members = {&empty, 1};
        rejects([&] { pool.encode(chunks, sizes); });
        rejects([&] { pool.measure(chunks, sizes); });
        chunks[0].new_members = {&b, 1};
        pool.measure(chunks, sizes);
        pool.encode(chunks, sizes); // Failed sibling tasks have drained safely.
        require(pool.scratch_bytes() <= budget, "parallel codec exceeded scratch");

        // Copy descriptors, then borrow their bytes while the same codecs
        // produce their next blocks. Every retained output is still budgeted.
        pool.encode(chunks, sizes, true);
        auto const borrowed = pool.encode(chunks, sizes, true);
        std::vector<EncodedBlock> stable(borrowed.begin(), borrowed.end());
        std::vector<std::vector<uint8_t>> expected;
        for (auto const& block : stable) expected.push_back(copy(block.bytes));
        auto future = std::async(std::launch::async, [&] { pool.encode(chunks, sizes, true); });
        future.get();
        for (size_t i = 0; i < stable.size(); ++i)
            require(copy(stable[i].bytes) == expected[i], "overlap reused borrowed output");
        require(pool.scratch_bytes() <= budget, "overlap exceeded aggregate scratch");
        auto const complete = pool.encode(chunks, sizes, true);
        stable.assign(complete.begin(), complete.end());
        expected.clear();
        for (auto const& block : stable) expected.push_back(copy(block.bytes));
        chunks[0].new_members = {&empty, 1};
        rejects([&]
        {
            auto failed = std::async(std::launch::async, [&] { pool.encode(chunks, sizes, true); });
            failed.get();
        });
        for (size_t i = 0; i < stable.size(); ++i)
            require(copy(stable[i].bytes) == expected[i], "failed overlap reused borrowed output");
        chunks[0].new_members = {&b, 1};
        pool.encode(chunks, sizes, true);
    }
    rejects([] { ChunkCodec codec(17, 1ull << 24); });
    rejects([] { ChunkCodec codec(18, 1); });
    rejects([] { ParallelEncoder pool(18, 0, 1ull << 24); });
    rejects([] { ChunkCodec::estimate_work(18, UINT64_MAX, 1); });
    rejects([] { ChunkCodec::fresh_bound(18, 0, {}); });
    rejects([] { ChunkCodec::fresh_summary(18, 0, {}); });
    rejects([] { check_gsz_size(UINT64_MAX, 1); });
    ChunkCodec small(18, 1ull << 16);
    std::vector<uint8_t> malformed(1ull << 18, 0xff);
    malformed[0] = 0x80;
    malformed[1] = 0x80;
    malformed[2] = 0x04;
    rejects([&] { small.scan(0, malformed, 1); });
    malformed.assign(10, 0x80);
    rejects([&] { small.scan(0, malformed, 1); });
    require(small.scratch_bytes() <= (1ull << 16), "malformed input exceeded scratch");
}

void summaries()
{
    for (uint16_t members : {uint16_t(1), uint16_t(64), uint16_t(UINT16_MAX)})
    {
        uint64_t const expected = gsz_expected_for_g(members);
        unsigned const rice = unsigned(gsz_rice_k_for_g(members));
        uint64_t const magnitude = (uint64_t(63 - rice) << rice) - 1;
        check_gsz_size(expected + magnitude, members);
        rejects([&] { check_gsz_size(expected + magnitude + 1, members); });
        if (expected > magnitude + 1)
        {
            check_gsz_size(expected - magnitude, members);
            rejects([&] { check_gsz_size(expected - magnitude - 1, members); });
        }
    }
    std::mt19937_64 random(912844);
    for (uint8_t k = 18; k <= 28; k += 2)
    {
        ChunkCodec codec(k, 1ull << 24);
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS), chunk = 3;
        for (unsigned trial = 0; trial < 40; ++trial)
        {
            std::vector<std::vector<uint64_t>> members(1 + random() % 8);
            std::vector<std::span<uint64_t const>> regions;
            ChunkSummary joined;
            for (auto& member : members)
            {
                for (uint64_t i = 0; i < 64; ++i)
                    member.push_back(chunk * range + i * (range / 64) + random() % (range / 64));
                member.insert(member.begin() + 5, member[4]);
                regions.push_back(member);
                (void)codec.encoded_size(chunk, {}, 0, std::span(regions).last(1));
                joined = codec.combine(joined, codec.summary(chunk, 1), member.front(), chunk);
            }
            auto encoded = copy(codec.encode(chunk, {}, 0, regions));
            require(codec.summary(chunk, uint16_t(members.size())) == joined,
                    "additive summary differs from joined canonical chunk");
            require(ChunkCodec::fresh_summary(k, chunk, regions) == joined,
                    "unencoded delta summary differs from canonical chunk");
            auto const [least, most] = ChunkCodec::size_bounds(joined);
            require(least <= encoded.size() && encoded.size() <= most, "summary bounds exclude canonical encoding");
            std::vector<uint64_t> decoded; std::vector<size_t> boundaries;
            decoded.reserve(members.size() * 64); boundaries.reserve(members.size() + 1);
            codec.decode_members(chunk, encoded, uint16_t(members.size()), decoded, boundaries);
            boundaries.push_back(decoded.size());
            require(boundaries.size() == members.size() + 1, "decoded summary member count differs");
            for (size_t i = 0; i < members.size(); ++i)
            {
                auto expected = members[i]; expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
                require(std::vector<uint64_t>(decoded.begin() + boundaries[i], decoded.begin() + boundaries[i + 1]) == expected,
                        "decoded member regions differ from canonical input");
            }
            require(codec.summary(chunk, uint16_t(members.size())) == joined, "decoded summary differs from encoded summary");
        }
    }
}

void joined_chunks()
{
    std::mt19937_64 random(612483);
    for (uint8_t k = 18; k <= 28; k += 2)
    {
        constexpr uint64_t budget = 1ull << 24;
        ChunkCodec codec(k, budget);
        uint64_t const chunk = 3, range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        uint64_t const threshold = 178ull << (k - 8), start = chunk * range;
        for (uint64_t alignment = 0; alignment < 8; ++alignment)
        {
            std::vector<std::vector<uint64_t>> members(5);
            std::vector<FragmentRegion> regions;
            for (size_t member = 0; member < members.size(); ++member)
            {
                uint64_t const first = start + alignment * threshold + member * 3;
                uint64_t const step = (range * 3 / 4 - alignment * threshold) / 64;
                for (uint64_t i = 0; i < 64; ++i)
                {
                    members[member].push_back(first + i * step + (i ? random() % (step / 2) : 0));
                    if (i == 3) members[member].push_back(members[member].back());
                }
                regions.emplace_back(std::span<uint64_t const>(members[member]));
            }
            auto const expected = copy(codec.encode(chunk, {}, 0, regions));
            auto const summary = codec.summary(chunk, uint16_t(regions.size()));
            for (size_t group : {size_t(1), size_t(2), members.size()})
            {
                std::vector<std::vector<uint8_t>> storage;
                std::vector<EncodedChunk> contributions;
                storage.reserve(members.size()); contributions.reserve(members.size());
                for (size_t first = 0; first < members.size(); first += group)
                {
                    size_t const count = std::min(group, members.size() - first);
                    storage.push_back(copy(codec.encode(chunk, {}, 0, std::span(regions).subspan(first, count))));
                    contributions.push_back({storage.back(), uint16_t(count)});
                }
                require(copy(codec.join(chunk, contributions)) == expected &&
                    codec.summary(chunk, uint16_t(members.size())) == summary, "joined contribution bytes or summary differ");
                require(codec.joined_size(chunk, contributions) == expected.size() && copy(codec.output()) == expected,
                        "joined exact sizing differs or replaced output");
                codec.clear_output();
                codec.append_join(chunk, contributions); codec.append_join(chunk, contributions);
                auto repeated = expected; repeated.insert(repeated.end(), expected.begin(), expected.end());
                require(copy(codec.output()) == repeated, "coalesced joined output differs");
                uint64_t source = 0;
                for (auto const& part : contributions) source += part.bytes.size();
                uint64_t const admitted = ChunkCodec::estimate_work(k, source, 0);
                ChunkCodec bounded(k, admitted);
                require(copy(bounded.join(chunk, contributions)) == expected && bounded.scratch_bytes() <= admitted,
                        "joined contribution exceeded source-byte scratch admission");
                ChunkCodec tight(k, ChunkCodec::estimate_base());
                rejects([&] { tight.join(chunk, contributions); });
                require(tight.scratch_bytes() <= ChunkCodec::estimate_base(), "rejected join exceeded scratch budget");
                uint64_t rejected = ChunkCodec::estimate_base(), accepted = admitted;
                while (accepted - rejected > 1)
                {
                    uint64_t const trial = rejected + (accepted - rejected) / 2;
                    ChunkCodec limited(k, trial);
                    std::vector<uint8_t> result;
                    bool fits = false;
                    try
                    {
                        result = copy(limited.join(chunk, contributions));
                        fits = true;
                    }
                    catch (std::runtime_error const&) {}
                    if (fits) { require(result == expected, "tight joined bytes differ"); accepted = trial; }
                    else rejected = trial;
                    require(limited.scratch_bytes() <= trial, "tight joined allocation exceeded admission");
                }
                ChunkCodec limited(k, accepted);
                require(copy(limited.join(chunk, contributions)) == expected &&
                    limited.joined_size(chunk, contributions) == expected.size(), "tight joined emission or sizing differs");
                ChunkCodec too_small(k, rejected);
                rejects([&] { too_small.join(chunk, contributions); });

                ParallelEncoder encoder(k, 2, budget);
                std::array<ChunkInput, 3> tasks;
                for (auto& task : tasks) task = {chunk, {}, 0, {}, contributions};
                std::array<uint64_t, 3> sizes{};
                std::array<ChunkSummary, 3> summaries{};
                encoder.measure(tasks, sizes, summaries);
                auto blocks = encoder.encode(tasks, sizes, false, summaries);
                for (auto const& block : blocks)
                    for (size_t i = block.begin, offset = 0; i < block.end; offset += sizes[i++])
                        require(sizes[i] == expected.size() && summaries[i] == summary &&
                            std::equal(expected.begin(), expected.end(), block.bytes.begin() + offset),
                            "parallel joined output differs");
                auto corrupt = storage.front(); corrupt.push_back(0);
                auto bad = contributions; bad.front().bytes = corrupt;
                rejects([&] { codec.join(chunk, bad); });
                rejects([&] { codec.joined_size(chunk, bad); });
                codec.scan(chunk, storage.front(), contributions.front().members);
                if (codec.summary(chunk, contributions.front().members).low_bits & 7)
                {
                    corrupt = storage.front(); corrupt.back() |= 0x80;
                    bad.front().bytes = corrupt;
                    rejects([&] { codec.join(chunk, bad); });
                    rejects([&] { codec.joined_size(chunk, bad); });
                }
                bad.front().bytes = std::span(storage.front()).first(storage.front().size() - 1);
                rejects([&] { codec.join(chunk, bad); });
                rejects([&] { codec.joined_size(chunk, bad); });
                bad = contributions; ++bad.front().members;
                rejects([&] { codec.join(chunk, bad); });
                rejects([&] { codec.joined_size(chunk, bad); });
                bad = contributions; bad.front().members = 0;
                rejects([&] { codec.joined_size(chunk, bad); });
                bad = contributions; bad.front().members = UINT16_MAX;
                rejects([&] { codec.join(chunk, bad, regions); });
                tasks[0].contributions = bad;
                rejects([&] { encoder.encode(tasks, sizes); });
                tasks[0].contributions = contributions;
                encoder.encode(tasks, sizes);
            }
            auto const prefix = copy(codec.encode(chunk, {}, 0, std::span(regions).first(3)));
            EncodedChunk const old{prefix, 3};
            require(copy(codec.join(chunk, {&old, 1}, std::span(regions).last(2))) == expected &&
                codec.joined_size(chunk, {&old, 1}, std::span(regions).last(2)) == expected.size(),
                "joined ordinary prefix plus incoming members differs");
            uint64_t const incoming = members[3].size() + members[4].size();
            uint64_t const admitted = ChunkCodec::estimate_work(k, prefix.size(), incoming);
            ChunkCodec bounded(k, admitted);
            require(copy(bounded.join(chunk, {&old, 1}, std::span(regions).last(2))) == expected &&
                bounded.joined_size(chunk, {&old, 1}, std::span(regions).last(2)) == expected.size() &&
                bounded.scratch_bytes() <= admitted, "joined incoming members exceeded scratch admission");
            ParallelEncoder summaries(k, 2, budget);
            std::array<ChunkInput, 3> fresh;
            for (auto& task : fresh) task = {chunk, {}, 0, regions};
            std::array<ChunkSummary, 3> details{};
            std::array<uint64_t, 3> sizes{};
            auto const blocks = summaries.encode(fresh, sizes);
            std::vector<std::vector<uint8_t>> stable;
            for (auto const& block : blocks) stable.push_back(copy(block.bytes));
            std::vector<EncodedBlock> borrowed(blocks.begin(), blocks.end());
            summaries.summarize(fresh, details);
            for (auto const& detail : details) require(detail == summary, "parallel fresh summary differs");
            summaries.summarize(std::span(fresh).first(1), std::span(details).first(1));
            for (size_t i = 0; i < borrowed.size(); ++i)
                require(blocks[i].begin == borrowed[i].begin && blocks[i].end == borrowed[i].end &&
                    copy(blocks[i].bytes) == stable[i], "fresh summary replaced encoded output");
            fresh[0].old_chunk = prefix;
            rejects([&] { summaries.summarize(fresh, details); });
            fresh[0].old_chunk = {};
            summaries.summarize(fresh, details);
        }
        // Boundary correction can itself occupy a complete machine word.
        // Vary the preceding symbol count so that insertion crosses each
        // destination byte alignment, and reject the next wider field.
        for (size_t alignment = 0; alignment < 8; ++alignment)
        {
            uint64_t const maximum = uint64_t(72 - k) * threshold - 1;
            std::vector<uint64_t> previous, next;
            for (size_t i = 0; i < 64 + alignment; ++i)
                previous.push_back(start + uint64_t(i) * (range - 1) / (63 + alignment));
            for (uint64_t i = 0; i < 64; ++i) next.push_back(start + maximum - 1 + i);
            std::array<FragmentRegion, 2> regions{std::span<uint64_t const>(previous), std::span<uint64_t const>(next)};
            auto const expected = copy(codec.encode(chunk, {}, 0, regions));
            auto const first = copy(codec.encode(chunk, {}, 0, std::span(regions).first(1)));
            auto const second = copy(codec.encode(chunk, {}, 0, std::span(regions).last(1)));
            std::array<EncodedChunk, 2> contributions{{{first, 1}, {second, 1}}};
            require(copy(codec.join(chunk, contributions)) == expected && codec.joined_size(chunk, contributions) == expected.size(),
                    "maximum-width joined boundary differs");
            uint64_t const admitted = ChunkCodec::estimate_work(k, first.size() + second.size(), 0);
            ChunkCodec bounded(k, admitted);
            require(copy(bounded.join(chunk, contributions)) == expected && bounded.scratch_bytes() <= admitted,
                    "maximum-width source exceeded joined scratch admission");
            for (auto& value : next) ++value;
            auto const wide = copy(codec.encode(chunk, {}, 0, std::span(regions).last(1)));
            contributions[1].bytes = wide;
            rejects([&] { codec.join(chunk, contributions); });
            rejects([&] { codec.joined_size(chunk, contributions); });
        }
        rejects([&] { codec.join(0, {}); });
    }
}

void compact_regions()
{
    FragmentBoundaries boundaries;
    boundaries.reset(2, UINT32_MAX);
    boundaries.set(2, UINT32_MAX);
    require(boundaries.narrow() && boundaries[2] == UINT32_MAX &&
            boundaries.storage_bytes() == FragmentBoundaries::allocation_bytes(2, UINT32_MAX),
            "narrow fragment indexes changed an extreme count");
    if constexpr (sizeof(size_t) > sizeof(uint32_t))
    {
        size_t const wide = size_t(UINT32_MAX) + 1;
        rejects([&] { boundaries.set(0, wide); });
        boundaries.reset(2, wide);
        boundaries.set(2, wide);
        require(!boundaries.narrow() && boundaries[2] == wide &&
                boundaries.storage_bytes() == FragmentBoundaries::allocation_bytes(2, wide),
                "wide fragment index fallback truncated its count");
    }
    rejects([&] { FragmentBoundaries::allocation_bytes(SIZE_MAX, 1); });
    rejects([&] { Member::compact_bytes(18, SIZE_MAX); });
    CompactMemberBuilder empty_builder(18, 0);
    auto empty = empty_builder.finish();
    uint64_t const small_chunks = PlotGroupFile::getChunkCountForK(18);
    require(empty.empty() && empty.view().region(0).empty() &&
            empty.view().region(small_chunks - 1).empty() &&
            empty.storage_bytes() == Member::compact_bytes(18, 0),
            "empty packed member has invalid chunk boundaries");
    std::array<uint64_t, 3> const short_values{1, 5, 8};
    CompactMemberBuilder trailing_builder(18, short_values.size());
    trailing_builder.append(short_values);
    auto trailing = trailing_builder.finish();
    auto const first_region = trailing.view().region(0);
    require(std::equal(first_region.begin(), first_region.end(), short_values.begin()) &&
            trailing.view().region(1).empty() && trailing.view().region(small_chunks - 1).empty(),
            "trailing empty packed chunks changed member readback");
    for (uint8_t k = 18; k <= 28; ++k)
    {
        if (k & 1)
        {
            rejects([&] { CompactMemberBuilder builder(k, 1); });
            continue;
        }
        uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        std::array<uint64_t, 3> const samples{0, chunks / 2, chunks - 1};
        std::vector<uint64_t> raw;
        for (uint64_t chunk : samples)
            for (uint64_t i = 0; i <= 256; ++i)
            {
                raw.push_back(chunk * range + i * ((range - 1) / 256));
                if (i == 256) raw.back() = (chunk + 1) * range - 1;
                if (i % 19 == 0) raw.push_back(raw.back());
            }
        CompactMemberBuilder builder(k, raw.size());
        for (size_t first = 0; first < raw.size();)
        {
            size_t const count = std::min(raw.size() - first, size_t(17 + first % 73));
            builder.append(std::span(raw).subspan(first, count));
            first += count;
        }
        Member packed = builder.finish();
        require(packed.compact() && packed.size() == raw.size() &&
                packed.storage_bytes() == Member::compact_bytes(k, raw.size()),
                "packed member exceeded its exact retained allocation model");
        rejects([&] { builder.finish(); });
        rejects([&] { builder.append({}); });
        Member borrowed = Member::borrowed(raw);
        require(borrowed.storage_bytes() == 0 && borrowed.view().raw().data() == raw.data(),
                "borrowed member unexpectedly copied its payload");
        Member borrowed_moved = std::move(borrowed);
        require(borrowed.empty() && borrowed.storage_bytes() == 0 &&
                borrowed_moved.view().raw().data() == raw.data() && borrowed_moved.size() == raw.size(),
                "moving a borrowed member changed its lease view");
        borrowed = std::move(borrowed_moved);
        Member owned{std::vector<uint64_t>(raw)};
        auto const owned_data = owned.view().raw().data();
        Member owned_moved = std::move(owned);
        require(owned.empty() && owned.storage_bytes() == 0 &&
                owned_moved.view().raw().data() == owned_data && owned_moved.size() == raw.size(),
                "moving a raw member changed its owned allocation");
        owned = std::move(owned_moved);
        owned.compact(k);
        Member moved = std::move(owned);
        require(owned.empty() && owned.storage_bytes() == 0 && !owned.view().immutable_id() &&
                moved.compact() && moved.view().immutable_id(),
                "moved member retained an invalid compact view");
        ChunkCodec codec(k, 1ull << 24);
        for (uint64_t chunk : samples)
        {
            auto const first = std::lower_bound(raw.begin(), raw.end(), chunk * range);
            auto const last = std::lower_bound(first, raw.end(), (chunk + 1) * range);
            auto const expected = std::span<uint64_t const>(first, last);
            auto const region = packed.view().region(chunk);
            auto const converted = moved.view().region(chunk);
            require(region.validated_deltas(k) && !FragmentRegion(expected).validated_deltas(k),
                    "compact construction did not preserve its trusted delta invariant");
            require(region.size() == expected.size() && std::equal(region.begin(), region.end(), expected.begin()) &&
                    std::equal(converted.begin(), converted.end(), expected.begin()),
                    "packed member failed exact cross-word fragment readback");
            auto const canonical = copy(codec.encode(chunk, {}, 0, {&expected, 1}));
            require(ChunkCodec::fresh_summary(k, chunk, {&region, 1}) == codec.summary(chunk, 1),
                    "compact delta summary differs from canonical chunk");
            require(copy(codec.encode(chunk, {}, 0, {&region, 1})) == canonical &&
                    codec.encoded_size(chunk, {}, 0, {&region, 1}) == canonical.size() &&
                    ChunkCodec::fresh_bound(k, chunk, {&region, 1}) ==
                        ChunkCodec::fresh_bound(k, chunk, {&expected, 1}),
                    "packed/raw regions changed canonical bytes or sizing");
            std::array<std::span<uint64_t const>, 2> const raw_pair{expected, expected};
            std::array<FragmentRegion, 2> const compact_pair{region, region};
            auto const combined = copy(codec.encode(chunk, {}, 0, raw_pair));
            require(copy(codec.encode(chunk, {}, 0, compact_pair)) == combined &&
                    copy(codec.encode(chunk, canonical, 1, {&region, 1})) == combined &&
                    codec.encoded_size(chunk, canonical, 1, {&region, 1}) == combined.size() &&
                    copy(codec.output()) == combined,
                    "compact bounds changed combined bytes, append bytes, or size-only output");
        }
        rejects([&] { packed.view().region(chunks); });
        rejects([&] { packed.view().region(0, 1, 1); });
        rejects([&] { borrowed.view().region(0, raw.size(), 1); });
        rejects([&] { packed.compact(k == 18 ? 20 : 18); });
        CompactMemberBuilder short_member(k, 2);
        short_member.append(std::span(raw).first(1));
        rejects([&] { short_member.finish(); });
        rejects([&] { short_member.append(std::span(raw).first(2)); });
        CompactMemberBuilder invalid(k, 2);
        std::array<uint64_t, 2> const unsorted{2, 1};
        rejects([&] { invalid.append(unsorted); });
        CompactMemberBuilder invalid_tile(k, unsorted.size());
        invalid_tile.append(std::span(unsorted).first(1));
        rejects([&] { invalid_tile.append(std::span(unsorted).last(1)); });
        rejects([&] { invalid_tile.finish(); });
        CompactMemberBuilder extreme(k, 1);
        uint64_t const out_of_range = 1ull << (2 * k);
        rejects([&] { extreme.append({&out_of_range, 1}); });
        uint64_t const maximum = UINT64_MAX;
        rejects([&] { extreme.append({&maximum, 1}); });
        uint64_t const maximum_delta = uint64_t(72 - k) * (178ull << (k - 8));
        std::array<uint64_t, 2> const wide_gap{0, maximum_delta};
        CompactMemberBuilder wide_builder(k, wide_gap.size());
        wide_builder.append(wide_gap);
        auto wide = wide_builder.finish();
        auto const wide_region = wide.view().region(0);
        require(!wide_region.validated_deltas(k), "wide compact delta was incorrectly trusted");
        rejects([&] { ChunkCodec::fresh_bound(k, 0, {&wide_region, 1}); });
        rejects([&] { codec.encode(0, {}, 0, {&wide_region, 1}); });
        rejects([&] { codec.encoded_size(0, {}, 0, {&wide_region, 1}); });
        rejects([&] { ChunkCodec::fresh_summary(k, 0, {&wide_region, 1}); });
        std::array<uint64_t, 2> const high_start{range - 1, range - 1};
        CompactMemberBuilder high_builder(k, high_start.size());
        high_builder.append(high_start);
        auto high = high_builder.finish();
        auto const high_region = high.view().region(0);
        require(high_region.validated_deltas(k), "duplicate compact values lost delta validation");
        rejects([&] { ChunkCodec::fresh_bound(k, 0, {&high_region, 1}); });
        rejects([&] { codec.encode(0, {}, 0, {&high_region, 1}); });
        rejects([&] { codec.encoded_size(0, {}, 0, {&high_region, 1}); });
        rejects([&] { ChunkCodec::fresh_summary(k, 0, {&high_region, 1}); });
    }
}

void compact_word_tails()
{
    for (uint8_t k = 18; k <= 28; k += 2)
    {
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        std::vector<uint64_t> raw;
        for (uint64_t index = 0; index < 129; ++index)
        {
            uint64_t const low = index % 7 == 6 ? range - 1 : (index % 7) * ((range - 1) / 6);
            raw.push_back((index / 7) * range + low);
        }
        for (size_t count : {0, 1, 2, 3, 31, 32, 33, 65, 129})
        {
            CompactMemberBuilder builder(k, count);
            builder.append({});
            for (size_t first = 0; first < count;)
            {
                size_t const tile = std::min(count - first, size_t(1 + count % 17));
                builder.append(std::span(raw).subspan(first, tile));
                builder.append({});
                first += tile;
            }
            auto member = builder.finish();
            size_t position = 0;
            uint64_t const chunks = count ? raw[count - 1] / range + 1 : 1;
            for (uint64_t chunk = 0; chunk < chunks; ++chunk)
                for (uint64_t value : member.view().region(chunk))
                {
                    require(position < raw.size() && value == raw[position],
                            "packed word tail changed tiled fragment readback");
                    ++position;
                }
            require(position == count && member.storage_bytes() == Member::compact_bytes(k, count) &&
                    member.view().region(PlotGroupFile::getChunkCountForK(k) - 1).empty(),
                    "packed word tail changed count or retained allocation");
        }
    }
}

void compact_received_tiles()
{
    for (uint8_t k = 18; k <= 28; k += 2)
    {
        size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(k));
        unsigned const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        uint64_t const range = 1ull << width;
        for (size_t count : {0, 1, 2, 3, 63, 64, 65, 129})
        {
            // Leave leading, interior and trailing empty chunks. Duplicates
            // and high fields exercise cross-word tails without large deltas.
            std::vector<uint64_t> raw;
            for (size_t index = 0; index < count; ++index)
            {
                size_t const chunk = index < 43 ? 1 : (index < 86 ? chunks / 2 : chunks - 1);
                raw.push_back(chunk * range + (index % 43 / 2) * (range / 64));
            }
            BitWriter canonical;
            for (uint64_t value : raw) canonical.append(value & (range - 1), width);
            std::vector<uint64_t> words((canonical.bitCount() + 63) / 64);
            if (!words.empty()) std::memcpy(words.data(), canonical.asBytes().data(), canonical.asBytes().size());
            std::vector<uint32_t> boundaries(chunks + 1);
            size_t first = 0;
            for (size_t chunk = 0; chunk <= chunks; ++chunk)
            {
                while (first < raw.size() && (raw[first] >> width) < chunk) ++first;
                boundaries[chunk] = uint32_t(first);
            }
            CompactMemberReceiver receiver(k, count);
            receiver.append({boundaries, {}, {count, 0}});
            for (size_t word = 0; word < words.size();)
            {
                size_t const tile = std::min(words.size() - word, size_t(1 + word % 7));
                receiver.append({{}, std::span(words).subspan(word, tile), {}});
                word += tile;
            }
            auto received = receiver.finish();
            CompactMemberBuilder builder(k, raw.size());
            builder.append(raw);
            auto packed = builder.finish();
            require(received.storage_bytes() == Member::compact_bytes(k, count) &&
                    received.view().immutable_id() != packed.view().immutable_id(),
                    "compact receiver changed exact storage or reused a member identity");
            ChunkCodec codec(k, 1ull << 24);
            for (size_t chunk = 0; chunk < chunks; ++chunk)
            {
                auto const actual = received.view().region(chunk);
                auto const expected = packed.view().region(chunk);
                require(actual.size() == expected.size() &&
                        std::equal(actual.begin(), actual.end(), expected.begin()) && actual.validated_deltas(k),
                        "compact receiver changed boundaries, fields, or validation");
                // The codec removes duplicates before the pinned FSE encoder,
                // which requires more than two symbols.
                size_t distinct = 0;
                uint64_t previous = UINT64_MAX;
                for (uint64_t value : expected)
                    if (value != previous) { ++distinct; previous = value; }
                if (distinct > 2)
                {
                    auto const reference = copy(codec.encode(chunk, {}, 0, {&expected, 1}));
                    require(copy(codec.encode(chunk, {}, 0, {&actual, 1})) == reference,
                            "compact receiver changed ordinary chunk bytes");
                }
            }
            rejects([&] { receiver.finish(); });
            rejects([&] { receiver.append({}); });
        }
    }

    constexpr uint8_t k = 18;
    size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(k));
    std::vector<uint32_t> boundaries(chunks + 1, 1);
    boundaries.front() = 0;
    std::array<uint64_t, 1> words{7};
    auto reject_prelude = [&](PackedFragmentTile const& prelude)
    {
        CompactMemberReceiver receiver(k, 1);
        rejects([&] { receiver.append(prelude); });
        rejects([&] { receiver.append({boundaries, {}, {1, 0}}); });
        rejects([&] { receiver.finish(); });
    };
    reject_prelude({{}, words, {1, 0}});
    reject_prelude({boundaries, {}, {0, 0}});
    reject_prelude({boundaries, {}, {2, 0}});
    for (uint32_t flag : {PackedMemberValidation::out_of_range, PackedMemberValidation::unsorted,
                          PackedMemberValidation::known_flags + 1})
        reject_prelude({boundaries, {}, {1, flag}});
    auto invalid = boundaries;
    invalid.front() = 1;
    reject_prelude({invalid, {}, {1, 0}});
    invalid = boundaries; invalid.back() = 0;
    reject_prelude({invalid, {}, {1, 0}});
    invalid = boundaries; invalid[1] = 2;
    reject_prelude({invalid, {}, {1, 0}});
    invalid = boundaries; invalid[2] = 0;
    reject_prelude({invalid, {}, {1, 0}});
    reject_prelude({std::span(boundaries).first(chunks), {}, {1, 0}});
    CompactMemberReceiver missing(k, 1);
    rejects([&] { missing.finish(); });
    missing.append({boundaries, {}, {1, 0}});
    rejects([&] { missing.finish(); });
    rejects([&] { missing.append({boundaries, {}, {1, 0}}); });
    CompactMemberReceiver excess(k, 1);
    excess.append({boundaries, {}, {1, 0}});
    std::array<uint64_t, 2> const long_payload{0, 0};
    rejects([&] { excess.append({{}, long_payload, {}}); });
    CompactMemberReceiver padded(k, 1);
    padded.append({boundaries, {}, {1, 0}});
    uint64_t const bad_tail = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    padded.append({{}, {&bad_tail, 1}, {}});
    rejects([&] { padded.finish(); });
    CompactMemberReceiver wide_delta(k, 1);
    wide_delta.append({boundaries, {}, {1, PackedMemberValidation::wide_delta}});
    wide_delta.append({{}, words, {}});
    auto untrusted = wide_delta.finish();
    require(!untrusted.view().region(0).validated_deltas(k), "compact receiver trusted an invalid device delta bound");
    // An authenticated remote peer may still declare a successful validation
    // for invalid words. The network receiver inspects the actual payload.
    std::vector<uint32_t> pair_boundaries(chunks + 1, 2);
    pair_boundaries.front() = 0;
    uint64_t const reversed = 9 | (uint64_t(7) << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS));
    CompactMemberReceiver remote(k, 2, false);
    remote.append({pair_boundaries, {}, {2, 0}});
    remote.append({{}, {&reversed, 1}, {}});
    rejects([&] { remote.finish(); });
    rejects([&] { remote.finish(); });
    if constexpr (sizeof(size_t) > sizeof(uint32_t))
        rejects([&] { CompactMemberReceiver too_wide(k, size_t(UINT32_MAX) + 1); });
}

// Conservative compact bounds must fall back to exact sizing when repeated
// fragments would reserve more symbols than fit. Keep coalesced and borrowed
// output live so that the fallback sees the same retained allocations.
void compact_bound_fallback()
{
    for (uint8_t k = 18; k <= 28; k += 2)
    {
        uint64_t const ordinal = 3;
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        std::vector<uint64_t> raw;
        for (uint64_t i = 0; i < 64; ++i)
            for (size_t repeat = 0; repeat < 64; ++repeat)
                raw.push_back(ordinal * range + i * (range / 64) + 7);
        CompactMemberBuilder builder(k, raw.size());
        for (size_t first = 0; first < raw.size();)
        {
            size_t const count = std::min(raw.size() - first, size_t(73));
            builder.append(std::span(raw).subspan(first, count));
            first += count;
        }
        auto member = builder.finish();
        auto const region = member.view().region(ordinal);
        require(region.validated_deltas(k), "duplicate compact fixture was not trusted");
        std::span<uint64_t const> values = raw;
        uint64_t const budget = ChunkCodec::estimate_base() + raw.size();
        ChunkCodec exact(k, budget), compact(k, budget);
        auto const first = copy(exact.encode(ordinal, {}, 0, {&values, 1}));
        require(copy(compact.encode(ordinal, {}, 0, {&region, 1})) == first,
                "compact bound rejected or changed an exact-fit duplicate chunk");
        require(ChunkCodec::fresh_summary(k, ordinal, {&region, 1}) == compact.summary(ordinal, 1),
                "duplicate-heavy delta summary differs from exact compact output");
        auto const borrowed = compact.output();
        compact.next_window();
        exact.next_window();
        auto const appended = copy(exact.encode(ordinal, first, 1, {&values, 1}));
        require(copy(compact.encode(ordinal, first, 1, {&region, 1})) == appended &&
                copy(borrowed) == first,
                "compact fallback changed incremental or borrowed output");
        exact.append_encode(ordinal, first, 1, {&values, 1});
        compact.append_encode(ordinal, first, 1, {&region, 1});
        auto const coalesced = copy(exact.output());
        require(copy(compact.output()) == coalesced &&
                compact.encoded_size(ordinal, first, 1, {&region, 1}) == appended.size() &&
                copy(compact.output()) == coalesced && copy(borrowed) == first,
                "compact fallback sizing changed retained output");
        CompactMemberBuilder empty_builder(k, 0);
        auto empty = empty_builder.finish();
        auto const empty_region = empty.view().region(ordinal);
        rejects([&] { ChunkCodec::fresh_summary(k, ordinal, {&empty_region, 1}); });
        rejects([&] { compact.append_encode(ordinal, first, 1, {&empty_region, 1}); });
        require(copy(compact.output()) == coalesced && copy(borrowed) == first &&
                compact.scratch_bytes() <= budget,
                "failed compact fallback changed output or exceeded scratch");
    }
}

void batch_hashing()
{
    constexpr uint8_t k = 18;
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::array<std::vector<uint64_t>, 3> raw{
        std::vector<uint64_t>{0, 0, 1, 31, range - 1, range + 5},
        std::vector<uint64_t>{2, 3, 3, range + 7, range + 9},
        std::vector<uint64_t>{4, 4, range + 1, range + 2}};
    MemberBatchHash hash(raw.size());
    auto build = [&](size_t member, MemberBatchHash* cache)
    {
        CompactMemberBuilder builder(k, raw[member].size(), cache);
        builder.append(std::span(raw[member]).first(1));
        builder.append(std::span(raw[member]).subspan(1));
        return builder.finish();
    };
    std::vector<Member> members;
    members.reserve(raw.size());
    members.push_back(build(0, &hash));
    auto const first_digest = hash.digest_for(members);
    require(first_digest && *first_digest == framed_digest(std::span(raw).first(1)),
            "fused member hash changed the native logical framing");
    members.push_back(build(1, &hash));
    require(hash.digest_for(members) == std::optional(framed_digest(std::span(raw).first(2))) &&
            hash.digest_for(Members(members).subspan(0, 1)) == first_digest,
            "batch hash did not preserve full and reduced prefixes");
    require(!hash.digest_for(std::span(raw).first(2)), "mutable raw data reused an immutable hash");
    std::array<Member, 2> borrowed{Member::borrowed(raw[0]), Member::borrowed(raw[1])};
    require(!hash.digest_for(borrowed), "borrowed raw data reused an immutable hash");
    auto moved = std::move(members);
    uint64_t const previous_id = moved[1].view().immutable_id();
    require(hash.digest_for(moved).has_value(), "moving compact ownership invalidated its hash identity");
    moved[1] = build(1, nullptr); // Reuse the same Member object address.
    require(moved[1].view().immutable_id() != previous_id && !hash.digest_for(moved),
            "replacement compact data reused a stale hash identity");
    moved.pop_back();
    hash.truncate(moved.size());
    require(hash.digest_for(moved) == first_digest, "truncation changed a committed prefix hash");
    rejects([&] { hash.truncate(2); });
    {
        CompactMemberBuilder failed(k, 3, &hash);
        std::array<uint64_t, 3> const invalid{1, 5, 4};
        rejects([&] { failed.append(invalid); });
        rejects([&] { failed.finish(); });
        rejects([&] { hash.reset(); });
        require(hash.size() == moved.size() && hash.digest_for(moved) == first_digest,
                "failed construction committed a partial batch hash");
    }
    moved.push_back(build(2, &hash));
    std::array<std::vector<uint64_t>, 2> const extended{raw[0], raw[2]};
    require(hash.digest_for(moved) == std::optional(framed_digest(extended)),
            "truncate-and-extend combined hashes instead of logical byte streams");
    hash.reset();
    require(!hash.digest_for(moved), "reset retained an obsolete member hash");
    CompactMemberBuilder empty(k, 0, &hash);
    std::array<Member, 1> empty_member{empty.finish()};
    require(hash.digest_for(empty_member) == std::optional(framed_digest(empty_member)),
            "empty member size prefix was omitted from the fused hash");
    rejects([&] { MemberBatchHash::memory_bytes(size_t(UINT16_MAX) + 1); });
    MemberBatchHash no_capacity(0);
    rejects([&] { CompactMemberBuilder builder(k, 0, &no_capacity); });
}

// A valid chunk can fit the aggregate scratch cap while being too large for
// overlap's two-window partition. Admission must fall back before any output
// is borrowed, and a later smaller operation may enable overlap again.
void adaptive_overlap()
{
    constexpr uint8_t k = 18;
    constexpr uint64_t dense_count = 256;
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    Config config;
    config.capacity_bytes = 128ull * 1024 * 1024;
    config.codec_bytes = 76000;
    config.memory_limit_bytes = 512ull * 1024 * 1024;
    config.workers = 2;
    uint64_t const work = ChunkCodec::estimate_work(k, 0, dense_count);
    require(work > config.codec_bytes / 2 && work <= config.codec_bytes,
            "adaptive admission fixture no longer crosses the overlap budget");
    std::vector<std::vector<uint64_t>> first(1), second(1);
    first[0].reserve(size_t(chunks * dense_count));
    second[0].reserve(size_t(chunks * 64));
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        for (uint64_t i = 0; i < dense_count; ++i)
            first[0].push_back(chunk * range + i * (range / dense_count) + 7);
        for (uint64_t i = 0; i < 64; ++i)
            second[0].push_back(chunk * range + i * (range / 64) + 11);
    }
    std::string name = (std::filesystem::temp_directory_path() / "grow-admission-XXXXXX").string();
    require(::mkdtemp(name.data()) != nullptr, "cannot create admission test directory");
    struct Cleanup
    {
        std::string path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{name};
    Identity identity;
    identity.header.magic = PlotGroupFile::MAGIC;
    identity.header.k = k;
    identity.header.strength = 2;
    for (bool overlap : {true, false})
    {
        config.io_overlap = overlap;
        Engine engine(name + (overlap ? "/adaptive.gplot" : "/reference.gplot"), config, false, identity);
        engine.confirm_identity();
        auto fresh = engine.plan_fresh(first);
        require(fresh && fresh->accepted, "large valid chunk was not admitted");
        require(fresh->members_digest == framed_digest(first),
                "engine member digest changed its original size-prefix framing");
        engine.apply(*fresh, first);
        require(engine.stats().encode_chunks == chunks && !engine.stats().overlap_windows,
                "large chunk did not use a single bounded serial encoding pass");
        auto append = engine.plan(second);
        require(append.accepted, "subsequent smaller operation was not admitted");
        engine.apply(append, second);
        require(bool(engine.stats().overlap_windows) == overlap,
                "smaller operation did not restore requested overlap");
    }
    std::ifstream adaptive(name + "/adaptive.gplot", std::ios::binary);
    std::ifstream reference(name + "/reference.gplot", std::ios::binary);
    require(std::filesystem::file_size(name + "/adaptive.gplot") ==
            std::filesystem::file_size(name + "/reference.gplot") &&
            std::equal(std::istreambuf_iterator<char>(adaptive), {}, std::istreambuf_iterator<char>(reference)),
            "adaptive overlap output differs from serial encoding");
}

// Vary region sizes and batch membership so retained indexes must be rebuilt,
// including chunks beginning exactly at a boundary and duplicate fragments.
void indexed_members()
{
    constexpr uint8_t k = 18;
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<std::vector<uint64_t>> members(5);
    for (size_t member = 0; member < members.size(); ++member)
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
        {
            uint64_t const count = 64 + (chunk + member) % 7;
            for (uint64_t i = 0; i < count; ++i)
            {
                uint64_t const value = chunk * range + i * (range / count) + member;
                members[member].push_back(value);
                if (i == 3) members[member].push_back(value);
            }
        }
    std::string name = (std::filesystem::temp_directory_path() / "grow-indexed-XXXXXX").string();
    require(::mkdtemp(name.data()) != nullptr, "cannot create indexed-member test directory");
    struct Cleanup
    {
        std::string path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{name};
    Config config;
    config.capacity_bytes = 128ull * 1024 * 1024;
    config.memory_limit_bytes = 512ull * 1024 * 1024;
    config.workers = 3;
    Identity identity;
    identity.header.magic = PlotGroupFile::MAGIC;
    identity.header.k = k;
    identity.header.strength = 2;
    std::vector<Member> retained;
    retained.reserve(members.size());
    retained.push_back(Member::borrowed(members[0]));
    retained.emplace_back(std::vector<uint64_t>(members[1]));
    for (size_t member = 2; member < members.size(); ++member)
    {
        CompactMemberBuilder builder(k, members[member].size());
        builder.append(members[member]);
        retained.push_back(builder.finish());
    }
    for (bool staged : {false, true})
    {
        Engine engine(name + (staged ? "/staged.gplot" : "/fresh.gplot"), config, false, identity);
        engine.confirm_identity();
        if (!staged)
        {
            CompactMemberBuilder empty_builder(k, 0);
            auto empty = empty_builder.finish();
            rejects([&] { engine.plan_fresh({&empty, 1}); });
            auto invalid = members;
            invalid[0][0] = 1ull << (2 * k);
            rejects([&] { engine.plan_fresh(invalid); });
            invalid = members;
            auto& first = invalid[0];
            first.erase(std::lower_bound(first.begin(), first.end(), range),
                        std::lower_bound(first.begin(), first.end(), 2 * range));
            rejects([&] { engine.plan_fresh(invalid); });
            require(std::filesystem::file_size(name + "/fresh.gplot") == 0,
                    "invalid indexed member changed the destination");
        }
        size_t offset = 0;
        for (size_t count : (staged ? std::vector<size_t>{2, 1, 2} : std::vector<size_t>{5}))
        {
            auto batch = (staged ? Members(members) : Members(retained)).subspan(offset, count);
            if (staged)
            {
                auto const result = engine.append(batch);
                require(result.accepted && result.result.members == offset + count,
                        "closed append did not commit indexed members");
            }
            else
            {
                auto fresh = engine.plan_fresh(batch);
                auto plan = fresh ? std::move(*fresh) : engine.plan(batch);
                require(plan.accepted, "indexed-member batch was not admitted");
                require(plan.members_digest == framed_digest(members),
                        "packed member hashing changed the original logical byte framing");
                ++members[0][0]; // Still sorted, but no longer the planned input.
                rejects([&] { engine.apply(plan, batch); });
                require(std::filesystem::file_size(name + "/fresh.gplot") == 0,
                        "changed planned input mutated the destination");
                --members[0][0];
                auto changed = members[2];
                ++changed[0];
                retained[2] = Member(std::move(changed));
                retained[2].compact(k);
                rejects([&] { engine.apply(plan, batch); });
                require(std::filesystem::file_size(name + "/fresh.gplot") == 0,
                        "changed packed planned input mutated the destination");
                CompactMemberBuilder restore(k, members[2].size());
                restore.append(members[2]);
                retained[2] = restore.finish();
                engine.apply(plan, batch);
            }
            offset += count;
        }
    }
    std::ifstream fresh(name + "/fresh.gplot", std::ios::binary);
    std::ifstream staged(name + "/staged.gplot", std::ios::binary);
    require(std::filesystem::file_size(name + "/fresh.gplot") ==
            std::filesystem::file_size(name + "/staged.gplot") &&
            std::equal(std::istreambuf_iterator<char>(fresh), {}, std::istreambuf_iterator<char>(staged)),
            "indexed-member batch changes altered canonical bytes");
    MemberBatchHash hash(members.size());
    std::vector<Member> immutable;
    immutable.reserve(members.size());
    for (auto const& raw : members)
    {
        CompactMemberBuilder builder(k, raw.size(), &hash);
        builder.append(raw);
        immutable.push_back(builder.finish());
    }
    auto const expected_digest = framed_digest(members);
    require(hash.digest_for(immutable) == std::optional(expected_digest),
            "fused complete batch hash differs from logical raw member framing");
    uint16_t const count = uint16_t(members.size());
    uint64_t const compact_memory = Engine::memory_bytes(k, config, count, 0, count);
    uint64_t const raw_memory = Engine::memory_bytes(k, config, count);
    require(raw_memory > compact_memory && raw_memory - compact_memory ==
                uint64_t(count) * (FragmentBoundaries::allocation_bytes(size_t(chunks), members[0].size()) - sizeof(uint64_t)),
            "compact memory admission still reserves unused raw boundary arrays");
    require(Engine::memory_bytes(k, config, count, 1) - raw_memory ==
                (chunks + 1) * (sizeof(size_t) - sizeof(uint32_t)),
            "generic wide raw member admission omitted its larger boundary table");
    require(raw_memory - Engine::memory_bytes(k, config, count, 0, 1) ==
                FragmentBoundaries::allocation_bytes(size_t(chunks), members[0].size()),
            "mixed member admission retained a compact member's raw boundary table");
    rejects([&] { Engine::memory_bytes(k, config, count, 1, count); });
    rejects([&] { Engine::memory_bytes(k, config, 1, 0, 2); });
    {
        auto tight = config;
        tight.memory_limit_bytes = compact_memory;
        Engine engine(name + "/compact-budget.gplot", tight, false, identity);
        engine.confirm_identity();
        require(engine.memory_bytes(immutable) == compact_memory && engine.memory_bytes(members) == raw_memory,
                "exact member storage admission disagrees with its planned shape");
        rejects([&] { engine.plan_fresh(members); });
        auto plan = engine.plan_fresh(immutable, &hash);
        require(plan && plan->accepted && plan->members_digest == expected_digest &&
                !std::filesystem::file_size(name + "/compact-budget.gplot"),
                "compact-only budget failed before payload-dependent admission");
    }
    std::vector<Member> moved;
    {
        Engine engine(name + "/cached.gplot", config, false, identity);
        engine.confirm_identity();
        auto plan = engine.plan_fresh(immutable, &hash);
        require(plan && plan->accepted, "prehashed immutable batch was not admitted");
        moved.reserve(immutable.size());
        for (auto& member : immutable) moved.push_back(std::move(member));
        require(hash.digest_for(moved).has_value(), "moving members changed a prehashed batch identity");
        ++members[0][0];
        rejects([&] { engine.apply(*plan, members); });
        --members[0][0];
        require(!std::filesystem::file_size(name + "/cached.gplot"),
                "raw input reused an immutable digest and changed the destination");
        plan = engine.plan_fresh(moved, &hash);
        auto changed = members[2];
        ++changed[0];
        moved[2] = Member(std::move(changed));
        moved[2].compact(k);
        require(!hash.digest_for(moved), "replacement data retained a stale construction hash");
        rejects([&] { engine.apply(*plan, moved); });
        require(!std::filesystem::file_size(name + "/cached.gplot"),
                "replaced compact input bypassed public plan/apply validation");
        moved[2] = Member(std::vector<uint64_t>(members[2]));
        moved[2].compact(k);
        engine.apply(*plan, moved);
    }
    auto contents = [](std::string const& path)
    {
        std::ifstream input(path, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
    };
    auto const canonical = contents(name + "/fresh.gplot");
    require(contents(name + "/cached.gplot") == canonical, "cached digest changed canonical output bytes");
    hash.reset();
    for (size_t i = 0; i < moved.size(); ++i)
    {
        CompactMemberBuilder builder(k, members[i].size(), &hash);
        builder.append(members[i]);
        moved[i] = builder.finish();
    }
    pid_t const child = ::fork();
    require(child >= 0, "cannot fork immutable digest recovery check");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "direct-committed", 1);
        try
        {
            Engine engine(name + "/cached-recovery.gplot", config, false, identity);
            engine.confirm_identity();
            engine.append(moved, &hash);
        }
        catch (...) { std::_Exit(1); }
        std::_Exit(0);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "immutable digest recovery did not reach a committed checkpoint");
    {
        Engine engine(name + "/cached-recovery.gplot", config, true);
        engine.confirm_identity();
        auto const before = contents(name + "/cached-recovery.gplot");
        auto changed = members[0];
        ++changed[0];
        moved[0] = Member(std::move(changed));
        moved[0].compact(k);
        rejects([&] { engine.resume(moved, &hash); });
        require(contents(name + "/cached-recovery.gplot") == before,
                "stale regenerated digest changed recovery bytes");
        moved[0] = Member(std::vector<uint64_t>(members[0]));
        moved[0].compact(k);
        engine.resume(moved, &hash);
    }
    require(contents(name + "/cached-recovery.gplot") == canonical,
            "immutable member digest recovery changed canonical bytes");
    auto group = PlotGroupFile::open(name + "/staged.gplot");
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        std::vector<std::vector<ProofFragment>> decoded(members.size());
        group.readChunk(chunk, decoded);
        for (size_t member = 0; member < members.size(); ++member)
        {
            auto const& values = members[member];
            auto first = std::lower_bound(values.begin(), values.end(), chunk * range);
            auto last = std::lower_bound(first, values.end(), (chunk + 1) * range);
            std::vector<uint64_t> expected(first, last);
            expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
            require(decoded[member] == expected, "indexed member decoded into the wrong chunk");
        }
    }
}

// Optional saved fixtures exercise the actual pinned writer/reader boundary.
// Reconstruct sampled chunks from upstream-decoded members, then compare every
// encoded byte both for a complete batch and for appending the final member.
void fixture(std::string const& path)
{
    auto in = group_tools::input(path);
    auto const header = group_tools::read<PlotGroupFile::Header>(in);
    require(header.magic == PlotGroupFile::MAGIC && header.version == PlotGroupFile::FORMAT_VERSION,
            "fixture is not an ordinary upstream group");
    auto const file_size = std::filesystem::file_size(path);
    require(header.chunk_index_offset < file_size, "invalid fixture index");
    auto const index_size = file_size - header.chunk_index_offset;
    std::vector<uint64_t> packed(index_size / 8 + (index_size % 8 != 0));
    in.seekg(header.chunk_index_offset);
    in.read(reinterpret_cast<char*>(packed.data()), index_size);
    BitReader index(packed, index_size * 8);
    size_t const count = PlotGroupFile::getChunkCountForK(header.k);
    std::vector<uint64_t> offsets(count + 1);
    gsz_decode(std::span(offsets).first(count), header.group_size, index);
    uint64_t position = sizeof(header) + header.memo_length;
    for (size_t i = 0; i < count; ++i)
    {
        auto const size = offsets[i];
        offsets[i] = position;
        position += size;
    }
    offsets.back() = position;
    require(position == header.chunk_index_offset, "invalid fixture chunk coverage");
    auto upstream = PlotGroupFile::open(path);
    ChunkCodec codec(header.k, 1ull << 28);
    std::vector<std::vector<ProofFragment>> members(header.group_size);
    std::vector<std::span<uint64_t const>> spans(header.group_size);
    std::array<size_t, 5> const samples{0, count / 4, count / 2, count * 3 / 4, count - 1};
    for (size_t chunk : samples)
    {
        for (auto& member : members) member.clear();
        upstream.readChunk(chunk, members);
        for (size_t i = 0; i < members.size(); ++i) spans[i] = members[i];
        std::vector<uint8_t> canonical(offsets[chunk + 1] - offsets[chunk]);
        in.seekg(offsets[chunk]);
        in.read(reinterpret_cast<char*>(canonical.data()), canonical.size());
        require(copy(codec.encode(chunk, {}, 0, spans)) == canonical,
                "codec differs from pinned upstream fixture");
        require(codec.encoded_size(chunk, {}, 0, spans) == canonical.size(),
                "sizing differs from pinned upstream fixture");
        require(codec.scan(chunk, canonical, header.group_size) == canonical.size(),
                "fixture scan length differs");
        require(copy(codec.encode(chunk, canonical, header.group_size, {})) == canonical,
                "canonical delta-prefix re-encode differs");
        if (header.group_size > 1)
        {
            auto const old = copy(codec.encode(chunk, {}, 0, std::span(spans).first(spans.size() - 1)));
            require(copy(codec.encode(chunk, old, header.group_size - 1,
                    std::span(spans).last(1))) == canonical, "fixture append differs");
        }
    }
    std::cout << "canonical fixture parity passed: " << path << '\n';
}

// Queue pressure, failed jobs and direct serial work all use the same finite
// compute grant. Barriers make the resource assertions independent of timing.
void shared_scheduler()
{
    rejects([] { CodecScheduler invalid(0); });
    {
#ifdef __linux__
        auto thread_count = []
        { return std::distance(std::filesystem::directory_iterator("/proc/self/task"), std::filesystem::directory_iterator{}); };
        auto const before = thread_count();
#endif
        auto preview = std::make_shared<CodecScheduler>(2, CodecScheduler::Execution::preview);
#ifdef __linux__
        require(thread_count() == before, "read-only scheduler started worker threads");
#endif
        require(preview->workers() == 2, "read-only scheduler lost its modeled worker count");
        bool ran = false;
        rejects([&] { (void)preview->submit([&] { ran = true; }); });
        rejects([&] { CodecScheduler::Lease invalid(preview.get()); ran = true; });
        require(!ran, "read-only scheduler dispatched codec work");
        Config modeled; modeled.workers = 2; modeled.cpu_scheduler = preview;
        auto executing = modeled; executing.cpu_scheduler = std::make_shared<CodecScheduler>(2);
        require(Engine::memory_bytes(18, modeled, 1) == Engine::memory_bytes(18, executing, 1),
            "read-only scheduler changed the runtime memory admission model");
    }
    for (unsigned workers : {1u, 2u})
    {
        auto scheduler = std::make_shared<CodecScheduler>(workers);
        rejects([&] { CodecScheduler::Lease invalid(scheduler.get(), 0); });
        rejects([&] { CodecScheduler::Lease invalid(scheduler.get(), workers + 1); });
        {
            CodecScheduler::Lease serial(scheduler.get());
            rejects([&] { CodecScheduler::Lease invalid(scheduler.get(), 0); });
            if (workers > 1) rejects([&] { CodecScheduler::Lease upgrade(scheduler.get(), workers); });
        }
        std::mutex mutex; std::condition_variable ready;
        unsigned active = 0, maximum = 0, started = 0, completed = 0;
        std::vector<unsigned> order;
        bool released = false;
        auto producer = std::async(std::launch::async, [&]
        {
            std::vector<std::future<void>> pending;
            for (unsigned destination = 0; destination < 60; ++destination)
                pending.push_back(scheduler->submit([&, destination]
                {
                    std::unique_lock lock(mutex);
                    ++active; ++started; order.push_back(destination);
                    maximum = std::max(maximum, active); ready.notify_all();
                    require(ready.wait_for(lock, std::chrono::seconds(10), [&] { return released; }),
                        "shared CPU barrier timed out");
                    --active; ++completed;
                    if (destination == 7) throw std::runtime_error("scheduled codec failure");
                }));
            unsigned failures = 0;
            for (auto& future : pending)
            {
                try { future.get(); }
                catch (std::runtime_error const&) { ++failures; }
            }
            require(failures == 1, "shared scheduler lost a task failure");
        });
        bool full;
        {
            std::unique_lock lock(mutex);
            full = ready.wait_for(lock, std::chrono::seconds(5), [&] { return started == workers; });
            released = true;
        }
        ready.notify_all(); producer.get();
        require(full && maximum == workers && !active && completed == 60,
            "shared scheduler did not bound or drain independent outputs");
        if (workers == 1) require(std::is_sorted(order.begin(), order.end()), "shared queue lost FIFO progress");

        // Serial fallback computation consumes the same grant as queued tasks.
        std::vector<std::future<void>> pending;
        std::promise<void> queued; auto enqueued = queued.get_future();
        // One caller can borrow a permit recursively without reserving another.
        {
            CodecScheduler::Lease lease(scheduler.get(), workers);
            CodecScheduler::Lease nested(scheduler.get());
            {
                pending.push_back(std::async(std::launch::async, [&]
                {
                    auto job = scheduler->submit([] {}); queued.set_value(); job.get();
                }));
                require(enqueued.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                    "serial codec probe did not enter the shared queue");
                require(pending.back().wait_for(std::chrono::seconds(0)) != std::future_status::ready,
                    "serial codec work bypassed shared CPU admission");
            }
        }
        for (auto& future : pending) future.get();
        auto nested = scheduler->submit([&] { (void)scheduler->submit([] {}); });
        rejects([&] { nested.get(); });
        scheduler->submit([] {}).get();
    }

    // Cancelling the head of the CPU queue must expose a fitting follower
    // while another caller still holds its permit, without leaking a ticket.
    {
        CodecScheduler scheduler(2);
        std::atomic<bool> cancel{false};
        std::promise<void> queued;
        auto ready = queued.get_future();
        std::future<void> head, follower;
        {
            CodecScheduler::Lease held(&scheduler);
            head = std::async(std::launch::async, [&]
            {
                unsigned checks = 0;
                CodecScheduler::Lease waiting(&scheduler, 2, [&]
                {
                    if (++checks == 2) queued.set_value();
                    require(!cancel.load(), "cancelled CPU admission");
                });
            });
            require(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "cancellable CPU admission did not queue");
            follower = std::async(std::launch::async, [&] { CodecScheduler::Lease fitting(&scheduler); });
            cancel.store(true);
            require(head.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "CPU admission ignored cancellation");
            rejects([&] { head.get(); });
            require(follower.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "cancelled CPU waiter blocked a fitting follower");
            follower.get();
        }
        scheduler.submit([] {}).get();
    }

    // Destruction drains queued operations even when their caller keeps only
    // the futures. A failed operation cannot abandon its following sibling.
    std::vector<std::future<void>> pending;
    unsigned completed = 0;
    {
        CodecScheduler scheduler(1);
        pending.push_back(scheduler.submit([] { throw std::runtime_error("failed drain"); }));
        pending.push_back(scheduler.submit([&] { ++completed; }));
    }
    rejects([&] { pending.front().get(); }); pending.back().get();
    require(completed == 1, "scheduler shutdown abandoned queued work");
}

// An ordered consumer may retain encoded bytes through a storage wait. Other
// drives must continue encoding without borrowing or reclaiming that storage.
void shared_consumers()
{
    constexpr uint8_t k = 18;
    constexpr uint64_t budget = 1ull << 24;
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<uint64_t> values;
    for (uint64_t i = 0; i < 64; ++i) values.push_back(i * (range / 64) + 7);
    FragmentRegion region{std::span<uint64_t const>(values)};
    std::array<ChunkInput, 4> chunks;
    for (auto& chunk : chunks) chunk = {0, {}, 0, {&region, 1}};
    ChunkCodec reference(k, budget);
    auto const expected = copy(reference.encode(0, {}, 0, {&region, 1}));
    for (unsigned workers : {1u, 2u})
    {
        auto scheduler = std::make_shared<CodecScheduler>(workers);
        ParallelEncoder first(k, workers, budget, scheduler), second(k, workers, budget, scheduler);
        std::array<uint64_t, 4> first_sizes{}, second_sizes{};
        std::promise<void> blocked, release;
        auto released = release.get_future().share();
        OrderedWriter drive([] {}), independent([] {});
        auto held = drive.submit([&]
        {
            auto const blocks = first.encode(chunks, first_sizes);
            blocked.set_value();
            require(released.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
                "ordered consumer barrier timed out");
            for (auto const& block : blocks)
                for (size_t task = block.begin, offset = 0; task < block.end; ++task)
                {
                    require(std::equal(expected.begin(), expected.end(), block.bytes.begin() + offset),
                        "independent output reused a blocked consumer's codec bytes");
                    offset += first_sizes[task];
                }
        });
        require(blocked.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "ordered consumer did not reach its storage wait");
        auto other = independent.submit([&]
        {
            for (unsigned destination = 0; destination < 60; ++destination)
            {
                second.measure(chunks, second_sizes);
                auto const blocks = second.encode(chunks, second_sizes, true);
                for (auto const size : second_sizes) require(size == expected.size(), "shared output size differs");
                require(!blocks.empty() && second.scratch_bytes() <= budget,
                    "shared encoder exceeded its retained workspace grant");
            }
            auto invalid = chunks; invalid.front().new_members = {};
            rejects([&] { second.encode(invalid, second_sizes); });
            second.encode(chunks, second_sizes);
        });
        auto const progressed = other.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        bool const still_held = held.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
        std::atomic<bool> same_drive_ran{false};
        auto queued = drive.submit([&] { same_drive_ran = true; });
        bool const serialized = !same_drive_ran.load();
        release.set_value(); other.get(); held.get(); queued.get();
        require(progressed && still_held && serialized && same_drive_ran,
            "blocked storage retained shared CPU admission or lost drive ordering");
    }
}

void remote_codec_handoff()
{
    constexpr uint8_t k = 18;
    constexpr uint64_t budget = 1ull << 24;
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<uint64_t> fragments;
    for (uint64_t i = 0; i < 64; ++i) fragments.push_back(i * (range / 64) + 7);
    FragmentRegion const region{std::span<uint64_t const>(fragments)};
    std::array<ChunkInput, 4> chunks;
    for (auto& chunk : chunks) chunk = {0, {}, 0, {&region, 1}};
    ChunkCodec reference(k, budget);
    auto const expected = copy(reference.encode(0, {}, 0, {&region, 1}));
    struct Provider final : RemoteEncoder
    {
        std::vector<uint8_t> expected;
        bool ready = false, decline = false, missing = false, trailing = false, zero = false;
        unsigned observations = 0;
        std::optional<RemoteEncodedWindow> encode(uint8_t, std::span<ChunkInput const> chunks, bool) override
        {
            if (!ready || decline) return {};
            RemoteEncodedWindow result;
            for (auto const& chunk : chunks)
            { result.bytes.insert(result.bytes.end(), expected.begin(), expected.end()); result.sizes.push_back(expected.size()); }
            if (missing) result.sizes.pop_back();
            if (trailing) result.bytes.push_back(0);
            if (zero) result.sizes.front() = 0;
            return result;
        }
        void observe_local(uint8_t, std::span<ChunkInput const> chunks, std::span<uint64_t const> sizes,
            std::span<EncodedBlock const> blocks, uint64_t elapsed) override
        {
            require(chunks.size() == 4 && sizes.size() == chunks.size() && !blocks.empty() && elapsed,
                "remote calibration did not receive the complete local window");
            size_t next = 0;
            for (auto const& block : blocks)
            {
                require(block.begin == next && block.end > block.begin && block.end <= chunks.size() &&
                    block.bytes.size() == (block.end - block.begin) * expected.size(),
                    "remote calibration changed the completed block boundaries");
                for (size_t chunk = block.begin; chunk < block.end; ++chunk)
                    require(sizes[chunk] == expected.size() && std::equal(expected.begin(), expected.end(),
                        block.bytes.begin() + (chunk - block.begin) * expected.size()),
                        "remote calibration changed completed local bytes");
                next = block.end;
            }
            require(next == chunks.size(), "remote calibration omitted completed chunks");
            ready = true; ++observations;
        }
    };
    auto provider = std::make_shared<Provider>(); provider->expected = expected;
    ParallelEncoder encoder(k, 2, budget, {}, provider);
    std::array<uint64_t, 4> sizes{};
    encoder.encode(chunks, sizes);
    require(provider->observations == 1 && encoder.cpu_time().measured, "local calibration contaminated CPU history");
    auto const remote = encoder.encode(chunks, sizes);
    require(remote.size() == 1 && remote.front().begin == 0 && remote.front().end == chunks.size() &&
        !encoder.cpu_time().measured, "remote output was not distinguished from local CPU work");
    auto const held = remote.front().bytes;
    encoder.encode(chunks, sizes, true);
    for (size_t offset = 0; offset < held.size(); offset += expected.size())
        require(std::equal(expected.begin(), expected.end(), held.begin() + offset), "remote retained output was reused early");
    provider->decline = true;
    auto const fallback = encoder.encode(chunks, sizes, true);
    require(!fallback.empty(), "remote admission decline lost local fallback");
    for (auto size : sizes) require(size == expected.size(), "remote fallback changed canonical lengths");
    auto const local_held = fallback.front().bytes;
    provider->decline = false;
    auto const resumed = encoder.encode(chunks, sizes, true);
    require(std::equal(expected.begin(), expected.end(), local_held.begin()),
        "remote output invalidated a retained local codec block");
    auto const remote_held = resumed.front().bytes;
    provider->decline = true; encoder.encode(chunks, sizes, true);
    require(std::equal(expected.begin(), expected.end(), remote_held.begin()),
        "local fallback invalidated a retained remote block");
    provider->decline = false; provider->missing = true;
    rejects([&] { encoder.encode(chunks, sizes); });
    provider->missing = false; provider->trailing = true;
    rejects([&] { encoder.encode(chunks, sizes); });
    provider->trailing = false; provider->zero = true;
    rejects([&] { encoder.encode(chunks, sizes); });
}

int main(int argc, char** argv) try
{
    shared_scheduler();
    shared_consumers();
    remote_codec_handoff();
    synthetic();
    summaries();
    joined_chunks();
    segmented_sha();
    compact_regions();
    compact_word_tails();
    compact_received_tiles();
    compact_bound_fallback();
    batch_hashing();
    adaptive_overlap();
    indexed_members();
    for (int i = 1; i < argc; ++i) fixture(argv[i]);
    std::cout << "group growth codec checks passed\n";
}
catch (std::exception const& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
