// Native C++20 CPU producer using pos2-chip's AES rounds, parameters and fragment codec.
// Keep the implementation with the callers that already include those headers.
#pragma once

#include "host/CpuPlotMemory.hpp"
#include "pos/ProofCore.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pos2gpu::cpu {
namespace detail {

// Every active lane appends to its own buckets. Full blocks are linked inside
// one allocation so table destruction releases the whole stage at once.
template<class T>
class Table {
    struct Bin { uint32_t first, last, count; };
    static_assert(sizeof(Bin) == 12);
public:
    Table(Layout const& layout, uint32_t rows, uint32_t bins)
        : block_(layout.block), limit_(rows), bins_(bins), lanes_(layout.lanes),
          capacity_(layout.blocks(rows, bins)), data_(std::make_unique_for_overwrite<T[]>(size_t(capacity_) * block_)),
          next_(std::make_unique_for_overwrite<uint32_t[]>(capacity_)), directory_(size_t(lanes_) * bins)
    {}

    void append(unsigned lane, unsigned bin, T value)
    {
        auto& dest = directory_[size_t(lane) * bins_ + bin];
        const uint32_t offset = dest.count & (block_ - 1);
        if (offset == 0) {
            const auto id = allocated_.fetch_add(1, std::memory_order_relaxed);
            if (id >= capacity_) throw std::length_error("CPU table exceeds admitted capacity");
            if (dest.count) next_[dest.last] = id;
            else dest.first = id;
            dest.last = id;
        }
        data_[size_t(dest.last) * block_ + offset] = value;
        ++dest.count;
    }

    void reset()
    {
        std::fill(directory_.begin(), directory_.end(), Bin{});
        allocated_.store(0, std::memory_order_relaxed);
    }

    uint32_t size() const
    {
        uint64_t total = 0;
        for (auto const& b : directory_) total += b.count;
        if (total > limit_) throw std::length_error("CPU table exceeds admitted row count");
        return uint32_t(total);
    }

    template<class F>
    void visit(unsigned bin, F&& f) const
    {
        for (unsigned lane = 0; lane < lanes_; ++lane) {
            const auto& b = directory_[size_t(lane) * bins_ + bin];
            uint32_t id = b.first, remaining = b.count;
            while (remaining) {
                const uint32_t count = std::min(remaining, block_);
                f(std::span<const T>(data_.get() + size_t(id) * block_, count));
                remaining -= count;
                if (remaining) id = next_[id];
            }
        }
    }
private:
    uint32_t block_, limit_, bins_, lanes_, capacity_;
    std::unique_ptr<T[]> data_;
    std::unique_ptr<uint32_t[]> next_;
    std::vector<Bin> directory_;
    std::atomic<uint32_t> allocated_{0};
};

class Work {
public:
    Work(unsigned lanes, std::function<void()> const& check) : lanes_(lanes), check_(check) {}

    void poll()
    {
        if (!check_) return;
        std::lock_guard lock(mutex_);
        check_();
    }

    template<class F>
    void parallel(size_t count, F&& task)
    {
        poll();
        if (!count) return;
        const auto active = unsigned(std::min<size_t>(lanes_, count));
        if (active == 1) {
            for (size_t i = 0; i < count; ++i) task(0, i);
            return;
        }
        std::atomic<size_t> next{0};
        std::atomic<bool> stopped{false};
        std::exception_ptr failure;
        auto run = [&](unsigned lane) {
            try {
                while (!stopped.load(std::memory_order_relaxed)) {
                    const auto i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) break;
                    task(lane, i);
                }
            } catch (...) {
                std::lock_guard lock(mutex_);
                if (!failure) failure = std::current_exception();
                stopped.store(true, std::memory_order_relaxed);
            }
        };
        {
            std::vector<std::jthread> threads;
            threads.reserve(active - 1);
            for (unsigned lane = 1; lane < active; ++lane) threads.emplace_back(run, lane);
            run(0);
        }
        if (failure) std::rethrow_exception(failure);
    }
private:
    unsigned lanes_;
    std::function<void()> const& check_;
    std::mutex mutex_;
};

struct Xs { uint32_t info, x; };
struct First { uint32_t info, lo, hi; };
struct Second { uint32_t info, lo, hi, xbits; };
static_assert(sizeof(Xs) == 8 && sizeof(First) == 12 && sizeof(Second) == 16);

inline uint64_t meta(Xs row) { return row.x; }
inline uint64_t meta(First row) { return row.lo | (uint64_t(row.hi) << 32); }
inline uint64_t meta(Second row) { return row.lo | (uint64_t(row.hi) << 32); }

// Independent AES states can occupy the CPU's round pipelines together. Use
// the proof implementation's round primitive on every supported architecture.
class HashBatch {
public:
    static constexpr unsigned width = 8;
    using States = rx_vec_i128[width];

    explicit HashBatch(PlotId const& id) : first_(key(id.bytes())), second_(key(id.bytes() + 16)) {}

    void hash(States& states, unsigned rounds) const
    {
        hash(states, rounds, std::make_index_sequence<width>{});
    }

    template<unsigned Stage, class Row, class F>
    void targets(std::span<const Row> rows, ProofCore& proof, unsigned match_key,
                 unsigned rounds, uint32_t mask, F&& emit) const
    {
        while (rows.size() >= width) {
            States states;
            for (unsigned i = 0; i < width; ++i) {
                const uint64_t value = meta(rows[i]);
                states[i] = rx_set_int_vec_i128(int32_t(value >> 32), int32_t(value), int32_t(match_key), Stage);
            }
            hash(states, rounds);
            for (unsigned i = 0; i < width; ++i)
                emit(rows[i], uint32_t(rx_vec_i128_x(states[i])) & mask);
            rows = rows.subspan(width);
        }
        for (auto row : rows) emit(row, proof.matching_target(Stage, meta(row), match_key));
    }

private:
    // Expand the eight independent states at compile time. Release builds
    // using -O2 otherwise reload the states from memory between AES rounds.
    template<size_t... I>
    void hash(States& states, unsigned rounds, std::index_sequence<I...>) const
    {
        const auto first = first_, second = second_;
        for (unsigned round = 0; round < rounds; ++round) {
            ((states[I] = aesenc<!HAVE_AES>(states[I], first)), ...);
            ((states[I] = aesenc<!HAVE_AES>(states[I], second)), ...);
        }
    }

    static rx_vec_i128 key(const uint8_t* bytes)
    {
        return rx_set_int_vec_i128(int32_t(load32(bytes + 12)), int32_t(load32(bytes + 8)),
                                   int32_t(load32(bytes + 4)), int32_t(load32(bytes)));
    }
    rx_vec_i128 first_, second_;
};

// Each lane owns its pending pairs. Only a full batch or the final partial
// batch performs the rounds, and only passing pairs create an output row.
template<unsigned Stage, class Row>
struct alignas(64) PairBatch {
    Row left[HashBatch::width], right[HashBatch::width];
    unsigned count = 0;

    template<class F>
    void flush(HashBatch const& hashes, unsigned k, uint32_t filter,
               unsigned rounds, F&& emit)
    {
        HashBatch::States states{};
        for (unsigned i = 0; i < count; ++i) {
            const uint64_t l = meta(left[i]), r = meta(right[i]);
            states[i] = rx_set_int_vec_i128(int32_t(r >> 32), int32_t(r), int32_t(l >> 32), int32_t(l));
        }
        hashes.hash(states, rounds);
        for (unsigned i = 0; i < count; ++i) {
            const auto state = states[i];
            if (uint32_t(rx_vec_i128_w(state)) & filter) continue;
            if constexpr (Stage == 3) {
                emit((uint64_t(left[i].xbits) << k) | right[i].xbits);
            } else {
                const uint32_t info = uint32_t(rx_vec_i128_x(state)) & ((uint32_t{1} << k) - 1);
                const uint64_t l = meta(left[i]), r = meta(right[i]);
                if constexpr (Stage == 1) {
                    const uint64_t xs = (l << k) | r;
                    emit(First{info, uint32_t(xs), uint32_t(xs >> 32)});
                } else {
                    const unsigned half = k / 2;
                    const uint32_t xs = (uint32_t(l >> (k + half)) << half) | uint32_t(r >> (k + half));
                    const uint32_t hi_mask = (uint32_t{1} << (2 * k - 32)) - 1;
                    emit(Second{info, uint32_t(rx_vec_i128_y(state)), uint32_t(rx_vec_i128_z(state)) & hi_mask, xs});
                }
            }
        }
        count = 0;
    }
};

// Encode consecutive successful T3 pairs together before scattering the
// resulting fragments to their sort buckets.
struct alignas(64) FragmentBatch {
    static constexpr unsigned width = 16;
    uint64_t values[width];
    unsigned count = 0;

    void flush(ProofFragmentCodec const& codec, Table<uint64_t>& output, unsigned lane, unsigned shift)
    {
        for (unsigned i = 0; i < count; ++i) values[i] = codec.encode(values[i]);
        for (unsigned i = 0; i < count; ++i) output.append(lane, unsigned(values[i] >> shift), values[i]);
        count = 0;
    }
};

template<class Row>
class RightIndex {
public:
    // Each tile retains all rows with the same target, including duplicates.
    template<class F>
    void tiles(Table<Row> const& input, unsigned bin, uint32_t base, uint32_t target_mask,
               uint32_t heads, uint32_t capacity, Work& work, F&& consume)
    {
        heads_.assign(heads, 0);
        if (capacity > capacity_) {
            nodes_ = std::make_unique_for_overwrite<Row[]>(capacity);
            capacity_ = capacity;
        }
        uint32_t filled = 0;
        unsigned blocks_seen = 0;
        input.visit(bin, [&](auto block) {
            if (!(blocks_seen++ % 16)) work.poll();
            for (auto row : block) {
                if ((row.info & ~target_mask) != base) continue;
                if (filled == capacity) {
                    consume();
                    std::fill(heads_.begin(), heads_.end(), 0);
                    filled = 0;
                }
                auto& head = heads_[row.info & (heads - 1)];
                row.info = head;
                nodes_[filled++] = row;
                head = filled;
            }
        });
        if (filled) consume();
    }

    template<class F>
    void visit(uint32_t target, F&& emit) const
    {
        for (auto id = heads_[target & (heads_.size() - 1)]; id; ) {
            const auto& row = nodes_[id - 1];
            emit(row);
            id = row.info;
        }
    }
private:
    uint32_t capacity_ = 0;
    std::unique_ptr<Row[]> nodes_;
    std::vector<uint32_t> heads_;
};

template<unsigned Stage, class Row, class Output>
void match(Layout const& layout, PlotProofParams const& params, HashBatch const& hashes,
           ProofCore& proof, Work& work, Table<Row> const& input, Table<Output>& output)
{
    const unsigned shift = layout.k - layout.bucket_bits;
    const unsigned output_shift = (Stage == 3 ? 2 * layout.k : layout.k) - layout.bucket_bits;
    const unsigned section_bins = layout.buckets / params.get_num_sections();
    const unsigned keys = unsigned(params.get_num_match_keys(Stage));
    const unsigned target_bits = unsigned(params.get_num_match_target_bits(Stage));
    const uint32_t mask = (uint32_t{1} << target_bits) - 1;
    const unsigned query_bins = std::max(1u, section_bins / keys);
    const unsigned head_count = std::min(layout.heads, mask + 1);
    const unsigned extra = Stage == 1 ? params.get_strength() - 2 : 0;
    const unsigned target_rounds = AES_MATCHING_TARGET_ROUNDS << extra;
    const unsigned pair_rounds = AES_PAIRING_ROUNDS << extra;
    std::vector<RightIndex<Row>> scratch(layout.lanes);
    std::vector<PairBatch<Stage, Row>> pairs(layout.lanes);
    std::vector<FragmentBatch> fragments;
    if constexpr (Stage == 3) fragments.resize(layout.lanes);
    auto flush = [&](unsigned lane) {
        pairs[lane].flush(hashes, layout.k, keys - 1, pair_rounds, [&](Output result) {
            if constexpr (Stage == 3) {
                auto& packed = fragments[lane];
                packed.values[packed.count++] = result;
                if (packed.count == FragmentBatch::width)
                    packed.flush(proof.fragment_codec, output, lane, output_shift);
            } else {
                output.append(lane, result.info >> output_shift, result);
            }
        });
    };
    auto add_pair = [&](unsigned lane, Row left, Row right) {
        auto& batch = pairs[lane];
        batch.left[batch.count] = left;
        batch.right[batch.count++] = right;
        if (batch.count == HashBatch::width) flush(lane);
    };

    std::unique_ptr<Table<Row>> queries;
    if (query_bins > 1) queries = std::make_unique<Table<Row>>(layout, layout.query_rows, query_bins);
    for (unsigned section = 0; section < params.get_num_sections(); ++section) {
        std::vector<std::span<const Row>> left;
        for (unsigned i = 0; i < section_bins; ++i)
            input.visit(section * section_bins + i, [&](auto block) { left.push_back(block); });
        auto targets = [&](size_t first, size_t count, unsigned key, auto&& emit) {
            constexpr unsigned blocks_per_task = 8;
            work.parallel((count + blocks_per_task - 1) / blocks_per_task, [&](unsigned lane, size_t task) {
                const auto start = task * blocks_per_task;
                for (size_t i = start; i < std::min(count, start + blocks_per_task); ++i) {
                    if (i == start || (Stage == 1 && params.get_strength() > 8)) work.poll();
                    hashes.targets<Stage>(left[first + i], proof, key, target_rounds, mask,
                                          [&](Row row, uint32_t target) { emit(lane, row, target); });
                }
            });
        };
        const unsigned right_section = proof.matching_section(section);
        for (unsigned key = 0; key < keys; ++key) {
            const uint32_t base = (right_section * keys + key) << target_bits;
            const unsigned first_bin = base >> shift;
            if (!queries) {
                // All targets fit one index. Share it read-only while lanes probe
                // their own left blocks. The grant holds two rows per target.
                // A larger bucket is processed in successive tiles.
                auto& index = scratch.front();
                index.tiles(input, first_bin, base, mask, head_count, 2 * head_count, work, [&] {
                    targets(0, left.size(), key, [&](unsigned lane, Row row, uint32_t target) {
                        index.visit(target, [&](Row right) { add_pair(lane, row, right); });
                    });
                });
                continue;
            }
            // Bound the left batch even when a section exceeds its expected size.
            const size_t batch = layout.query_rows / layout.block;
            for (size_t first = 0; first < left.size(); first += batch) {
                queries->reset();
                targets(first, std::min(batch, left.size() - first), key, [&](unsigned lane, Row row, uint32_t target) {
                    row.info = target;
                    queries->append(lane, target >> shift, row);
                });
                work.parallel(query_bins, [&](unsigned lane, size_t bin) {
                    auto& index = scratch[lane];
                    index.tiles(input, first_bin + unsigned(bin), base, mask, head_count, layout.right_tile, work, [&] {
                        unsigned blocks_seen = 0;
                        queries->visit(unsigned(bin), [&](auto block) {
                            if (!(blocks_seen++ % 16)) work.poll();
                            for (auto row : block)
                                index.visit(row.info, [&](Row right) { add_pair(lane, row, right); });
                        });
                    });
                });
            }
        }
    }
    for (unsigned lane = 0; lane < layout.lanes; ++lane) {
        if (pairs[lane].count) flush(lane);
        if constexpr (Stage == 3) fragments[lane].flush(proof.fragment_codec, output, lane, output_shift);
    }
    output.size();
}

inline void sort_fragments(std::span<uint64_t> values, unsigned bits, Work& work)
{
    if (values.size() < 2) return;
    auto temp = std::make_unique_for_overwrite<uint64_t[]>(values.size());
    uint64_t* from = values.data();
    uint64_t* to = temp.get();
    for (unsigned shift = 0; shift < bits; shift += 8) {
        work.poll();
        std::array<uint32_t, 256> offsets{};
        for (size_t i = 0; i < values.size(); ++i) ++offsets[(from[i] >> shift) & 255];
        uint32_t sum = 0;
        for (auto& n : offsets) { const auto count = n; n = sum; sum += count; }
        for (size_t i = 0; i < values.size(); ++i) to[offsets[(from[i] >> shift) & 255]++] = from[i];
        std::swap(from, to);
    }
    if (from != values.data()) std::copy_n(from, values.size(), values.data());
}
} // namespace detail

inline std::vector<uint64_t> generate(PlotProofParams const& params, unsigned threads, uint64_t ram_bytes,
                                      std::function<void()> const& check = {})
{
    const Layout layout(unsigned(params.get_k()), threads);
    if (layout.peak_bytes(params.get_strength()) > ram_bytes)
        throw std::length_error("CPU plotting exceeds the available RAM grant");
    detail::Work work(layout.lanes, check);
    work.poll();
    ProofCore proof(params);
    using namespace detail;
    const HashBatch hashes(params.get_plot_id());
    auto xs = std::make_unique<Table<Xs>>(layout, layout.n, layout.buckets);
    constexpr unsigned tile = 8192;
    work.parallel(layout.n / tile, [&](unsigned lane, size_t part) {
        work.poll();
        for (uint32_t x = uint32_t(part) * tile; x < (part + 1) * tile; ++x) {
            const uint32_t info = proof.hashing.g(x);
            xs->append(lane, info >> (layout.k - layout.bucket_bits), Xs{info, x});
        }
    });
    auto t1 = std::make_unique<Table<First>>(layout, layout.bound, layout.buckets);
    match<1>(layout, params, hashes, proof, work, *xs, *t1);
    xs.reset();
    auto t2 = std::make_unique<Table<Second>>(layout, layout.bound, layout.buckets);
    match<2>(layout, params, hashes, proof, work, *t1, *t2);
    t1.reset();
    Table<uint64_t> fragments(layout, layout.bound, layout.buckets);
    match<3>(layout, params, hashes, proof, work, *t2, fragments);
    t2.reset();

    std::vector<uint64_t> result;
    result.reserve(fragments.size());
    std::vector<size_t> offsets(layout.buckets + 1);
    for (unsigned bin = 0; bin < layout.buckets; ++bin) {
        work.poll();
        fragments.visit(bin, [&](auto block) { result.insert(result.end(), block.begin(), block.end()); });
        offsets[bin + 1] = result.size();
    }
    work.parallel(layout.buckets, [&](unsigned, size_t bin) {
        sort_fragments(std::span(result).subspan(offsets[bin], offsets[bin + 1] - offsets[bin]),
                       layout.k * 2 - layout.bucket_bits, work);
    });
    work.poll();
    return result;
}

} // namespace pos2gpu::cpu
