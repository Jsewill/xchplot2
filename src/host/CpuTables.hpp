// Native C++20 CPU producer. Only the proof-format primitives come from pos2-chip.
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

template<class Left, class Right, class F>
void pair(ProofCore& proof, Left l, Right r, F&& emit)
{
    if constexpr (std::is_same_v<Left, Xs>) {
        if (auto hit = proof.pairing_t1(l.x, r.x))
            emit(First{hit->match_info, hit->meta_lo, hit->meta_hi}, hit->match_info);
    } else if constexpr (std::is_same_v<Left, First>) {
        if (auto hit = proof.pairing_t2(meta(l), meta(r)))
            emit(Second{hit->match_info, uint32_t(hit->meta), uint32_t(hit->meta >> 32), hit->x_bits}, hit->match_info);
    } else {
        if (auto hit = proof.pairing_t3(meta(l), meta(r), l.xbits, r.xbits))
            emit(hit->proof_fragment, hit->proof_fragment);
    }
}

template<class Row>
struct RightIndex {
    std::unique_ptr<Row[]> nodes;
    std::vector<uint32_t> heads;
};

template<unsigned Stage, class Row, class Output>
void match(Layout const& layout, PlotProofParams const& params, ProofCore& proof, Work& work,
           Table<Row> const& input, Table<Output>& output)
{
    const unsigned shift = layout.k - layout.bucket_bits;
    const unsigned output_shift = (Stage == 3 ? 2 * layout.k : layout.k) - layout.bucket_bits;
    const unsigned section_bins = layout.buckets / params.get_num_sections();
    const unsigned keys = unsigned(params.get_num_match_keys(Stage));
    const unsigned target_bits = unsigned(params.get_num_match_target_bits(Stage));
    const uint32_t mask = (uint32_t{1} << target_bits) - 1;
    const unsigned query_bins = std::max(1u, section_bins / keys);
    const unsigned head_count = std::min(layout.heads, mask + 1);
    Table<Row> queries(layout, layout.query_rows, query_bins);
    std::vector<RightIndex<Row>> scratch(layout.lanes);

    for (unsigned section = 0; section < params.get_num_sections(); ++section) {
        std::vector<std::span<const Row>> left;
        for (unsigned i = 0; i < section_bins; ++i)
            input.visit(section * section_bins + i, [&](auto block) { left.push_back(block); });
        const unsigned right_section = proof.matching_section(section);
        for (unsigned key = 0; key < keys; ++key) {
            const uint32_t base = (right_section * keys + key) << target_bits;
            const unsigned first_bin = base >> shift;
            // A section can be larger than its expected occupancy. Work in
            // bounded groups of blocks while retaining every row and duplicate.
            const size_t batch = layout.query_rows / layout.block;
            for (size_t first = 0; first < left.size(); first += batch) {
                queries.reset();
                const size_t count = std::min(batch, left.size() - first);
                constexpr unsigned blocks_per_task = 8;
                work.parallel((count + blocks_per_task - 1) / blocks_per_task, [&](unsigned lane, size_t task) {
                    for (size_t i = task * blocks_per_task; i < std::min(count, (task + 1) * blocks_per_task); ++i) {
                        if (i == task * blocks_per_task || (Stage == 1 && params.get_strength() > 8)) work.poll();
                        for (auto row : left[first + i]) {
                            row.info = proof.matching_target(Stage, meta(row), key);
                            queries.append(lane, row.info >> shift, row);
                        }
                    }
                });
                work.parallel(query_bins, [&](unsigned lane, size_t bin) {
                    auto& index = scratch[lane];
                    index.heads.assign(head_count, 0);
                    if (!index.nodes) index.nodes = std::make_unique_for_overwrite<Row[]>(layout.right_tile);
                    uint32_t filled = 0;
                    unsigned blocks_seen = 0;
                    auto join = [&] {
                        if (!filled) return;
                        queries.visit(unsigned(bin), [&](auto block) {
                            if (!(blocks_seen++ % 16)) work.poll();
                            for (const auto l : block) {
                                // The bucket fixes the high target bits. The head index
                                // fixes every remaining bit, including for duplicate rows.
                                for (auto j = index.heads[l.info & (head_count - 1)]; j; ) {
                                    const auto& r = index.nodes[j - 1];
                                    pair(proof, l, r, [&](Output result, auto sort_key) {
                                        output.append(lane, unsigned(sort_key >> output_shift), result);
                                    });
                                    j = r.info;
                                }
                            }
                        });
                    };
                    input.visit(first_bin + unsigned(bin), [&](auto block) {
                        if (!(blocks_seen++ % 16)) work.poll();
                        for (auto r : block) {
                            if ((r.info & ~mask) != base) continue;
                            if (filled == layout.right_tile) {
                                join();
                                std::fill(index.heads.begin(), index.heads.end(), 0);
                                filled = 0;
                            }
                            auto& head = index.heads[r.info & (head_count - 1)];
                            r.info = head;
                            index.nodes[filled++] = r;
                            head = filled;
                        }
                    });
                    join();
                });
            }
        }
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
    if (layout.peak_bytes() > ram_bytes) throw std::length_error("CPU plotting exceeds the available RAM grant");
    detail::Work work(layout.lanes, check);
    work.poll();
    ProofCore proof(params);
    using namespace detail;
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
    match<1>(layout, params, proof, work, *xs, *t1);
    xs.reset();
    auto t2 = std::make_unique<Table<Second>>(layout, layout.bound, layout.buckets);
    match<2>(layout, params, proof, work, *t1, *t2);
    t1.reset();
    Table<uint64_t> fragments(layout, layout.bound, layout.buckets);
    match<3>(layout, params, proof, work, *t2, fragments);
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
