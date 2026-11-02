// Bounded SYCL compact transport kernels for the experimental group producer.
#pragma once
#include "host/GroupPacked.hpp"
#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>

#include <algorithm>
#include <stdexcept>

namespace group_grow::detail
{
constexpr uint64_t packed_threads = 256;
constexpr uint64_t packed_values_per_group = packed_threads * 8;
constexpr unsigned packed_chunk_bits = 6;

struct PackedTransferLayout
{
    uint64_t boundary_count, boundary_words, validation_groups, metadata_words, payload_words;
    PackedTransferLayout(int k, uint64_t count)
    {
        if (k < 18 || k > 28 || (k & 1) || count > UINT32_MAX || count > SIZE_MAX / sizeof(uint64_t))
            throw std::invalid_argument("GPU compact transfer requires supported k and narrow fragment indexes");
        static_assert(sizeof(PackedMemberValidation) % sizeof(uint64_t) == 0);
        static_assert(alignof(PackedMemberValidation) <= alignof(uint64_t));
        // k and count are bounded before shifts, products, or narrowing. The
        // word-aligned summary offset also leaves boundary padding unexposed.
        boundary_count = (1ull << (k - packed_chunk_bits)) + 1;
        boundary_words = (boundary_count * sizeof(uint32_t) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        validation_groups = (count + packed_values_per_group - 1) / packed_values_per_group;
        metadata_words = boundary_words + validation_groups * (sizeof(PackedMemberValidation) / sizeof(uint64_t));
        payload_words = (count * uint64_t(k + packed_chunk_bits) + 63) / 64;
    }
};

inline sycl::event launch_validate_packed_fragments(int k, uint64_t const* source, uint64_t count,
                                                 uint64_t* scratch, sycl::queue& queue)
{
    PackedTransferLayout const layout(k, count);
    auto const zero = queue.memset(scratch, 0, layout.metadata_words * sizeof(uint64_t));
    if (!count) return zero;
    auto* const boundaries = reinterpret_cast<uint32_t*>(scratch);
    auto* const summaries = reinterpret_cast<PackedMemberValidation*>(scratch + layout.boundary_words);
    uint64_t const chunks = layout.boundary_count - 1;
    unsigned const width = unsigned(k) + packed_chunk_bits;
    uint64_t const end = 1ull << (2 * k);
    uint64_t const maximum_delta = uint64_t(72 - k) * (178ull << (k - 8));
    return queue.submit([&](sycl::handler& handler)
    {
        handler.depends_on(zero);
        sycl::local_accessor<PackedMemberValidation, 1> local(sycl::range<1>(packed_threads), handler);
        handler.parallel_for(sycl::nd_range<1>{layout.validation_groups * packed_threads, packed_threads},
            [=](sycl::nd_item<1> item)
        {
            uint64_t const lane = item.get_local_id(0), group = item.get_group(0);
            uint64_t const first = group * packed_values_per_group;
            uint64_t const group_end = count < first + packed_values_per_group ? count : first + packed_values_per_group;
            PackedMemberValidation validation;
            auto set_boundary = [=](uint64_t chunk, uint32_t offset)
            {
                // Valid sorted input has one writer per boundary. Atomic
                // stores also keep malformed, repeated transitions race-free
                // until their validation error is consumed on the host.
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space> boundary(boundaries[chunk]);
                boundary.store(offset);
            };
            for (uint64_t index = first + lane; index < group_end; index += packed_threads)
            {
                uint64_t const value = source[index];
                ++validation.checked_count;
                if (value >= end) validation.flags |= PackedMemberValidation::out_of_range;
                uint64_t const previous = index ? source[index - 1] : 0;
                if (index && value < previous) validation.flags |= PackedMemberValidation::unsorted;
                uint64_t const chunk = value >> width, previous_chunk = previous >> width;
                if (index && value >= previous && chunk == previous_chunk && value - previous >= maximum_delta)
                    validation.flags |= PackedMemberValidation::wide_delta;
                // Each ordered chunk transition has one writer, even across
                // workgroups. Invalid inputs are reported without indexing
                // their out-of-range chunks into the admitted boundary array.
                if (value < end)
                {
                    if (!index)
                        for (uint64_t next = 0; next <= chunk; ++next) set_boundary(next, 0);
                    else if (previous < end && value >= previous && chunk > previous_chunk)
                        for (uint64_t next = previous_chunk + 1; next <= chunk; ++next)
                            set_boundary(next, uint32_t(index));
                    if (index + 1 == count)
                        for (uint64_t next = chunk + 1; next <= chunks; ++next) set_boundary(next, uint32_t(count));
                }
            }
            local[lane] = validation;
            item.barrier(sycl::access::fence_space::local_space);
            for (uint64_t stride = packed_threads / 2; stride; stride /= 2)
            {
                if (lane < stride)
                {
                    local[lane].checked_count += local[lane + stride].checked_count;
                    local[lane].flags |= local[lane + stride].flags;
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (!lane) summaries[group] = local[0];
        });
    });
}

inline sycl::event launch_pack_fragment_words(int k, uint64_t const* source, uint64_t count,
                                            uint64_t word_offset, uint64_t words,
                                            uint64_t* destination, sycl::queue& queue)
{
    PackedTransferLayout const layout(k, count);
    if (!words || word_offset > layout.payload_words || words > layout.payload_words - word_offset)
        throw std::invalid_argument("GPU compact payload tile exceeds the fragment count");
    unsigned const width = unsigned(k) + packed_chunk_bits;
    uint64_t const mask = (1ull << width) - 1;
    uint64_t const global = ((words + packed_threads - 1) / packed_threads) * packed_threads;
    return queue.parallel_for(sycl::nd_range<1>{global, packed_threads}, [=](sycl::nd_item<1> item)
    {
        uint64_t const output = item.get_global_id(0);
        if (output >= words) return;
        uint64_t const bit = (word_offset + output) * 64;
        uint64_t index = bit / width;
        unsigned const shift = unsigned(bit % width);
        uint64_t word = (source[index] & mask) >> shift;
        unsigned used = width - shift;
        while (used < 64 && ++index < count)
        {
            word |= (source[index] & mask) << used;
            used += width;
        }
        destination[output] = word;
    });
}
} // namespace group_grow::detail
