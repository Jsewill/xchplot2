#include "host/GroupGpuPacked.hpp"
#include "common/Utils.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace group_grow;

void require(bool value, char const* message)
{
    if (!value) throw std::runtime_error(message);
}

void check(sycl::queue& queue, int k, std::span<uint64_t const> raw, uint32_t expected_flags)
{
    detail::PackedTransferLayout const layout(k, raw.size());
    auto free_device = [&](uint64_t* pointer) { if (pointer) sycl::free(pointer, queue); };
    std::unique_ptr<uint64_t, decltype(free_device)> source(
        sycl::malloc_device<uint64_t>(std::max(size_t(1), raw.size()), queue), free_device);
    std::unique_ptr<uint64_t, decltype(free_device)> scratch(
        sycl::malloc_device<uint64_t>(std::max(layout.metadata_words, uint64_t(31)), queue), free_device);
    struct Drain
    {
        sycl::queue& queue;
        ~Drain() { try { queue.wait_and_throw(); } catch (...) {} }
    } drain{queue};
    require(source && scratch, "compact kernel test allocation failed");
    if (!raw.empty()) queue.memcpy(source.get(), raw.data(), raw.size_bytes()).wait_and_throw();
    detail::launch_validate_packed_fragments(k, source.get(), raw.size(), scratch.get(), queue).wait_and_throw();
    std::vector<uint64_t> metadata(layout.metadata_words);
    queue.memcpy(metadata.data(), scratch.get(), metadata.size() * sizeof(uint64_t)).wait_and_throw();
    PackedMemberValidation validation;
    for (uint64_t group = 0; group < layout.validation_groups; ++group)
    {
        PackedMemberValidation summary;
        std::memcpy(&summary, metadata.data() + layout.boundary_words +
            group * (sizeof(summary) / sizeof(uint64_t)), sizeof(summary));
        uint64_t const expected = std::min(detail::packed_values_per_group,
            uint64_t(raw.size()) - group * detail::packed_values_per_group);
        require(summary.checked_count == expected && !(summary.flags & ~PackedMemberValidation::known_flags),
                "compact device validation group coverage differs");
        validation.checked_count += summary.checked_count;
        validation.flags |= summary.flags;
    }
    require(validation.checked_count == raw.size() && validation.flags == expected_flags,
            "compact device validation summary differs");
    if (expected_flags & (PackedMemberValidation::unsorted | PackedMemberValidation::out_of_range)) return;

    unsigned const width = unsigned(k) + detail::packed_chunk_bits;
    uint64_t const mask = (1ull << width) - 1;
    size_t first = 0;
    for (uint64_t chunk = 0; chunk < layout.boundary_count; ++chunk)
    {
        while (first < raw.size() && (raw[first] >> width) < chunk) ++first;
        uint32_t actual;
        std::memcpy(&actual, reinterpret_cast<uint8_t const*>(metadata.data()) + chunk * sizeof(actual), sizeof(actual));
        require(actual == first, "compact device boundary differs across an empty or populated chunk");
    }
    BitWriter reference;
    for (uint64_t value : raw) reference.append(value & mask, width);
    std::vector<uint64_t> canonical(layout.payload_words), packed(layout.payload_words);
    if (!canonical.empty()) std::memcpy(canonical.data(), reference.asBytes().data(), reference.asBytes().size());
    for (uint64_t word = 0; word < layout.payload_words;)
    {
        uint64_t const words = std::min(layout.payload_words - word, uint64_t(1 + word % 31));
        detail::launch_pack_fragment_words(k, source.get(), raw.size(), word, words, scratch.get(), queue).wait_and_throw();
        queue.memcpy(packed.data() + word, scratch.get(), words * sizeof(uint64_t)).wait_and_throw();
        word += words;
    }
    require(packed == canonical, "compact device words differ from the pinned little-endian BitWriter");
}

int main() try
{
    sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    for (int k : {18, 20, 22, 24, 26, 28})
    {
        unsigned const width = unsigned(k) + detail::packed_chunk_bits;
        uint64_t const range = 1ull << width;
        uint64_t const chunks = 1ull << (k - detail::packed_chunk_bits);
        for (size_t count : {0, 1, 2, 3, 63, 64, 65, 129})
        {
            std::vector<uint64_t> raw;
            for (size_t index = 0; index < count; ++index)
            {
                uint64_t const chunk = index < 43 ? 1 : (index < 86 ? chunks / 2 : chunks - 1);
                raw.push_back(chunk * range + (index % 43 / 2) * (range / 64));
            }
            check(queue, k, raw, 0);
        }
        // Coverage and predecessor checks cross both workgroup and payload
        // tile boundaries; reverse a pair at the workgroup boundary itself.
        std::vector<uint64_t> ordered(detail::packed_values_per_group + 3);
        for (size_t index = 0; index < ordered.size(); ++index) ordered[index] = index;
        check(queue, k, ordered, 0);
        std::swap(ordered[detail::packed_values_per_group - 1], ordered[detail::packed_values_per_group]);
        check(queue, k, ordered, PackedMemberValidation::unsorted);
        std::array<uint64_t, 3> const repeated_transition{range, 0, range};
        check(queue, k, repeated_transition, PackedMemberValidation::unsorted);
        uint64_t const out_of_range = 1ull << (2 * k);
        check(queue, k, {&out_of_range, 1}, PackedMemberValidation::out_of_range);
        uint64_t const maximum = UINT64_MAX;
        check(queue, k, {&maximum, 1}, PackedMemberValidation::out_of_range);
        uint64_t const delta = uint64_t(72 - k) * (178ull << (k - 8));
        std::array<uint64_t, 2> const valid_delta{0, delta - 1}, wide_delta{0, delta};
        check(queue, k, valid_delta, 0);
        check(queue, k, wide_delta, PackedMemberValidation::wide_delta);
    }
    std::cout << "bounded GPU compact kernel checks passed\n";
}
catch (std::exception const& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
