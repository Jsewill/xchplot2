// Retained member storage for the experimental ordinary-group grow path.
#pragma once
#include "host/GroupFormat.hpp"
#include "host/GroupPacked.hpp"
#include "pos2_keygen.h"

#include <atomic>
#include <bit>
#include <iterator>
#include <optional>
#include <utility>

namespace group_grow
{
// Fragment counts usually fit in a narrow index. Generic raw inputs can be
// larger, so select the representation from the actual count before filling it.
class FragmentBoundaries
{
public:
    FragmentBoundaries() = default;
    FragmentBoundaries(FragmentBoundaries&& other) noexcept { *this = std::move(other); }
    FragmentBoundaries& operator=(FragmentBoundaries&& other) noexcept
    {
        if (this == &other) return *this;
        narrow_values_ = std::move(other.narrow_values_);
        wide_ = std::move(other.wide_);
        narrow_ = other.narrow_;
        capacity_ = std::exchange(other.capacity_, 0);
        return *this;
    }
    FragmentBoundaries(FragmentBoundaries const&) = delete;
    FragmentBoundaries& operator=(FragmentBoundaries const&) = delete;
    static uint64_t allocation_bytes(size_t chunks, size_t count)
    {
        size_t const width = count <= UINT32_MAX ? sizeof(uint32_t) : sizeof(size_t);
        group_tools::require(chunks < SIZE_MAX && chunks + 1 <= SIZE_MAX / width,
                             "member boundary length overflow");
        return uint64_t(chunks + 1) * width;
    }
    void reset(size_t chunks, size_t count)
    {
        (void)allocation_bytes(chunks, count);
        bool const narrow = count <= UINT32_MAX;
        if (narrow == narrow_ && capacity_ == chunks + 1) return;
        narrow_values_.reset();
        wide_.reset();
        capacity_ = 0;
        narrow_ = narrow;
        if (narrow_) narrow_values_.reset(new uint32_t[chunks + 1]);
        else wide_.reset(new size_t[chunks + 1]);
        capacity_ = chunks + 1;
    }
    size_t operator[](size_t index) const
    { return narrow_ ? size_t(narrow_values_[index]) : wide_[index]; }
    void set(size_t index, size_t value)
    {
        group_tools::require(!narrow_ || value <= UINT32_MAX, "member boundary index overflow");
        if (narrow_) narrow_values_[index] = uint32_t(value);
        else wide_[index] = value;
    }
    uint64_t storage_bytes() const
    { return uint64_t(capacity_) * (narrow_ ? sizeof(uint32_t) : sizeof(size_t)); }
    bool narrow() const { return narrow_; }
    void clear() { *this = FragmentBoundaries{}; }
private:
    bool narrow_ = true;
    std::unique_ptr<uint32_t[]> narrow_values_;
    std::unique_ptr<size_t[]> wide_;
    size_t capacity_ = 0;
};

// A codec region always knows its chunk base. Packed fragments can therefore
// be read directly using the same little-endian word extraction as BitReader.
// No temporary unpacked member or window is needed.
class FragmentRegion
{
public:
    static_assert(18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS >= 13 &&
                  28 + PlotGroupFile::PROOFS_PER_CHUNK_BITS < 64);
    FragmentRegion() = default;
    FragmentRegion(std::span<uint64_t const> values) : data_(values.data()), count_(values.size()) {}
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    uint64_t operator[](size_t index) const
    {
        unsigned const width = unsigned(packed_base_ & 63);
        if (!width) return data_[index];
        size_t const bit = size_t((packed_base_ >> 6) & 63) + index * width;
        size_t const word = bit >> 6;
        unsigned const shift = unsigned(bit & 63);
        uint64_t value = data_[word] >> shift;
        if (width > 64 - shift) value |= data_[word + 1] << (64 - shift);
        return (packed_base_ & ~uint64_t(8191)) + (value & ((1ull << width) - 1));
    }
    // A bounded scratch reader owns these little-endian low-field words until
    // every codec task borrowing the region has joined.
    static FragmentRegion packed(std::span<uint64_t const> words, uint8_t k, uint64_t chunk,
                                 unsigned shift, size_t count, bool validated);
    uint64_t front() const { return (*this)[0]; }
    uint64_t back() const { return (*this)[count_ - 1]; }
    bool validated_deltas(uint8_t k) const
    {
        return (packed_base_ & 63) == k + PlotGroupFile::PROOFS_PER_CHUNK_BITS &&
               (packed_base_ & (1ull << 12));
    }
    class Iterator
    {
    public:
        using value_type = uint64_t;
        using difference_type = std::ptrdiff_t;
        using reference = uint64_t;
        using pointer = void;
        using iterator_category = std::random_access_iterator_tag;
        Iterator() = default;
        Iterator(FragmentRegion const* region, size_t index) : region_(region), index_(index) {}
        uint64_t operator*() const { return (*region_)[index_]; }
        uint64_t operator[](difference_type offset) const { return *(*this + offset); }
        Iterator& operator++() { ++index_; return *this; }
        Iterator operator++(int) { auto prior = *this; ++*this; return prior; }
        Iterator& operator--() { --index_; return *this; }
        Iterator operator--(int) { auto prior = *this; --*this; return prior; }
        Iterator& operator+=(difference_type offset) { index_ += offset; return *this; }
        Iterator& operator-=(difference_type offset) { index_ -= offset; return *this; }
        friend Iterator operator+(Iterator it, difference_type offset) { it += offset; return it; }
        friend Iterator operator+(difference_type offset, Iterator it) { return it + offset; }
        friend Iterator operator-(Iterator it, difference_type offset) { it -= offset; return it; }
        friend difference_type operator-(Iterator a, Iterator b)
        { return difference_type(a.index_) - difference_type(b.index_); }
        friend bool operator==(Iterator a, Iterator b) { return a.index_ == b.index_; }
        friend auto operator<=>(Iterator a, Iterator b) { return a.index_ <=> b.index_; }
    private:
        FragmentRegion const* region_ = nullptr;
        size_t index_ = 0;
    };
    Iterator begin() const { return {this, 0}; }
    Iterator end() const { return {this, count_}; }
private:
    friend class MemberView;
    uint64_t const* data_ = nullptr;
    size_t count_ = 0;
    // Chunk bases have k + PROOFS_PER_CHUNK_BITS trailing zeroes. Reuse only
    // their low bits for field width and initial word shift, keeping bounded
    // window descriptors small without narrowing any file/fragment value.
    uint64_t packed_base_ = 0;
};

inline FragmentRegion FragmentRegion::packed(std::span<uint64_t const> words, uint8_t k, uint64_t chunk,
                                             unsigned shift, size_t count, bool validated)
{
    unsigned const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
    group_tools::require(k >= 18 && k <= 28 && !(k & 1) && shift < 64 &&
        chunk < PlotGroupFile::getChunkCountForK(k) && count <= (SIZE_MAX - shift) / width &&
        (shift + count * width) / 64 + ((shift + count * width) % 64 != 0) <= words.size(),
        "invalid packed scratch region");
    FragmentRegion result;
    result.data_ = words.data(); result.count_ = count;
    result.packed_base_ = (chunk << width) | (uint64_t(shift) << 6) | width |
        (validated ? 1ull << 12 : 0);
    return result;
}

class MemberRegions
{
public:
    MemberRegions() = default;
    MemberRegions(std::span<uint64_t const> const* raw, size_t count)
        : data_(raw), count_(count), raw_(true) {}
    MemberRegions(FragmentRegion const* regions, size_t count) : data_(regions), count_(count) {}
    MemberRegions(std::span<std::span<uint64_t const> const> raw)
        : MemberRegions(raw.data(), raw.size()) {}
    MemberRegions(std::span<std::span<uint64_t const>> raw)
        : MemberRegions(raw.data(), raw.size()) {}
    MemberRegions(std::vector<std::span<uint64_t const>> const& raw)
        : MemberRegions(raw.data(), raw.size()) {}
    template <size_t N> MemberRegions(std::array<std::span<uint64_t const>, N> const& raw)
        : MemberRegions(raw.data(), raw.size()) {}
    MemberRegions(std::span<FragmentRegion const> regions)
        : MemberRegions(regions.data(), regions.size()) {}
    MemberRegions(std::span<FragmentRegion> regions)
        : MemberRegions(regions.data(), regions.size()) {}
    MemberRegions(std::vector<FragmentRegion> const& regions)
        : MemberRegions(regions.data(), regions.size()) {}
    template <size_t N> MemberRegions(std::array<FragmentRegion, N> const& regions)
        : MemberRegions(regions.data(), regions.size()) {}
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    FragmentRegion operator[](size_t index) const
    {
        return raw_ ? FragmentRegion(static_cast<std::span<uint64_t const> const*>(data_)[index])
                    : static_cast<FragmentRegion const*>(data_)[index];
    }
private:
    void const* data_ = nullptr;
    size_t count_ = 0;
    bool raw_ = false;
};

class Member;
class MemberView
{
public:
    MemberView() = default;
    MemberView(std::span<uint64_t const> raw) : raw_(raw), count_(raw.size()) {}
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    bool compact() const { return boundaries_ != nullptr; }
    uint8_t k() const { return k_; }
    uint64_t immutable_id() const { return id_; }
    std::span<uint64_t const> raw() const { return raw_; }
    size_t boundary(size_t chunk) const { return (*boundaries_)[chunk]; }
    FragmentRegion region(uint64_t chunk, size_t first, size_t count) const
    {
        group_tools::require(first <= count_ && count <= count_ - first, "member region exceeds bounds");
        if (!compact()) return raw_.subspan(first, count);
        group_tools::require(chunk < PlotGroupFile::getChunkCountForK(k_) &&
            first == boundary(size_t(chunk)) && count == boundary(size_t(chunk + 1)) - first,
            "packed member chunk mismatch");
        FragmentRegion result;
        unsigned const width = k_ + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        size_t const bit = first * width;
        result.data_ = packed_ ? packed_ + (bit >> 6) : nullptr;
        result.count_ = count;
        result.packed_base_ = (chunk << width) | ((bit & 63) << 6) | width |
            (validated_deltas_ ? 1ull << 12 : 0);
        return result;
    }
    FragmentRegion region(uint64_t chunk) const
    {
        group_tools::require(compact() && chunk < PlotGroupFile::getChunkCountForK(k_),
                             "invalid packed member chunk");
        size_t const first = boundary(size_t(chunk));
        return region(chunk, first, boundary(size_t(chunk + 1)) - first);
    }
private:
    friend class Member;
    std::span<uint64_t const> raw_;
    uint64_t const* packed_ = nullptr;
    FragmentBoundaries const* boundaries_ = nullptr;
    size_t count_ = 0;
    uint64_t id_ = 0;
    uint8_t k_ = 0;
    bool validated_deltas_ = false;
};

class Members;
class CompactMemberBuilder;
class CompactMemberReceiver;

// Preserve the exact batch framing while packing incoming tiles. Every prefix
// owns an independent SHA snapshot, so a rejected suffix can be removed before
// extending that prefix. A builder commits only after its member is complete.
// Serialize mutation with all cache readers; concurrent producer/consumer
// batches use separate caches.
class MemberBatchHash
{
    struct FreeState
    {
        void operator()(pos2_sha256_state* state) const noexcept { pos2_sha256_state_free(state); }
    };
    using State = std::unique_ptr<pos2_sha256_state, FreeState>;
    struct Prefix
    {
        uint64_t id = 0;
        std::array<uint8_t, 32> digest{};
        State state;
    };
public:
    explicit MemberBatchHash(size_t capacity) : capacity_(capacity)
    {
        (void)memory_bytes(capacity);
        if (capacity) prefixes_ = std::make_unique<Prefix[]>(capacity);
        base_.reset(pos2_sha256_state_new());
        group_tools::require(bool(base_), "cannot allocate group hash state");
    }
    // Builders borrow this fixed address. Own the cache through a unique_ptr
    // when transferring incoming/prefetched batch ownership between controller
    // slots; destroying it while a builder is active is a caller lifetime error.
    MemberBatchHash(MemberBatchHash const&) = delete;
    MemberBatchHash& operator=(MemberBatchHash const&) = delete;
    MemberBatchHash(MemberBatchHash&&) = delete;
    MemberBatchHash& operator=(MemberBatchHash&&) = delete;
    static uint64_t memory_bytes(size_t capacity)
    {
        group_tools::require(capacity <= UINT16_MAX, "group hash capacity exceeds membership limit");
        // Include the empty base, an active builder clone, the temporary digest
        // clone, allocator allowances, and all committed prefix snapshots.
        return sizeof(MemberBatchHash) + uint64_t(capacity) * sizeof(Prefix) + 64 +
            uint64_t(capacity + 3) * pos2_sha256_state_bytes();
    }
    size_t size() const { return size_; }
    void reset() { truncate(0); }
    void truncate(size_t count)
    {
        group_tools::require(!active_ && count <= size_, "invalid group hash truncation");
        for (size_t i = count; i < size_; ++i) prefixes_[i] = Prefix{};
        size_ = count;
    }
    std::optional<std::array<uint8_t, 32>> digest_for(Members members) const;
private:
    friend class CompactMemberBuilder;
    State start(uint64_t count)
    {
        group_tools::require(!active_ && size_ < capacity_, "group hash builder capacity exceeded");
        State state(pos2_sha256_state_clone(size_ ? prefixes_[size_ - 1].state.get() : base_.get()));
        group_tools::require(bool(state), "cannot clone group hash state");
        group_tools::require(pos2_sha256_state_update(state.get(),
            reinterpret_cast<uint8_t const*>(&count), sizeof(count)) == POS2_OK,
            "cannot hash incoming member size");
        active_ = true;
        return state;
    }
    void commit(uint64_t id, State state)
    {
        group_tools::require(active_ && size_ < capacity_ && id, "invalid group hash commit");
        std::array<uint8_t, 32> digest{};
        group_tools::require(pos2_sha256_state_digest(state.get(), digest.data()) == POS2_OK,
                             "cannot finish incoming member hash");
        prefixes_[size_].id = id;
        prefixes_[size_].digest = digest;
        prefixes_[size_].state = std::move(state);
        ++size_;
        active_ = false;
    }
    std::unique_ptr<Prefix[]> prefixes_;
    State base_;
    size_t capacity_ = 0, size_ = 0;
    bool active_ = false;
};

class Member
{
public:
    Member() = default;
    explicit Member(std::vector<uint64_t>&& raw) : raw_(std::move(raw)) {}
    Member(Member&& other) noexcept { *this = std::move(other); }
    Member& operator=(Member&& other) noexcept
    {
        if (this == &other) return *this;
        raw_ = std::move(other.raw_);
        packed_ = std::move(other.packed_);
        packed_words_ = std::exchange(other.packed_words_, 0);
        boundaries_ = std::move(other.boundaries_);
        borrowed_ = std::exchange(other.borrowed_, {});
        count_ = std::exchange(other.count_, 0);
        id_ = std::exchange(other.id_, 0);
        k_ = std::exchange(other.k_, 0);
        validated_deltas_ = std::exchange(other.validated_deltas_, false);
        return *this;
    }
    Member(Member const&) = delete;
    Member& operator=(Member const&) = delete;
    // The caller keeps the producer's lease alive through every engine call.
    static Member borrowed(std::span<uint64_t const> raw)
    { Member result; result.borrowed_ = raw; return result; }
    size_t size() const { return k_ ? count_ : (borrowed_.empty() ? raw_.size() : borrowed_.size()); }
    bool empty() const { return size() == 0; }
    bool compact() const { return k_ != 0; }
    MemberView view() const
    {
        if (!compact()) return borrowed_.empty() ? std::span<uint64_t const>(raw_) : borrowed_;
        MemberView result;
        result.packed_ = packed_.get();
        result.boundaries_ = &boundaries_;
        result.count_ = count_;
        result.id_ = id_;
        result.k_ = k_;
        result.validated_deltas_ = validated_deltas_;
        return result;
    }
    uint64_t storage_bytes() const
    {
        return uint64_t(raw_.capacity()) * sizeof(uint64_t) + uint64_t(packed_words_) * sizeof(uint64_t)
            + boundaries_.storage_bytes();
    }
    static uint64_t compact_bytes(uint8_t k, size_t count)
    {
        auto const words = packed_words(k, count);
        uint64_t const payload = uint64_t(words) * sizeof(uint64_t);
        uint64_t const boundaries = FragmentBoundaries::allocation_bytes(
            size_t(PlotGroupFile::getChunkCountForK(k)), count);
        group_tools::require(boundaries <= UINT64_MAX - payload, "packed member length overflow");
        return payload + boundaries;
    }
    // Conversion is opt-in. Admit compact_bytes() in addition to all resident
    // raw/producer allocations first; both representations coexist until done.
    // An already compact member keeps its identity; hash is used only when a
    // new compact representation is constructed.
    void compact(uint8_t k, MemberBatchHash* hash = nullptr);
private:
    friend class CompactMemberBuilder;
    friend class CompactMemberReceiver;
    void identify()
    {
        // IDs are never recycled, including when allocator addresses repeat.
        // Exhaustion permanently prevents issuing a wrapped identity.
        static std::atomic<uint64_t> next_id{1};
        uint64_t id = next_id.load(std::memory_order_relaxed);
        for (;;)
        {
            group_tools::require(id != UINT64_MAX, "compact member identity exhausted");
            if (next_id.compare_exchange_weak(id, id + 1, std::memory_order_relaxed)) break;
        }
        id_ = id;
    }
    static size_t packed_words(uint8_t k, size_t count)
    {
        group_tools::require(k >= 18 && k <= 28 && !(k & 1), "invalid packed member k");
        size_t const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        group_tools::require(count <= SIZE_MAX / width, "packed member bit length overflow");
        size_t const bits = count * width;
        size_t const words = bits / 64 + (bits % 64 != 0);
        group_tools::require(words <= SIZE_MAX / sizeof(uint64_t), "packed member length overflow");
        return words;
    }
    std::vector<uint64_t> raw_;
    std::unique_ptr<uint64_t[]> packed_;
    size_t packed_words_ = 0;
    std::span<uint64_t const> borrowed_;
    FragmentBoundaries boundaries_;
    size_t count_ = 0;
    uint64_t id_ = 0;
    uint8_t k_ = 0;
    bool validated_deltas_ = true;
};

// Supports ordered GPU drain tiles as well as a single owned raw member.
// Exact allocations avoid an unmodelled vector-growth replacement peak.
class CompactMemberBuilder
{
public:
    CompactMemberBuilder(uint8_t k, size_t count, MemberBatchHash* hash = nullptr) : expected_(count)
    {
        (void)Member::compact_bytes(k, count);
        member_.k_ = k;
        member_.count_ = count;
        member_.packed_words_ = Member::packed_words(k, count);
        // The accumulator writes complete words once and keeps the unwritten
        // tail zero-padded, so the payload needs no initial clearing pass.
        if (member_.packed_words_) member_.packed_.reset(new uint64_t[member_.packed_words_]);
        next_word_ = member_.packed_.get();
        member_.boundaries_.reset(size_t(PlotGroupFile::getChunkCountForK(k)), count);
        member_.boundaries_.set(0, 0);
        if (hash)
        {
            hash_state_ = hash->start(count);
            hash_ = hash;
        }
    }
    ~CompactMemberBuilder() { if (hash_) hash_->active_ = false; }
    CompactMemberBuilder(CompactMemberBuilder const&) = delete;
    CompactMemberBuilder& operator=(CompactMemberBuilder const&) = delete;
    CompactMemberBuilder(CompactMemberBuilder&&) = delete;
    CompactMemberBuilder& operator=(CompactMemberBuilder&&) = delete;
    void append(std::span<uint64_t const> values)
    {
        group_tools::require(!finished_ && !failed_ && values.size() <= expected_ - written_,
                             "packed member count exceeds declaration");
        try
        {
            unsigned const width = member_.k_ + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
            uint64_t const end = 1ull << (2 * member_.k_);
            uint64_t const mask = (1ull << width) - 1;
            uint64_t const maximum_delta = uint64_t(72 - member_.k_) * (178ull << (member_.k_ - 8));
            for (uint64_t value : values)
            {
                group_tools::require(value < end && (!written_ || value >= previous_),
                                     "empty, unsorted or out-of-range incoming member");
                size_t const chunk = size_t(value >> width);
                if (written_ && chunk == chunk_ && value - previous_ >= maximum_delta)
                    member_.validated_deltas_ = false;
                while (chunk_ < chunk) member_.boundaries_.set(++chunk_, written_);
                uint64_t const low = value & mask;
                packed_word_ |= low << packed_bits_;
                unsigned const remaining = 64 - packed_bits_;
                if (width >= remaining)
                {
                    *next_word_++ = packed_word_;
                    packed_word_ = width > remaining ? low >> remaining : 0;
                    packed_bits_ = width - remaining;
                }
                else packed_bits_ += width;
                previous_ = value;
                ++written_;
            }
            if (hash_)
            {
                group_tools::require(values.size() <= SIZE_MAX / sizeof(uint64_t) &&
                    pos2_sha256_state_update(hash_state_.get(), reinterpret_cast<uint8_t const*>(values.data()),
                        values.size_bytes()) == POS2_OK, "cannot hash incoming member fragments");
            }
        }
        catch (...) { failed_ = true; throw; }
    }
    Member finish()
    {
        group_tools::require(!finished_ && !failed_ && written_ == expected_, "incomplete packed member");
        size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(member_.k_));
        while (chunk_ < chunks) member_.boundaries_.set(++chunk_, written_);
        member_.identify();
        if (hash_)
        {
            hash_->commit(member_.id_, std::move(hash_state_));
            hash_ = nullptr;
        }
        if (packed_bits_) *next_word_ = packed_word_;
        finished_ = true;
        return std::move(member_);
    }
private:
    Member member_;
    size_t expected_, written_ = 0, chunk_ = 0;
    uint64_t previous_ = 0;
    uint64_t* next_word_ = nullptr;
    uint64_t packed_word_ = 0;
    unsigned packed_bits_ = 0;
    bool finished_ = false, failed_ = false;
    MemberBatchHash* hash_ = nullptr;
    MemberBatchHash::State hash_state_;
};

// Receives the same little-endian low-field words as CompactMemberBuilder,
// without a second full member or a host packing scan. Only the internal GPU
// producer may supply the validation summary, after its device work completes.
// Precomputed recovery-hash consumers use CompactMemberBuilder and raw tiles.
class CompactMemberReceiver
{
public:
    CompactMemberReceiver(uint8_t k, size_t count, bool trust_validation = true)
        : trust_validation_(trust_validation)
    {
        static_assert(PlotGroupFile::PROOFS_PER_CHUNK_BITS == 6,
                      "GPU compact transport requires the ordinary chunk layout");
        static_assert(std::endian::native == std::endian::little,
                      "packed transfer uses little-endian 64-bit fields");
        group_tools::require(count <= UINT32_MAX, "packed transfer requires narrow fragment indexes");
        (void)Member::compact_bytes(k, count);
        member_.k_ = k;
        member_.count_ = count;
        member_.packed_words_ = Member::packed_words(k, count);
        if (member_.packed_words_) member_.packed_.reset(new uint64_t[member_.packed_words_]);
        member_.boundaries_.reset(size_t(PlotGroupFile::getChunkCountForK(k)), count);
    }
    CompactMemberReceiver(CompactMemberReceiver const&) = delete;
    CompactMemberReceiver& operator=(CompactMemberReceiver const&) = delete;
    CompactMemberReceiver(CompactMemberReceiver&&) = delete;
    CompactMemberReceiver& operator=(CompactMemberReceiver&&) = delete;
    void append(PackedFragmentTile const& tile)
    {
        group_tools::require(!finished_ && !failed_, "packed transfer is already finished or failed");
        try
        {
            if (!started_)
            {
                size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(member_.k_));
                auto const validation = tile.validation;
                group_tools::require(tile.words.empty() && tile.boundaries.size() == chunks + 1 &&
                    validation.checked_count == member_.count_ &&
                    !(validation.flags & ~PackedMemberValidation::known_flags) &&
                    !(validation.flags & (PackedMemberValidation::out_of_range | PackedMemberValidation::unsorted)),
                    "invalid packed transfer validation or boundary prelude");
                group_tools::require(tile.boundaries.front() == 0 && tile.boundaries.back() == member_.count_,
                                     "packed transfer boundary endpoints differ");
                uint32_t previous = 0;
                for (size_t chunk = 0; chunk <= chunks; ++chunk)
                {
                    uint32_t const next = tile.boundaries[chunk];
                    group_tools::require(next >= previous && next <= member_.count_,
                                         "packed transfer boundaries are not ordered");
                    member_.boundaries_.set(chunk, next);
                    previous = next;
                }
                member_.validated_deltas_ = !(validation.flags & PackedMemberValidation::wide_delta);
                started_ = true;
                return;
            }
            group_tools::require(tile.boundaries.empty() && !tile.validation.checked_count && !tile.validation.flags &&
                tile.words.size() <= member_.packed_words_ - written_, "invalid packed transfer payload tile");
            if (!tile.words.empty())
                std::copy(tile.words.begin(), tile.words.end(), member_.packed_.get() + written_);
            written_ += tile.words.size();
        }
        catch (...) { failed_ = true; throw; }
    }
    Member finish()
    {
        group_tools::require(started_ && !finished_ && !failed_ && written_ == member_.packed_words_,
                             "incomplete packed transfer");
        unsigned const tail = unsigned((member_.count_ *
            (member_.k_ + PlotGroupFile::PROOFS_PER_CHUNK_BITS)) & 63);
        if (tail)
        {
            failed_ = (member_.packed_[written_ - 1] >> tail) != 0;
            group_tools::require(!failed_, "packed transfer tail is not zero padded");
        }
        // A network peer's validation declaration cannot substitute for local
        // inspection. Packed words encode low fields only; the received chunk
        // boundaries provide their high fields. Check each region before
        // allowing codecs to rely on the validated-delta fast path.
        if (!trust_validation_)
        {
            failed_ = true;
            auto const view = member_.view();
            uint64_t const maximum_delta = uint64_t(72 - member_.k_) * (178ull << (member_.k_ - 8));
            member_.validated_deltas_ = true;
            for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(member_.k_); ++chunk)
            {
                auto const region = view.region(chunk);
                for (size_t i = 1; i < region.size(); ++i)
                {
                    group_tools::require(region[i] >= region[i - 1], "unsorted packed transfer payload");
                    if (region[i] - region[i - 1] >= maximum_delta) member_.validated_deltas_ = false;
                }
            }
            failed_ = false;
        }
        member_.identify();
        finished_ = true;
        return std::move(member_);
    }
private:
    Member member_;
    size_t written_ = 0;
    bool started_ = false, finished_ = false, failed_ = false;
    bool trust_validation_;
};

inline void Member::compact(uint8_t k, MemberBatchHash* hash)
{
    if (compact())
    {
        group_tools::require(k == k_, "packed member k mismatch");
        return;
    }
    CompactMemberBuilder builder(k, size(), hash);
    builder.append(view().raw());
    *this = builder.finish();
}

// Preserve raw fixture callers while allowing owned/borrowed compact storage.
// This adapter owns no payload; its source collection must outlive engine work.
class Members
{
public:
    Members() = default;
    Members(std::vector<uint64_t> const* raw, size_t count) : raw_(raw, count) {}
    Members(Member const* stored, size_t count) : stored_(stored, count) {}
    Members(std::span<std::vector<uint64_t> const> raw) : raw_(raw) {}
    Members(std::span<std::vector<uint64_t>> raw) : raw_(raw) {}
    Members(std::vector<std::vector<uint64_t>> const& raw) : raw_(raw) {}
    template <size_t N> Members(std::array<std::vector<uint64_t>, N> const& raw) : raw_(raw) {}
    Members(std::span<Member const> stored) : stored_(stored) {}
    Members(std::span<Member> stored) : stored_(stored) {}
    Members(std::vector<Member> const& stored) : stored_(stored) {}
    template <size_t N> Members(std::array<Member, N> const& stored) : stored_(stored) {}
    size_t size() const { return raw_.empty() ? stored_.size() : raw_.size(); }
    bool empty() const { return size() == 0; }
    MemberView operator[](size_t index) const
    { return raw_.empty() ? stored_[index].view() : MemberView(raw_[index]); }
    Members subspan(size_t first, size_t count = std::dynamic_extent) const
    { return raw_.empty() ? Members(stored_.subspan(first, count)) : Members(raw_.subspan(first, count)); }
private:
    std::span<std::vector<uint64_t> const> raw_;
    std::span<Member const> stored_;
};

inline std::optional<std::array<uint8_t, 32>> MemberBatchHash::digest_for(Members members) const
{
    if (members.empty() || members.size() > size_) return std::nullopt;
    for (size_t i = 0; i < members.size(); ++i)
    {
        auto const member = members[i];
        if (!member.compact() || !member.immutable_id() || member.immutable_id() != prefixes_[i].id)
            return std::nullopt;
    }
    return prefixes_[members.size() - 1].digest;
}
} // namespace group_grow
