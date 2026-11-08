// Compatibility is an explicit contract; the source hash is provenance.
#include "host/ComputeWorker.hpp"
#include <bit>
#include <climits>

// Change the affected identity when its wire semantics, canonical output or
// interpretation changes. Generation includes coencoded contributions, so it
// also requires the codec contract. Resource profiles and supported task kinds
// are negotiated separately by the broker, not encoded in these identities.
#ifndef GROW_COMPUTE_TASK_IDENTITY
#define GROW_COMPUTE_TASK_IDENTITY "compute-task-1"
#endif
#ifndef GROW_COMPUTE_GENERATION_IDENTITY
#define GROW_COMPUTE_GENERATION_IDENTITY "pos2-1.0-generation-1"
#endif
#ifndef GROW_COMPUTE_CODEC_IDENTITY
#define GROW_COMPUTE_CODEC_IDENTITY "ordinary-group-chunk-codec-1"
#endif
#ifndef GROW_COMPUTE_REPRESENTATION_IDENTITY
#define GROW_COMPUTE_REPRESENTATION_IDENTITY "packed-fields-le-1"
#endif

namespace group_grow
{
std::string worker_compatibility(pool::Kind kind)
{
    // The network uses explicit-width integers. The canonical group codec's
    // bit fields require little-endian octets, as enforced by GroupCodec.
    static_assert(CHAR_BIT == 8 && std::endian::native == std::endian::little);
    static_assert(sizeof(uint64_t) == 8 && sizeof(uint32_t) == 4 && sizeof(uint16_t) == 2);
    std::string result = GROW_COMPUTE_TASK_IDENTITY ":" GROW_COMPUTE_REPRESENTATION_IDENTITY ":" GROW_COMPUTE_CODEC_IDENTITY;
    if (kind != pool::Kind::Pack) result += ":" GROW_COMPUTE_GENERATION_IDENTITY;
    return result;
}
std::string worker_producer_build()
{
#ifdef GROW_COMPUTE_PRODUCER_BUILD
    return GROW_COMPUTE_PRODUCER_BUILD;
#else
    return {};
#endif
}
} // namespace group_grow
