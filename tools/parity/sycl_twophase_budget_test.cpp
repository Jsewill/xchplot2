#undef NDEBUG
#include "gpu/TwoPhaseScratch.hpp"
#include <cassert>
#include <limits>

int main()
{
    using namespace pos2gpu::sycl_backend;
    sycl::queue q;
    set_twophase_budget(q, 812);
    auto* scratch = acquire_twophase_scratch(q, 100);
    assert(scratch && twophase_bytes_held(q) == 812);
    assert(acquire_twophase_scratch(q, 100) == scratch);
    assert(!acquire_twophase_scratch(q, 101));
    set_twophase_budget(q, 811);
    assert(twophase_bytes_held(q) == 0);
    assert(!acquire_twophase_scratch(q, 100));
    set_twophase_budget(q, 0);
    assert(!acquire_twophase_scratch(q, 1));
    set_twophase_budget(q, std::numeric_limits<uint64_t>::max());
    assert(!acquire_twophase_scratch(q, std::numeric_limits<uint64_t>::max()));
    assert(acquire_twophase_scratch(q, 100));
    release_twophase_scratch(q);
    assert(twophase_bytes_held(q) == 0);
    assert(twophase_map().find(&q) == twophase_map().end());
}
