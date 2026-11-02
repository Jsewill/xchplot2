// Worker-only overlay: inherit a scoped CPU allowance through reference threads.
#ifndef GROW_WORKER_THREADS_HPP
#define GROW_WORKER_THREADS_HPP
#include <algorithm>
#include <thread>

namespace group_grow::worker_detail
{
inline thread_local unsigned cpu_limit = 0;
inline unsigned cpu_threads()
{
    if (cpu_limit) return cpu_limit;
    return std::thread::hardware_concurrency();
}
class CpuLimit
{
public:
    explicit CpuLimit(unsigned value) : previous_(cpu_limit)
    { cpu_limit = previous_ ? std::min(previous_, value) : value; }
    ~CpuLimit() { cpu_limit = previous_; }
    CpuLimit(CpuLimit const&) = delete;
private:
    unsigned previous_;
};
} // namespace group_grow::worker_detail
#endif
