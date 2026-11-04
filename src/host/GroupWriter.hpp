// One ordered file consumer per physical-drive group, with at most one waiting
// task in addition to the running task. Borrowed arguments outlive the worker.
#pragma once
#include "host/GroupEngine.hpp"
#include <type_traits>

namespace group_grow
{
// Snapshot before inspecting work, then wait for a generation change. Keeping
// the predicate under the notifier's mutex closes the inspection-to-wait gap.
class ControllerWake
{
public:
    uint64_t generation() const { std::lock_guard lock(mutex_); return generation_; }
    void notify()
    {
        { std::lock_guard lock(mutex_); ++generation_; }
        ready_.notify_one();
    }
    template <typename Rep, typename Period>
    bool wait_for(uint64_t observed, std::chrono::duration<Rep, Period> timeout)
    {
        std::unique_lock lock(mutex_);
        return ready_.wait_for(lock, timeout, [&] { return generation_ != observed; });
    }
private:
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    uint64_t generation_ = 0;
};

// std::async publishes its result after the callback returns. Publish through
// a separate promise first so the notification always follows value/exception
// readiness. get/wait join the runner before captured borrowers can be freed;
// callers using submit adapters must drain them before destroying their state.
template <typename T>
class NotifiedFuture
{
public:
    NotifiedFuture() = default;
    NotifiedFuture(NotifiedFuture&&) = default;
    NotifiedFuture& operator=(NotifiedFuture&&) = default;
    template <typename Operation, typename Submit>
    static NotifiedFuture submit(ControllerWake& wake, Operation&& operation, Submit&& submit)
    {
        auto promise = std::make_shared<std::promise<T>>();
        auto result = promise->get_future();
        auto runner = submit([promise = std::move(promise), operation = std::forward<Operation>(operation), &wake]() mutable
        {
            try
            {
                if constexpr (std::is_void_v<T>) { operation(); promise->set_value(); }
                else promise->set_value(operation());
            }
            catch (...)
            {
                promise->set_exception(std::current_exception()); wake.notify();
                throw; // Submit adapters retain the same operation failure.
            }
            wake.notify();
        });
        return {std::move(result), std::move(runner)};
    }
    template <typename Operation>
    static NotifiedFuture launch(ControllerWake& wake, Operation&& operation)
    {
        return submit(wake, std::forward<Operation>(operation), [](auto task)
        { return std::async(std::launch::async, std::move(task)); });
    }
    bool valid() const { return result_.valid(); }
    void wait() const { runner_.wait(); }
    template <typename Rep, typename Period>
    std::future_status wait_for(std::chrono::duration<Rep, Period> timeout) const
    {
        if (runner_.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            return std::future_status::ready;
        return result_.wait_for(timeout);
    }
    T get()
    {
        auto result = std::move(result_); auto runner = std::move(runner_);
        runner.get(); return result.get();
    }
private:
    NotifiedFuture(std::future<T> result, std::future<void> runner)
        : result_(std::move(result)), runner_(std::move(runner)) {}
    std::future<T> result_;
    std::future<void> runner_;
};

class OrderedWriter
{
public:
    explicit OrderedWriter(std::function<void()> notify) : notify_(std::move(notify)), thread_([this] { run(); }) {}
    ~OrderedWriter()
    {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        ready_.notify_one(); thread_.join();
    }
    OrderedWriter(OrderedWriter const&) = delete;
    OrderedWriter& operator=(OrderedWriter const&) = delete;
    bool has_room() const { std::lock_guard lock(mutex_); return !task_.valid(); }
    bool busy() const { std::lock_guard lock(mutex_); return active_ || task_.valid(); }
    std::future<void> submit(std::function<void()> operation)
    {
        std::packaged_task<void()> task(std::move(operation)); auto result = task.get_future();
        { std::lock_guard lock(mutex_); require(!stopping_ && !task_.valid(), "destination writer queue is full"); task_ = std::move(task); }
        ready_.notify_one(); return result;
    }
private:
    void run()
    {
        for (;;)
        {
            std::packaged_task<void()> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || task_.valid(); });
                if (stopping_ && !task_.valid()) return;
                task = std::move(task_); active_ = true;
            }
            notify_(); task();
            { std::lock_guard lock(mutex_); active_ = false; }
            notify_();
        }
    }
    std::function<void()> notify_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::packaged_task<void()> task_;
    bool stopping_ = false, active_ = false;
    std::thread thread_;
};
} // namespace group_grow
