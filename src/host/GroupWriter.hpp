// One ordered file consumer per physical-drive group, with at most one waiting
// task in addition to the running task. Borrowed arguments outlive the worker.
#pragma once
#include "host/GroupEngine.hpp"

namespace group_grow
{
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
