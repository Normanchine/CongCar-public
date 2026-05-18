#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

class ThreadPool {
private:
    struct PrioritizedTask {
        int priority;
        std::function<void()> task;
        uint64_t sequence_id;

        bool operator<(const PrioritizedTask& other) const {
            if (priority != other.priority) {
                return priority > other.priority;
            }
            return sequence_id > other.sequence_id;
        }
    };

    std::vector<std::thread> workers;
    std::mutex queue_mtx;
    std::atomic<bool> stop{false};
    std::condition_variable condition;
    std::priority_queue<PrioritizedTask> tasks;
    std::atomic<uint64_t> next_sequence_id{0};

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

public:
    explicit ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mtx);
                        condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.top().task);
                        this->tasks.pop();
                        if (this->tasks.empty()) {
                            condition.notify_all();
                        }
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueueWithPriority(int priority, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mtx);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            uint64_t seq = next_sequence_id.fetch_add(1, std::memory_order_relaxed);
            tasks.emplace(PrioritizedTask{
                priority,
                [task]() { (*task)(); },
                seq
            });
        }
        condition.notify_one();
        return res;
    }

    void waitAllTasks() {
        std::unique_lock<std::mutex> lock(queue_mtx);
        condition.wait(lock, [this]() {
            return tasks.empty();
        });
    }

    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable())
                worker.join();
        }
    }
};
