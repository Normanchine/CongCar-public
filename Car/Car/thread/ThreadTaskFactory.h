#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "IThreadTask.h"

class ThreadTaskFactory {
public:
    static ThreadTaskFactory& instance() {
        static ThreadTaskFactory m_instance;
        return m_instance;
    }

    ThreadTaskFactory(const ThreadTaskFactory&) = delete;
    ThreadTaskFactory& operator=(const ThreadTaskFactory&) = delete;

    using Creator = std::function<std::unique_ptr<IThreadTask>(SharedContext*)>;

    void reg(const std::string& name, Creator c) {
        std::lock_guard<std::mutex> lock(_mutex);
        _creators[name] = std::move(c);
    }

    std::unique_ptr<IThreadTask> create(const std::string& name, SharedContext* ctx) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _creators.find(name);
        if (it != _creators.end()) {
            return it->second(ctx);
        }
        return nullptr;
    }

    std::vector<std::string> all_names() const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<std::string> names;
        for (const auto& [name, _] : _creators)
            names.push_back(name);
        return names;
    }

private:
    ThreadTaskFactory() = default;
    std::unordered_map<std::string, Creator> _creators;
    mutable std::mutex _mutex;
};

#define REGISTER_THREAD_TASK(CLASS) \
    static bool _thread_##CLASS = [] { \
        ThreadTaskFactory::instance().reg(#CLASS, \
            [](SharedContext* ctx) { return std::make_unique<CLASS>(ctx); }); \
        return true; \
    }();
