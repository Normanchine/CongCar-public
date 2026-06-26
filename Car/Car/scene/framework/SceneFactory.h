#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include "IScene.h"

class SceneFactory {
public:
    static SceneFactory& instance() {
        static SceneFactory m_instance;
        return m_instance;
    }

    SceneFactory(const SceneFactory&) = delete;
    SceneFactory& operator=(const SceneFactory&) = delete;

    using Creator = std::function<std::shared_ptr<IScene>()>;

    void reg(const std::string& name, Creator c) {
        std::lock_guard<std::mutex> lock(_mutex);
        _creators[name] = std::move(c);
    }

    std::shared_ptr<IScene> create(const std::string& name) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _creators.find(name);
        if (it != _creators.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    SceneFactory() = default;
    std::unordered_map<std::string, Creator> _creators;
    mutable std::mutex _mutex;
};

#define REGISTER_SCENE(CLASS) \
    static bool _scene_##CLASS = [] { \
        SceneFactory::instance().reg(#CLASS, [] { return std::make_shared<CLASS>(); }); \
        return true; \
    }();
