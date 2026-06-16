#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <atomic>
#include "IScene.h"
#include "SceneFactory.h"

class SceneManager {
public:
    SceneManager() {
        // 启动时自动创建默认场景，确保 run() 有场景可执行
        auto scene = SceneFactory::instance().create(_currentName);
        if (scene) {
            std::atomic_store(&_currentScene, std::move(scene));
            std::cout << "[SceneMgr] 初始场景: " << _currentName << std::endl;
        } else {
            std::cerr << "[SceneMgr] 无法创建初始场景: " << _currentName << std::endl;
        }
    }

    // 切换场景：被语义分割/目标检测线程调用
    // shared_ptr + atomic_store 保证 run() 中正在执行的旧场景不会被析构
    bool switchTo(const std::string& sceneName) {
        if (sceneName == _currentName) return true;  // 相同场景不切换

        auto newScene = SceneFactory::instance().create(sceneName);
        if (!newScene) {
            std::cout << "[SceneMgr] 切换到 " << sceneName << " 失败" << std::endl;
            return false;
        }

        std::atomic_store(&_currentScene, std::move(newScene));
        std::cout << "[SceneMgr] 场景切换: " << _currentName
                  << " → " << sceneName << std::endl;
        _currentName = sceneName;
        return true;
    }

    // 执行当前场景：被控制线程(CT)每帧调用
    // atomic_load 持有一份引用计数，switchTo 替换时旧场景不会析构
    void run(SharedContext* ctx) {
        auto scene = std::atomic_load(&_currentScene);
        if (!scene) return;
        scene->run(ctx);
    }

    // 获取当前场景指针（只读，调用方不可持有超过一个 run() 周期）
    std::shared_ptr<IScene> current() const {
        return std::atomic_load(&_currentScene);
    }

    const std::string& currentName() const { return _currentName; }

private:
    std::shared_ptr<IScene> _currentScene;
    std::string             _currentName = "NormalScene";  // 默认启动场景
};
