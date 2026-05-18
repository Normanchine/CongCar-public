#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <atomic>
#include "IScene.h"
#include "SceneFactory.h"

class SceneManager {
public:
    // 切换场景：被语义分割/目标检测线程调用
    // shared_ptr + atomic_store 保证 run() 中正在执行的旧场景不会被析构
    bool switchTo(const std::string& sceneName) {
        auto newScene = SceneFactory::instance().create(sceneName);
        if (!newScene) {
            std::cout << "\n  切换到：" << sceneName << " 失败" << "\n";
            return false;
        }

        std::atomic_store(&_currentScene, std::move(newScene));
        std::cout << "\n  实时切换到：" << sceneName << "\n";
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

private:
    std::shared_ptr<IScene> _currentScene;
};
