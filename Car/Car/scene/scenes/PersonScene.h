#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

// 检测到person时触发的场景：近距离停车 + 人进入中心带时恢复巡线
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 巡线跟随（可设外部速度）
    PathFollower follower_;

    struct PersonTarget {
        bool valid = false;
        float bird_x = 0.0f;
        float bird_y = 0.0f;
        float distance_m = 0.0f;
    };

    void updatePersonClassId(SharedContext* ctx);
    PersonTarget findClosestPerson(SharedContext* ctx) const;
    void stopCar(SharedContext* ctx);
    void followLine(SharedContext* ctx);
    void startWaiting(const PersonTarget& person, SharedContext* ctx);
    void releaseAndFollow(SharedContext* ctx);
    void resetWaitState();

    int person_class_id_ = -1;
    bool waiting_ = false;
    int lost_frames_ = 0;
    int release_hold_frames_ = 0;
    float last_bird_x_ = -1.0f;
};
