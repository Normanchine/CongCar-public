#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

// 行人避让场景
//   DRIVING -> WAITING：人进入车未来通行走廊
//   WAITING -> RELEASE_HOLD：预测人将离开走廊，提前发车并短暂防抖
//   CAR_PERSON_WAIT：右侧车近距离消失后先停车，等人往右清出左偏走廊后发车
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 重写onSpeed：停车后的加速恢复逻辑
    class Follower : public PathFollower {
    public:
        float center_offset_ratio = 0.0f; // 右正左负，发车后短暂偏移避人

    protected:
        bool onPurePursuit(const cv::Mat& mask,
                           const std::vector<cv::Point2f>& centers,
                           float& error,
                           cv::Point2f& aim_pt,
                           SharedContext* ctx) override;
        float onSpeed(float error, SharedContext* ctx) override;
    };

    Follower follower_;

    int person_class_id_ = -1;
    int car_class_id_ = -1;
    bool waiting_ = false;
    bool car_person_waiting_ = false;
    int lost_frames_ = 0;
    int car_watch_frames_ = 0;
    int release_ready_frames_ = 0;
    int release_hold_frames_ = 0;
    bool tracker_valid_ = false;
    cv::Point2f last_foot_bird_{0.0f, 0.0f};
    float foot_dx_px_ = 0.0f;

    void stopCar(SharedContext* ctx);
    void followLine(SharedContext* ctx);
    void resetWaitState();
    void releaseAndFollow(float avoid_offset_ratio, SharedContext* ctx);

    // 查找最近行人脚底点（原图坐标），返回是否检测到
    bool findPersonFoot(cv::Point2f& foot_img, SharedContext* ctx);

    // 更新鸟瞰脚底横向速度（像素/帧）
    void updatePersonTracker(const cv::Point2f& foot_bird);

    // 判断人是否在纵向危险距离和车未来通行走廊内（鸟瞰）
    bool isPersonInDanger(const cv::Point2f& foot_bird, SharedContext* ctx) const;

    bool shouldRelease(const cv::Point2f& foot_bird,
                       SharedContext* ctx,
                       float& avoid_offset_ratio);

    bool shouldReleaseCarPerson(const cv::Point2f& foot_bird,
                                SharedContext* ctx,
                                float& avoid_offset_ratio);

    bool isRightSideCarPresent(SharedContext* ctx);
    bool findRightSideCar(float& dist_m, SharedContext* ctx);
};
