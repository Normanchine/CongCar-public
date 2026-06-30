#include "CarScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"

REGISTER_SCENE(CarScene)

void CarScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    follower_.process(seg->mask, ctx);
}
