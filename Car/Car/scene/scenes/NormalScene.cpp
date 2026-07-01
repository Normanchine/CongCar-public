#include "NormalScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"

REGISTER_SCENE(NormalScene)

void NormalScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    follower_.process(seg->mask, ctx);
}
