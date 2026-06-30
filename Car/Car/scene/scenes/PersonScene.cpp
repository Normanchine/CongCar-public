#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"

REGISTER_SCENE(PersonScene)

void PersonScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    follower_.process(seg->mask, ctx);
}
