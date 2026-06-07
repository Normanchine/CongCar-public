#pragma once

#include "../thread/IThreadTask.h"
#include "../common/HttpStreamServer.h"
#include <memory>

class DisplayTask : public IThreadTask {
public:
    explicit DisplayTask(SharedContext* ctx);
    ~DisplayTask() override;

    void run() override;

private:
    std::unique_ptr<HttpStreamServer> http_server_;
};
