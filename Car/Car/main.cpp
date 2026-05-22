#include "thread/Car.h"
#include <csignal>

namespace {
Car* g_car = nullptr;

void onSignal(int) {
    if (g_car) {
        g_car->signalStop();   // 仅置标志位，信号安全
    }
}
} // namespace

int main(int argc, char* argv[]) {
    Car car;
    g_car = &car;

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    if (!car.init(argc, argv)) {
        return -1;
    }

    car.start();
    car.wait();

    return 0;
}
