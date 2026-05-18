#include "thread/Car.h"

int main(int argc, char* argv[]) {
    Car car;

    if (!car.init(argc, argv)) {
        return -1;
    }

    car.start();
    car.wait();

    return 0;
}
