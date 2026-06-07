cmake_minimum_required(VERSION 3.16)
project(Car LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(car
    main.cpp
    thread/Car.cpp
    scene/NormalScene.cpp
    hal/memory/shared_memory_manager.cpp
    hal/uart/UartIO.cpp
    common/PerfStats.cpp
    inference/RKNNEngine.cpp
    tasks/CaptureTask.cpp
    tasks/SegmentationTask.cpp
    tasks/DisplayTask.cpp
    tasks/DetectionTask.cpp
    tasks/ControlTask.cpp
)

target_link_libraries(car
    ${OpenCV_LIBS}
    Threads::Threads
    rknnrt
    rt
)

# ============================================================================
# 标定工具 (car_calib)
# ============================================================================
add_executable(car_calib
    ../tools/calibrate.cpp
)

target_link_libraries(car_calib
    ${OpenCV_LIBS}
)
