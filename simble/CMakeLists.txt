cmake_minimum_required(VERSION 3.16)
project(simble LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# ---- 公共头文件路径 ----
# RKNN SDK 头文件
include_directories(${CMAKE_SOURCE_DIR}/../Car/Car/hal/rknn)
# 透视变换工具（PerspectiveMapper.h）
include_directories(${CMAKE_SOURCE_DIR}/../Car/Car/common)
# PathPlanning.h 在本地，无需引用 Car/Car/scene

# ===================================================
# 目标1: simble_test — 共享内存 → 分割 → 巡线 → 实时显示
# ===================================================
add_executable(simble_test main.cpp)

target_link_libraries(simble_test
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    rknnrt
    rt          # POSIX shm_open
)

# ===================================================
# 目标2: simble_calib — 透视变换标定工具
# ===================================================
add_executable(simble_calib calibrate.cpp)

target_link_libraries(simble_calib
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    rt          # POSIX shm_open
)