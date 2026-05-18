cmake_minimum_required(VERSION 3.16)
project(seg_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/../Car/Car/hal/rknn)

add_executable(seg_test main.cpp)

target_link_libraries(seg_test
    ${OpenCV_LIBS}
    rknnrt
)
