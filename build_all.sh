#!/bin/bash
# ========================================
# build_all.sh - 构建所有程序
# 用法: ./build_all.sh [Release|Debug]
# 示例: ./build_all.sh
#       ./build_all.sh Debug
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-Release}"

echo "========================================"
echo "  构建所有程序"
echo "========================================"
echo "构建类型: $BUILD_TYPE"
echo "----------------------------------------"

set -e  # 遇错即停

# ---- 1. 构建 Car 主程序 ----
echo ""
echo "[1/3] 构建 Car 主程序..."
cd "$SCRIPT_DIR/Car"
mkdir -p build
cd build
cmake ../Car -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make -j$(nproc)
echo "[OK] Car 构建完成: $SCRIPT_DIR/Car/build/car"

# ---- 2. 构建 seg_test ----
echo ""
echo "[2/3] 构建 seg_test..."
cd "$SCRIPT_DIR/seg"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make -j$(nproc)
echo "[OK] seg_test 构建完成: $SCRIPT_DIR/seg/build/seg_test"

# ---- 3. 构建 simble ----
echo ""
echo "[3/3] 构建 simble..."
cd "$SCRIPT_DIR/simble"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make -j$(nproc)
echo "[OK] simble 构建完成:"
echo "     - $SCRIPT_DIR/simble/build/simble_test"
echo "     - $SCRIPT_DIR/simble/build/simble_calib"

# ---- 完成 ----
echo ""
echo "========================================"
echo "  全部构建完成!"
echo "========================================"
echo "可执行文件:"
echo "  Car/build/car              - 智能车主程序"
echo "  seg/build/seg_test         - 语义分割测试"
echo "  simble/build/simble_test   - 巡线一体化测试"
echo "  simble/build/simble_calib  - 透视变换标定"
echo ""
echo "运行脚本:"
echo "  ./run_car.sh [配置文件]           - 运行Car主程序"
echo "  ./run_calibrate.sh <图片> [配置]  - Car透视变换标定(Python版)"
echo "  ./run_seg.sh <图片> [模型]        - 语义分割测试"
echo "  ./run_simble_test.sh [配置文件]   - 巡线一体化测试"
echo "  ./run_simble_calib.sh <图片> [配置] - 透视变换标定(C++版)"
