#!/bin/bash
# ========================================
# run_car.sh - 运行Car主程序
# 用法: ./run_car.sh [配置文件路径]
# 示例: ./run_car.sh
#       ./run_car.sh Car/config/default.json
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAR_DIR="$SCRIPT_DIR/Car"
BUILD_DIR="$CAR_DIR/build"
EXECUTABLE="$BUILD_DIR/car"
DEFAULT_CONFIG="$CAR_DIR/config/default.json"

# 检查是否已构建
if [ ! -f "$EXECUTABLE" ]; then
    echo "[错误] 未找到可执行文件: $EXECUTABLE"
    echo "[提示] 请先构建: cd Car && mkdir -p build && cd build && cmake ../Car && make -j8"
    exit 1
fi

# 确定配置文件
if [ -n "$1" ]; then
    CONFIG="$1"
else
    CONFIG="$DEFAULT_CONFIG"
fi

echo "========================================"
echo "  Car 智能车上位机"
echo "========================================"
echo "可执行文件: $EXECUTABLE"
echo "配置文件: $CONFIG"
echo "----------------------------------------"

# 运行
cd "$BUILD_DIR"
"$EXECUTABLE" "$CONFIG"
