#!/bin/bash
# ========================================
# run_calibrate.sh - 运行Car透视变换标定工具(C++版)
# 用法:
#      从共享内存读取实时帧：./run_calibrate.sh
#                               ./run_calibrate.sh Car/config/default.json
#      从图片文件读取：       ./run_calibrate.sh <图片路径> [配置文件路径]
#                               ./run_calibrate.sh track.jpg
#                               ./run_calibrate.sh track.jpg Car/config/default.json
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/Car/build"
EXECUTABLE="$BUILD_DIR/car_calib"
DEFAULT_CONFIG="$SCRIPT_DIR/Car/config/default.json"

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "[错误] 未找到可执行文件: $EXECUTABLE"
    echo "[提示] 请先运行构建: ./build_all.sh"
    exit 1
fi

echo "========================================"
echo "  car_calib 透视变换标定工具"
echo "========================================"
echo "可执行文件: $EXECUTABLE"
echo "----------------------------------------"

# 根据参数数量决定运行模式
if [ -z "$1" ]; then
    # 无参数：从共享内存读取实时帧
    echo "[模式] 从共享内存读取实时帧"
    echo "[提示] 请确保 Python 视频推流服务已启动"
    echo ""
    "$EXECUTABLE"
elif [ "$#" -eq 1 ]; then
    # 一个参数：判断是图片还是配置文件
    if [[ "$1" == *.json ]] || [[ "$1" == *.JSON ]]; then
        # 是配置文件，从共享内存读取
        echo "[模式] 从共享内存读取实时帧"
        echo "配置输出: $1"
        echo "[提示] 请确保 Python 视频推流服务已启动"
        echo ""
        "$EXECUTABLE" "$1"
    else
        # 是图片路径
        echo "[模式] 从图片文件读取"
        echo "图片: $1"
        echo "配置输出: $DEFAULT_CONFIG"
        echo ""
        "$EXECUTABLE" "$1" "$DEFAULT_CONFIG"
    fi
else
    # 两个参数：图片路径 + 配置文件路径
    echo "[模式] 从图片文件读取"
    echo "图片: $1"
    echo "配置输出: $2"
    echo ""
    "$EXECUTABLE" "$1" "$2"
fi
