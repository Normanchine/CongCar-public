#!/bin/bash
# ========================================
# run_simble_calib.sh - 运行透视变换标定工具
# 
# 支持两种模式：
#   1. 从共享内存读取实时帧（推荐）
#      用法: ./run_simble_calib.sh
#            ./run_simble_calib.sh simble/config.json
#
#   2. 从图片文件读取
#      用法: ./run_simble_calib.sh <图片路径> [配置文件路径]
#            ./run_simble_calib.sh track.jpg
#            ./run_simble_calib.sh track.jpg simble/config.json
#
# 前提条件（共享内存模式）：
#   Python 视频推流服务已启动
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/simble/build"
EXECUTABLE="$BUILD_DIR/simble_calib"

# 检查是否已构建
if [ ! -f "$EXECUTABLE" ]; then
    echo "[错误] 未找到可执行文件: $EXECUTABLE"
    echo "[提示] 请先构建: ./build_all.sh"
    exit 1
fi

echo "========================================"
echo "  simble_calib 透视变换标定工具"
echo "========================================"

# 将相对路径转为绝对路径的函数
to_abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$SCRIPT_DIR/$path"
    fi
}

# 根据参数判断模式
if [ -z "$1" ]; then
    # 无参数：从共享内存读取
    echo "[模式] 从共享内存读取实时帧"
    echo "提示: 请确保 Python 视频推流服务已启动"
    echo "----------------------------------------"
    cd "$BUILD_DIR"
    "$EXECUTABLE"

elif [ -n "$1" ] && [ -z "$2" ]; then
    # 一个参数
    if [[ "$1" == *.json ]] || [[ "$1" == *.JSON ]]; then
        # 是 json 文件：从共享内存读取，输出到该配置
        ABS_CONFIG=$(to_abs_path "$1")
        echo "[模式] 从共享内存读取实时帧"
        echo "配置输出: $ABS_CONFIG"
        echo "提示: 请确保 Python 视频推流服务已启动"
        echo "----------------------------------------"
        cd "$BUILD_DIR"
        "$EXECUTABLE" "$ABS_CONFIG"
    else
        # 是图片文件
        ABS_IMAGE=$(to_abs_path "$1")
        echo "[模式] 从图片文件读取"
        echo "图片: $ABS_IMAGE"
        echo "----------------------------------------"
        cd "$BUILD_DIR"
        "$EXECUTABLE" "$ABS_IMAGE"
    fi

else
    # 两个参数：图片 + 配置
    ABS_IMAGE=$(to_abs_path "$1")
    ABS_CONFIG=$(to_abs_path "$2")
    echo "[模式] 从图片文件读取"
    echo "图片: $ABS_IMAGE"
    echo "配置输出: $ABS_CONFIG"
    echo "----------------------------------------"
    cd "$BUILD_DIR"
    "$EXECUTABLE" "$ABS_IMAGE" "$ABS_CONFIG"
fi
