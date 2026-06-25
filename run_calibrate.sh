#!/bin/bash
# ========================================
# run_calibrate.sh - 运行Car透视变换标定工具(C++版)
# 用法:
#      从共享内存读取实时帧：./run_calibrate.sh
#                               ./run_calibrate.sh Car/config/default.json
#      从图片文件读取：       ./run_calibrate.sh <图片路径> [配置文件路径]
#                               ./run_calibrate.sh track.jpg
#                               ./run_calibrate.sh track.jpg Car/config/default.json
#      裁剪标定（frame_crop模式）：--crop x,y,w,h 放在参数最前面
#                               ./run_calibrate.sh --crop 0,240,640,240 track.jpg
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

# 分离 --crop 参数和位置参数
CROP_FLAG=""
POS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --crop)
            CROP_FLAG="--crop $2"
            shift 2
            ;;
        *)
            POS_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "========================================"
echo "  car_calib 透视变换标定工具"
echo "========================================"
echo "可执行文件: $EXECUTABLE"
[ -n "$CROP_FLAG" ] && echo "裁剪: $CROP_FLAG"
echo "----------------------------------------"

# 根据位置参数数量决定运行模式
if [ ${#POS_ARGS[@]} -eq 0 ]; then
    # 无参数：从共享内存读取实时帧
    echo "[模式] 从共享内存读取实时帧"
    echo "配置输出: $DEFAULT_CONFIG"
    echo "[提示] 请确保 Python 视频推流服务已启动"
    echo ""
    $EXECUTABLE $CROP_FLAG "$DEFAULT_CONFIG"
elif [ ${#POS_ARGS[@]} -eq 1 ]; then
    # 一个参数：判断是图片还是配置文件
    if [[ "${POS_ARGS[0]}" == *.json ]] || [[ "${POS_ARGS[0]}" == *.JSON ]]; then
        echo "[模式] 从共享内存读取实时帧"
        echo "配置输出: ${POS_ARGS[0]}"
        echo "[提示] 请确保 Python 视频推流服务已启动"
        echo ""
        $EXECUTABLE $CROP_FLAG "${POS_ARGS[0]}"
    else
        echo "[模式] 从图片文件读取"
        echo "图片: ${POS_ARGS[0]}"
        echo "配置输出: $DEFAULT_CONFIG"
        echo ""
        $EXECUTABLE $CROP_FLAG "${POS_ARGS[0]}" "$DEFAULT_CONFIG"
    fi
else
    # 两个参数：图片路径 + 配置文件路径
    echo "[模式] 从图片文件读取"
    echo "图片: ${POS_ARGS[0]}"
    echo "配置输出: ${POS_ARGS[1]}"
    echo ""
    $EXECUTABLE $CROP_FLAG "${POS_ARGS[0]}" "${POS_ARGS[1]}"
fi
