#!/bin/bash
# ========================================
# run_calibrate.sh - 运行Car透视变换标定工具(Python版)
# 用法: ./run_calibrate.sh <图片路径> [配置文件路径]
# 示例: ./run_calibrate.sh track.jpg
#       ./run_calibrate.sh track.jpg Car/config/default.json
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CALIB_SCRIPT="$SCRIPT_DIR/Car/tools/calibrate_perspective.py"
DEFAULT_CONFIG="$SCRIPT_DIR/Car/config/default.json"

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <图片路径> [配置文件路径]"
    echo "示例: $0 track.jpg"
    echo "      $0 track.jpg Car/config/default.json"
    echo ""
    echo "流程:"
    echo "  1. 在图片上依次点击: 左上 → 右上 → 右下 → 左下"
    echo "  2. 输入鸟瞰图尺寸（默认640x480）"
    echo "  3. 输入所选区域物理尺寸（米）"
    echo "  4. 预览鸟瞰效果，确认后写入配置文件"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "$CALIB_SCRIPT" ]; then
    echo "[错误] 未找到标定脚本: $CALIB_SCRIPT"
    exit 1
fi

IMAGE_PATH="$1"
CONFIG_PATH="${2:-$DEFAULT_CONFIG}"

echo "========================================"
echo "  calibrate_perspective.py 透视变换标定"
echo "========================================"
echo "脚本: $CALIB_SCRIPT"
echo "图片: $IMAGE_PATH"
echo "输出配置: $CONFIG_PATH"
echo "----------------------------------------"

# 运行
python3 "$CALIB_SCRIPT" "$IMAGE_PATH" "$CONFIG_PATH"
