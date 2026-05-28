#!/bin/bash
# ========================================
# run_seg.sh - 运行语义分割单图测试
# 用法: ./run_seg.sh <图片路径> [模型路径]
# 示例: ./run_seg.sh test.jpg
#       ./run_seg.sh test.jpg model/pplite_seg.rknn
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/seg/build"
EXECUTABLE="$BUILD_DIR/seg_test"

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <图片路径> [模型路径]"
    echo "示例: $0 test.jpg"
    echo "      $0 test.jpg model/pplite_seg.rknn"
    exit 1
fi

# 检查是否已构建
if [ ! -f "$EXECUTABLE" ]; then
    echo "[错误] 未找到可执行文件: $EXECUTABLE"
    echo "[提示] 请先构建: cd seg && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

IMAGE_PATH="$1"
MODEL_PATH="${2:-}"

echo "========================================"
echo "  seg_test 语义分割测试"
echo "========================================"
echo "可执行文件: $EXECUTABLE"
echo "图片: $IMAGE_PATH"
if [ -n "$MODEL_PATH" ]; then
    echo "模型: $MODEL_PATH"
fi
echo "----------------------------------------"

# 运行
cd "$BUILD_DIR"
if [ -n "$MODEL_PATH" ]; then
    "$EXECUTABLE" "$IMAGE_PATH" "$MODEL_PATH"
else
    "$EXECUTABLE" "$IMAGE_PATH"
fi
