#!/bin/bash
# ========================================
# run_simble_test.sh - 运行巡线一体化测试
# 用法: ./run_simble_test.sh [配置文件路径]
# 示例: ./run_simble_test.sh
#       ./run_simble_test.sh simble/config.json
# ========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/simble/build"
EXECUTABLE="$BUILD_DIR/simble_test"
DEFAULT_CONFIG="$SCRIPT_DIR/simble/config.json"

# 检查是否已构建
if [ ! -f "$EXECUTABLE" ]; then
    echo "[错误] 未找到可执行文件: $EXECUTABLE"
    echo "[提示] 请先构建: cd simble && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# 确定配置文件
if [ -n "$1" ]; then
    CONFIG="$1"
else
    # 检查默认配置是否存在
    if [ -f "$DEFAULT_CONFIG" ]; then
        CONFIG="$DEFAULT_CONFIG"
    else
        CONFIG="config.json"
    fi
fi

echo "========================================"
echo "  simble_test 巡线一体化测试"
echo "========================================"
echo "可执行文件: $EXECUTABLE"
echo "配置文件: $CONFIG"
echo ""
echo "功能: 共享内存→语义分割→透视变换→中线提取→纯跟踪→实时显示"
echo "退出: 按 ESC 键"
echo "----------------------------------------"

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "[警告] 配置文件不存在: $CONFIG"
    echo "[提示] 请先运行标定工具: ./run_simble_calib.sh <图片路径> $CONFIG"
fi

# 运行
cd "$BUILD_DIR"
"$EXECUTABLE" "$CONFIG"
