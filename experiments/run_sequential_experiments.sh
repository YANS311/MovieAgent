#!/bin/bash
# ==========================================
# 顺序执行消融实验 + 网格搜索 (后台运行)
# 日志独立保存，两实验不会争抢显存
# ==========================================
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$BASE_DIR/experiment_logs"
mkdir -p "$LOG_DIR"

cd "$BASE_DIR"

echo "================================================"
echo "  顺序实验启动"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志目录: $LOG_DIR"
echo "================================================"
echo ""
echo "[1/2] run_local_ablation.py"
echo "----------------------------------------"

python3 -u run_local_ablation.py 2>&1 | tee "$LOG_DIR/local_ablation_$(date +%m%d_%H%M).log"

echo ""
echo "[2/2] run_grid_search.py"
echo "----------------------------------------"

python3 -u run_grid_search.py 2>&1 | tee "$LOG_DIR/grid_search_$(date +%m%d_%H%M).log"

echo ""
echo "================================================"
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志保存在: $LOG_DIR"
echo "================================================"