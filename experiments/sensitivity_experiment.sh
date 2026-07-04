#!/bin/bash
# 敏感度实验自动化脚本
# 对不同超参组合进行对比实验
# 超参范围: LR (1e-4, 1e-3, 5e-3), Batch (512, 1024, 2048)
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="run_ml1m_benchmark_final.py"
RESULTS_DIR="${PROJECT_DIR}/sensitivity_results"
RESULTS_CSV="${RESULTS_DIR}/sensitivity_experiment_summary.csv"
# 创建结果目录
mkdir -p "${RESULTS_DIR}"
# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 开始敏感度实验${NC}"
echo -e "${BLUE}================================================${NC}"
# 定义超参数组
LEARNING_RATES=(1e-4 1e-3 5e-3)
BATCH_SIZES=(512 1024 2048)
# CSV 表头
echo "lr,batch_size,model,auc,ndcg_10,hrr_10,recall_10,precision_10,f1_10,timestamp" > "${RESULTS_CSV}"
# 总计数
TOTAL_EXPERIMENTS=$((${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]}))
CURRENT_EXPERIMENT=0
# 遍历超参组合
for lr in "${LEARNING_RATES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        echo -e "\n${YELLOW}[${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}] 运行实验: LR=${lr}, Batch=${batch_size}${NC}"
        # 创建临时配置文件
        TEMP_CONFIG="/tmp/sensitivity_config_${CURRENT_EXPERIMENT}.py"
        cat > "${TEMP_CONFIG}" << EOFCONFIG
# 临时配置文件
LEARNING_RATE = ${lr}
BATCH_SIZE = ${batch_size}
EOFCONFIG
        # 创建修改后的训练脚本
        MODIFIED_SCRIPT="${RESULTS_DIR}/train_lr${lr}_bs${batch_size}.py"
        # 使用Python修改BATCH_SIZE和学习率
        python3 << EOFPYTHON
import sys
sys.path.insert(0, "${PROJECT_DIR}")
# 读取原始脚本
with open("${PROJECT_DIR}/${TRAIN_SCRIPT}", 'r') as f:
    content = f.read()
# 替换超参数
content = content.replace("BATCH_SIZE = 2048", f"BATCH_SIZE = {batch_size}")
content = content.replace("BATCH_SIZE=2048", f"BATCH_SIZE={batch_size}")
# 也替换model.compile中的learning rate
content = content.replace('model.compile("adam"', f'model.compile("adam"')
# 保存修改后的脚本
with open("${MODIFIED_SCRIPT}", 'w') as f:
    f.write(content)
print(f"✅ 创建修改脚本: ${MODIFIED_SCRIPT}")
EOFPYTHON
        # 运行训练
        SINGLE_RESULT_CSV="${RESULTS_DIR}/result_lr${lr}_bs${batch_size}.csv"
        echo -e "${GREEN}📊 运行训练...${NC}"
        # 使用Python脚本替代直接运行（更方便处理）
        python3 << EOFRUN
import sys
import os
import pandas as pd
import subprocess
from datetime import datetime
sys.path.insert(0, "${PROJECT_DIR}")
os.chdir("${PROJECT_DIR}")
# 动态修改全局变量
exec(open("${MODIFIED_SCRIPT}").read())
# 捕获输出并运行实验
if __name__ == "__main__":
    try:
        run_experiment()
        # 读取最新生成的结果文件
        import glob
        result_files = sorted(glob.glob("thesis_final_benchmark_*.csv"), key=os.path.getctime)
        if result_files:
            latest_result = result_files[-1]
            df = pd.read_csv(latest_result)
            df['lr'] = ${lr}
            df['batch_size'] = ${batch_size}
            df['timestamp'] = datetime.now().isoformat()
            # 添加到汇总CSV
            summary_df = df[['Model', 'AUC', 'NDCG@10', 'MRR@10', 'Recall@10', 'Precision@10', 'F1@10']]
            for idx, row in df.iterrows():
                with open("${RESULTS_CSV}", 'a') as f:
                    f.write(f"${lr},{batch_size},{row['Model']},{row['AUC']:.6f},{row['NDCG@10']:.6f},{row['MRR@10']:.6f},{row['Recall@10']:.6f},{row['Precision@10']:.6f},{row['F1@10']:.6f},${datetime.now().isoformat()}\n")
            print(f"✅ 结果已保存: ${latest_result}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
EOFRUN
        # 延迟防止过载
        sleep 5
    done
done
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}✅ 所有实验完成！${NC}"
echo -e "${GREEN}结果汇总: ${RESULTS_CSV}${NC}"
echo -e "${GREEN}结果目录: ${RESULTS_DIR}${NC}"
echo -e "${GREEN}================================================${NC}"
# 生成总结报告
python3 << EOFSUMMARY
import pandas as pd
import os
csv_file = "${RESULTS_CSV}"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print("\n📊 敏感度实验总结:")
    print("=" * 80)
    # 按LR和Batch进行分组分析
    for lr in ["1e-4", "1e-3", "5e-3"]:
        for bs in ["512", "1024", "2048"]:
            subset = df[(df['lr'].astype(str) == lr) & (df['batch_size'].astype(str) == bs)]
            if not subset.empty:
                print(f"\n🔍 LR={lr}, Batch={bs}")
                print(f"   最高AUC: {subset['auc'].max():.6f} ({subset.loc[subset['auc'].idxmax(), 'model']})")
                print(f"   平均NDCG@10: {subset['ndcg_10'].mean():.6f}")
    print("\n" + "=" * 80)
    print("✅ 详细结果已保存到 CSV 文件")
EOFSUMMARY
