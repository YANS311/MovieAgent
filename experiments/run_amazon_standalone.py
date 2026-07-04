import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import gc
import json
import copy
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# DeepCTR-Torch 核心组件
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN
from deepctr_torch.models import *

try:
    from finalmlp import FinalMLP
except ImportError:
    FinalMLP = None

# ==========================================
# 0. 核心配置 (致敬 ML-1M 巅峰版)
# ==========================================
DATA_FILE = 'Electronics_5.json'
SEED = 2024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🏆 核心超参数
UNIFIED_EMBED_DIM = 32
SEQ_LEN = 10
BATCH_SIZE = 1024
EPOCHS = 5
PATIENCE = 2
NEG_SAMPLES = 4
GROUP_SIZE = 100  # 🎯 恢复 1:99 的严苛大考

# 🌟 控制变量：对齐所有基线
SHARED_MLP_UNITS = (128,64)
SHARED_DROPOUT = 0.1


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def numpy_pad_sequences(sequences, maxlen, value=0):
    out = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if not seq: continue
        trunc = seq[-maxlen:]
        out[i, -len(trunc):] = trunc
    return out


# ==========================================
# 1. 核心架构：SKB-FMLP (1:1 像素级复刻 ML-1M)
# ==========================================
class SKB_FMLP(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(256, 128), mlp2_hidden_units=(256, 128),
                 att_hidden_units=(128, 64), dnn_dropout=0.3, **kwargs):
        super(SKB_FMLP, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)

        self.history_feat_names = history_feature_list
        q_name = history_feature_list[0]
        self.embed_dim = self.embedding_dict[q_name].embedding_dim

        # 🎯 灵魂 1：复刻 ML-1M 的动态 Attention 网络
        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim,
                           hidden_units=att_hidden_units,
                           activation='relu', device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)

        input_dim = self.compute_input_dim(dnn_feature_columns)

        # 属性流
        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, use_bn=True, device=self.device)

        # 🎯 灵魂 2：行为流强制融合 Attention 输出！(这是与 FinalMLP 拉开差距的关键)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units,
                                dropout_rate=dnn_dropout, use_bn=True, device=self.device)

        # 🎯 灵魂 3：复刻 ML-1M 的向量门控融合 (Vector Gating)
        self.vector_gate = nn.Sequential(
            nn.Linear(mlp1_hidden_units[-1], mlp1_hidden_units[-1]),
            nn.Sigmoid()
        )
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # --- A. 计算动态 Attention ---
        q_name = self.history_feat_names[0]
        query = self.embedding_dict[q_name](X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long())
        keys = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

        T = keys.size(1)
        query_rep = query.expand(-1, T, -1)

        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        # --- B. 双流不对称计算 ---
        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        sk_out = self.mlp_sk(dnn_input)
        beh_out = self.mlp_behavior(torch.cat([dnn_input, hist_attn], dim=-1))

        # --- C. 向量门控融合 ---
        gate = self.vector_gate(sk_out)
        fusion_out = gate * beh_out + (1 - gate) * sk_out

        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return self.out(logit)


# ==========================================
# 2. 评估函数
# ==========================================
def calculate_metrics(y_true, y_pred, k_list=[5, 10]):
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)

    if len(y_true) % GROUP_SIZE != 0:
        return {"AUC": auc, "LogLoss": ll}

    num_groups = len(y_true) // GROUP_SIZE
    y_true_matrix = y_true.reshape(num_groups, GROUP_SIZE)
    y_pred_matrix = y_pred.reshape(num_groups, GROUP_SIZE)

    metrics = {'AUC': auc, 'LogLoss': ll}
    for k in k_list:
        ndcg_sum, mrr_sum, hit_sum = 0.0, 0.0, 0.0
        prec_sum, recall_sum = 0.0, 0.0

        for i in range(num_groups):
            preds = y_pred_matrix[i]
            rank = np.argsort(preds)[::-1]
            pos = np.where(rank == 0)[0][0] + 1

            if pos <= k:
                ndcg_sum += 1.0 / np.log2(pos + 1)
                mrr_sum += 1.0 / pos
                hit_sum += 1.0
                recall_sum += 1.0
                prec_sum += 1.0 / k

        metrics[f'NDCG@{k}'] = ndcg_sum / num_groups
        metrics[f'MRR@{k}'] = mrr_sum / num_groups
        metrics[f'HitRate@{k}'] = hit_sum / num_groups
        metrics[f'Precision@{k}'] = prec_sum / num_groups
        metrics[f'Recall@{k}'] = recall_sum / num_groups
        if (metrics[f'Precision@{k}'] + metrics[f'Recall@{k}']) > 0:
            metrics[f'F1@{k}'] = 2 * (metrics[f'Precision@{k}'] * metrics[f'Recall@{k}']) / (
                        metrics[f'Precision@{k}'] + metrics[f'Recall@{k}'])
        else:
            metrics[f'F1@{k}'] = 0.0

    return metrics


# ==========================================
# 3. 数据预处理 (ID-Free，全均匀采样防错位)
# ==========================================
def prepare_amazon_data():
    print(f"[{datetime.now()}] >>> 解析 {DATA_FILE}...")
    data_list = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    df = pd.DataFrame(data_list)[['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user_id', 'item_id', 'timestamp']

    # 🎯 仅对 item_id 进行编码，彻底抛弃 user_id 的编码和统计
    lbe_item = LabelEncoder()
    df['item_id'] = lbe_item.fit_transform(df['item_id']) + 1
    item_count = df['item_id'].max() + 1

    print(f"[{datetime.now()}] >>> 构造序列与测试集(1:99 纯净无 ID 版)...")
    df = df.sort_values(['user_id', 'timestamp'])
    train_data, test_data = [], []
    item_pool = df['item_id'].unique()

    for uid, group in tqdm(df.groupby('user_id')):
        items = group['item_id'].tolist()
        if len(items) < 3: continue

        for i in range(1, len(items)):
            hist = items[max(0, i - SEQ_LEN):i]
            target = items[i]

            # 🎯 字典中绝不包含 'user_id'
            sample = {'item_id': target, 'hist': hist}

            if i == len(items) - 1:
                test_data.append({**sample, 'label': 1})
                for _ in range(99):
                    neg = random.choice(item_pool)
                    while neg in items: neg = random.choice(item_pool)
                    test_data.append({'item_id': neg, 'hist': hist, 'label': 0})
            else:
                train_data.append({**sample, 'label': 1})
                for _ in range(NEG_SAMPLES):
                    neg = random.choice(item_pool)
                    while neg in items: neg = random.choice(item_pool)
                    train_data.append({'item_id': neg, 'hist': hist, 'label': 0})

    def format_x(list_data):
        df_tmp = pd.DataFrame(list_data)
        x = {
            # 🎯 格式化特征时绝不提取 user_id
            'item_id': df_tmp['item_id'].values,
            'hist_item_id': numpy_pad_sequences(df_tmp['hist'].tolist(), SEQ_LEN),
            'seq_length': np.array([min(len(h), SEQ_LEN) for h in df_tmp['hist']])
        }
        return x, df_tmp['label'].values

    train_x, train_y = format_x(train_data)
    test_x, test_y = format_x(test_data)

    # 🎯 核心特征注册中，有且仅有 item_id，强行掐断所有模型的作弊通路！
    feature_cols = [
        SparseFeat('item_id', item_count, UNIFIED_EMBED_DIM)
    ]
    behavior_col = [
        # 🎯 保持 sum 防止过拟合，防止报错
        VarLenSparseFeat(SparseFeat('hist_item_id', item_count, UNIFIED_EMBED_DIM, embedding_name='item_id'),
                         maxlen=SEQ_LEN, length_name='seq_length', combiner='sum')
    ]

    return train_x, train_y, test_x, test_y, feature_cols, behavior_col


# ==========================================
# 4. 全家桶实验主循环
# ==========================================
def run_thesis_experiment():
    seed_everything()
    train_x, train_y, test_x, test_y, f_cols, b_cols = prepare_amazon_data()
    dnn_cols = f_cols + b_cols

    print(f"[{datetime.now()}] >>> 切分验证集 (10%)...")
    tr_idx, val_idx = train_test_split(range(len(train_y)), test_size=0.1, random_state=SEED)

    def subset_x(x_dict, indices):
        return {k: v[indices] for k, v in x_dict.items()}

    tr_x, tr_y = subset_x(train_x, tr_idx), train_y[tr_idx]
    val_x, val_y = subset_x(train_x, val_idx), train_y[val_idx]

    models = {
         'LR': lambda: WDL(f_cols, [], task='binary', device=DEVICE),
         'WDL': lambda: WDL(f_cols, dnn_cols, dnn_hidden_units=SHARED_MLP_UNITS, dnn_dropout=SHARED_DROPOUT,
                          dnn_use_bn=True, task='binary', device=DEVICE),
         'DeepFM': lambda: DeepFM(f_cols, dnn_cols, dnn_hidden_units=SHARED_MLP_UNITS, dnn_dropout=SHARED_DROPOUT,
                                  dnn_use_bn=True, task='binary', device=DEVICE),
         'DCN-V2': lambda: DCNMix(f_cols, dnn_cols, dnn_hidden_units=SHARED_MLP_UNITS, dnn_dropout=SHARED_DROPOUT,
                                  dnn_use_bn=True, task='binary', device=DEVICE),
         'DIN': lambda: DIN(dnn_cols, ['item_id'], dnn_hidden_units=SHARED_MLP_UNITS, dnn_dropout=SHARED_DROPOUT,
                            dnn_use_bn=True, task='binary', device=DEVICE),
     }

    if FinalMLP:
         models['FinalMLP'] = lambda: FinalMLP(f_cols, dnn_cols, dnn_cols, mlp1_hidden_units=SHARED_MLP_UNITS,
                                               mlp2_hidden_units=SHARED_MLP_UNITS, mlp1_dropout=SHARED_DROPOUT,
                                               mlp2_dropout=SHARED_DROPOUT, dnn_use_bn=True, task='binary',
                                               device=DEVICE)

    models['SKB-FMLP (Proposed)'] = lambda: SKB_FMLP(f_cols, dnn_cols, ['item_id'], mlp1_hidden_units=SHARED_MLP_UNITS,
                                                     mlp2_hidden_units=SHARED_MLP_UNITS, dnn_dropout=SHARED_DROPOUT,
                                                     task='binary', device=DEVICE)

    results = []
    for m_name, builder in models.items():
        print(f"\n>>>> 正在训练: {m_name}")
        try:
            model = builder()
            emb_params = [p for n, p in model.named_parameters() if 'embedding' in n]
            net_params = [p for n, p in model.named_parameters() if 'embedding' not in n]
            optimizer = torch.optim.Adam(
                [{'params': emb_params, 'weight_decay': 1e-4}, {'params': net_params, 'weight_decay': 5e-3}], lr=5e-4)
            model.compile(optimizer, "binary_crossentropy", metrics=["auc"])

            best_auc, patience_counter, best_weights = 0, 0, None
            for epoch in range(EPOCHS):
                model.fit(tr_x, tr_y, batch_size=BATCH_SIZE, epochs=1, verbose=1)
                val_pred = model.predict(val_x, batch_size=BATCH_SIZE).flatten()
                curr_val_auc = roc_auc_score(val_y, val_pred)
                print(f"[*] {m_name} 验证集 AUC: {curr_val_auc:.4f}")
                if curr_val_auc > best_auc:
                    best_auc, patience_counter, best_weights = curr_val_auc, 0, copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print("🛑 触发早停！")
                        break

            if best_weights: model.load_state_dict(best_weights)
            pred = model.predict(test_x, batch_size=BATCH_SIZE).flatten()
            met = calculate_metrics(test_y, pred)
            met['Model'] = m_name
            results.append(met)
            print(f">>> {m_name} 最终 AUC: {met['AUC']:.4f}, NDCG@10: {met.get('NDCG@10', 0):.4f}")

            del model;
            gc.collect();
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ {m_name} 崩溃: {e}");
            continue

    final_df = pd.DataFrame(results)
    final_df.to_csv(f'thesis_amazon_Overnight_{datetime.now().strftime("%m%d_%H%M")}.csv', index=False)
    print("\n" + "=" * 50 + "\n全家桶战报\n" + "=" * 50)
    print(final_df[['Model', 'AUC', 'NDCG@10', 'MRR@10']])


if __name__ == "__main__":
    run_thesis_experiment()