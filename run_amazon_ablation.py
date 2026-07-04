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

# ==========================================
# 0. 核心配置 (Amazon 结构消融实验 - 对齐 ML-1M 4变体)
# ==========================================
DATA_FILE = 'Electronics_5.json'
SEED = 2024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🏆 核心超参数
UNIFIED_EMBED_DIM = 128
SEQ_LEN = 10
BATCH_SIZE = 1024
EPOCHS = 5
PATIENCE = 2
NEG_SAMPLES = 4
GROUP_SIZE = 100  # 恢复 1:99 严苛大考

# 🔥 核心修正：最优参数对齐
SHARED_MLP_UNITS = (128,64)
ATT_UNITS = (128,64) # 🚀 增强 Attention 层
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
# 1. 模型库：完整版 + 对齐 ML-1M 的 4 大对称变体
# ==========================================

# 1.0 完整版 (Full)
class SKB_FMLP_Full(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(512, 256), mlp2_hidden_units=(512, 256),
                 att_hidden_units=(128, 64), dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP_Full, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.history_feat_names = history_feature_list
        q_name = history_feature_list[0]
        self.embed_dim = self.embedding_dict[q_name].embedding_dim

        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim, hidden_units=att_hidden_units, activation='relu',
                           device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)
        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, use_bn=True, device=self.device)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units, dropout_rate=dnn_dropout, use_bn=True,
                                device=self.device)

        self.vector_gate = nn.Sequential(nn.Linear(mlp1_hidden_units[-1], mlp1_hidden_units[-1]), nn.Sigmoid())
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        q_name = self.history_feat_names[0]
        query = self.embedding_dict[q_name](X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long())
        keys = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

        T = keys.size(1)
        query_rep = query.expand(-1, T, -1)
        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        sk_out = self.mlp_sk(dnn_input)
        beh_out = self.mlp_behavior(torch.cat([dnn_input, hist_attn], dim=-1))

        gate = self.vector_gate(sk_out)
        fusion_out = gate * beh_out + (1 - gate) * sk_out
        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return self.out(logit)


# 1.1 对应 ML-1M: w/o Vector Gating
class SKB_FMLP_NoGating(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(512, 256), mlp2_hidden_units=(512, 256),
                 att_hidden_units=(128, 64), dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP_NoGating, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.history_feat_names = history_feature_list
        q_name = history_feature_list[0]
        self.embed_dim = self.embedding_dict[q_name].embedding_dim

        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim, hidden_units=att_hidden_units, activation='relu',
                           device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)
        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, use_bn=True, device=self.device)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units, dropout_rate=dnn_dropout, use_bn=True,
                                device=self.device)

        # ❌ 无向量门控，直接预测
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        q_name = self.history_feat_names[0]
        query = self.embedding_dict[q_name](X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long())
        keys = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

        T = keys.size(1)
        query_rep = query.expand(-1, T, -1)
        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        sk_out = self.mlp_sk(dnn_input)
        beh_out = self.mlp_behavior(torch.cat([dnn_input, hist_attn], dim=-1))

        # 🔴 直接相加融合
        fusion_out = sk_out + beh_out
        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return self.out(logit)


# 1.2 对应 ML-1M: w/o Attention
class SKB_FMLP_NoAttention(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(512, 256), mlp2_hidden_units=(512, 256), dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP_NoAttention, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.history_feat_names = history_feature_list
        q_name = history_feature_list[0]
        self.embed_dim = self.embedding_dict[q_name].embedding_dim
        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, use_bn=True, device=self.device)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units, dropout_rate=dnn_dropout, use_bn=True,
                                device=self.device)
        self.vector_gate = nn.Sequential(nn.Linear(mlp1_hidden_units[-1], mlp1_hidden_units[-1]), nn.Sigmoid())
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        q_name = self.history_feat_names[0]
        keys = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

        # 🔴 直接平均池化
        hist_attn = keys.mean(dim=1)

        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        sk_out = self.mlp_sk(dnn_input)
        beh_out = self.mlp_behavior(torch.cat([dnn_input, hist_attn], dim=-1))
        gate = self.vector_gate(sk_out)
        fusion_out = gate * beh_out + (1 - gate) * sk_out
        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return self.out(logit)


# 1.3 对应 ML-1M: w/o RAG (这里替换为 w/o DualStream)
class SKB_FMLP_NoDualStream(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(512, 256), mlp2_hidden_units=(512, 256),
                 att_hidden_units=(128, 64), dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP_NoDualStream, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.history_feat_names = history_feature_list
        q_name = history_feature_list[0]
        self.embed_dim = self.embedding_dict[q_name].embedding_dim

        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim, hidden_units=att_hidden_units, activation='relu',
                           device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)
        input_dim = self.compute_input_dim(dnn_feature_columns)

        # 🔴 只有一个 MLP (单流)
        self.mlp_single = DNN(input_dim + self.embed_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, use_bn=True,
                              device=self.device)
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        q_name = self.history_feat_names[0]
        query = self.embedding_dict[q_name](X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long())
        keys = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

        T = keys.size(1)
        query_rep = query.expand(-1, T, -1)
        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        fusion_out = self.mlp_single(torch.cat([dnn_input, hist_attn], dim=-1))
        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return self.out(logit)


# 1.4 对应 ML-1M: w/o RAG (替换为 Semantic Only / 无历史行为)
class SKB_FMLP_SemanticOnly(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(512, 256), dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP_SemanticOnly, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)

        # 🔴 只有属性流，完全砍掉 Attention 和历史序列
        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, use_bn=True, device=self.device)
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # 🔴 只提取目标物品的特征，无视 hist_item_id
        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        sk_out = self.mlp_sk(dnn_input)

        logit = self.dnn_predict(sk_out) + self.linear_model(X)
        return self.out(logit)


# ==========================================
# 2. 通用评估与数据处理
# ==========================================
def calculate_metrics(y_true, y_pred, k_list=[5, 10]):
    # 基础指标计算
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)

    # 按照 1:99 协议重塑数据 (每组 100 个样本，index 0 为正样本)
    group_size = 100
    num_groups = len(y_true) // group_size
    y_true_r = y_true[:num_groups * group_size].reshape(num_groups, group_size)
    y_pred_r = y_pred[:num_groups * group_size].reshape(num_groups, group_size)

    metrics = {'AUC': auc, 'LogLoss': ll}

    # 为每个 K 初始化累加器
    for k in k_list:
        ndcg_sum, mrr_sum, hit_sum = 0.0, 0.0, 0.0
        prec_sum, recall_sum = 0.0, 0.0

        for i in range(num_groups):
            # 获取当前组预测分数的降序排名索引
            pred_scores = y_pred_r[i]
            # 真实正样本始终是 index 0
            rank = np.argsort(pred_scores)[::-1]
            pos = np.where(rank == 0)[0][0] + 1  # 真实排名位置 (从1开始)

            if pos <= k:
                # 排序指标
                ndcg_sum += 1.0 / np.log2(pos + 1)
                mrr_sum += 1.0 / pos
                hit_sum += 1.0

                # 分类指标逻辑
                # Recall@K = 命中数 / 1 (总正样本数)
                recall_sum += 1.0
                # Precision@K = 命中数 / K (推荐位总数)
                prec_sum += 1.0 / k

        # 计算均值
        avg_recall = recall_sum / num_groups
        avg_prec = prec_sum / num_groups

        metrics[f'NDCG@{k}'] = ndcg_sum / num_groups
        metrics[f'MRR@{k}'] = mrr_sum / num_groups
        metrics[f'HitRate@{k}'] = hit_sum / num_groups  # 传统推荐常叫 HR
        metrics[f'Recall@{k}'] = avg_recall
        metrics[f'Precision@{k}'] = avg_prec

        # 计算 F1-Score
        if (avg_prec + avg_recall) > 0:
            metrics[f'F1@{k}'] = 2 * (avg_prec * avg_recall) / (avg_prec + avg_recall)
        else:
            metrics[f'F1@{k}'] = 0.0

    return metrics


def prepare_amazon_data():
    print(f"[{datetime.now()}] >>> 解析 {DATA_FILE} (纯ID无外部特征版)...")
    data_list = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    df = pd.DataFrame(data_list)[['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user_id', 'item_id', 'timestamp']

    lbe_item = LabelEncoder()
    df['item_id'] = lbe_item.fit_transform(df['item_id']) + 1
    item_count = df['item_id'].max() + 1

    df = df.sort_values(['user_id', 'timestamp'])
    train_data, test_data = [], []
    item_pool = df['item_id'].unique()

    for uid, group in tqdm(df.groupby('user_id')):
        items = group['item_id'].tolist()
        if len(items) < 3: continue
        for i in range(1, len(items)):
            hist = items[max(0, i - SEQ_LEN):i]
            target = items[i]
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
            'item_id': df_tmp['item_id'].values,
            'hist_item_id': numpy_pad_sequences(df_tmp['hist'].tolist(), SEQ_LEN),
            'seq_length': np.array([min(len(h), SEQ_LEN) for h in df_tmp['hist']])
        }
        return x, df_tmp['label'].values

    train_x, train_y = format_x(train_data)
    test_x, test_y = format_x(test_data)

    feature_cols = [SparseFeat('item_id', item_count, UNIFIED_EMBED_DIM)]
    behavior_col = [
        VarLenSparseFeat(SparseFeat('hist_item_id', item_count, UNIFIED_EMBED_DIM, embedding_name='item_id'),
                         maxlen=SEQ_LEN, length_name='seq_length', combiner='sum')]
    return train_x, train_y, test_x, test_y, feature_cols, behavior_col


# ==========================================
# 3. 对称消融实验执行主循环
# ==========================================
def run_amazon_ablation_experiment():
    seed_everything()
    train_x, train_y, test_x, test_y, f_cols, b_cols = prepare_amazon_data()
    dnn_cols = f_cols + b_cols

    print(f"[{datetime.now()}] >>> 切分验证集 (10%)...")
    tr_idx, val_idx = train_test_split(range(len(train_y)), test_size=0.1, random_state=SEED)

    def subset_x(x_dict, indices):
        return {k: v[indices] for k, v in x_dict.items()}

    tr_x, tr_y = subset_x(train_x, tr_idx), train_y[tr_idx]
    val_x, val_y = subset_x(train_x, val_idx), train_y[val_idx]

    # 🎯 严苛对齐 ML-1M 的 1个Full + 4个核心变体 + 1个基线
    models = {

        'Full SKB-FMLP': lambda: SKB_FMLP_Full(
        f_cols, dnn_cols, ['item_id'],
        mlp1_hidden_units=SHARED_MLP_UNITS, mlp2_hidden_units=SHARED_MLP_UNITS,
        att_hidden_units=ATT_UNITS, dnn_dropout=SHARED_DROPOUT, task='binary', device=DEVICE),

    'w/o Vector Gating': lambda: SKB_FMLP_NoGating(
        f_cols, dnn_cols, ['item_id'],
        mlp1_hidden_units=SHARED_MLP_UNITS, mlp2_hidden_units=SHARED_MLP_UNITS,
        att_hidden_units=ATT_UNITS, dnn_dropout=SHARED_DROPOUT, task='binary', device=DEVICE),

    'w/o Attention': lambda: SKB_FMLP_NoAttention(
        f_cols, dnn_cols, ['item_id'],
        mlp1_hidden_units=SHARED_MLP_UNITS, mlp2_hidden_units=SHARED_MLP_UNITS,
        dnn_dropout=SHARED_DROPOUT, task='binary', device=DEVICE),

    'w/o DualStream': lambda: SKB_FMLP_NoDualStream(
        f_cols, dnn_cols, ['item_id'],
        mlp1_hidden_units=SHARED_MLP_UNITS,
        att_hidden_units=ATT_UNITS, dnn_dropout=SHARED_DROPOUT, task='binary', device=DEVICE),

    'w/o History (Semantic Only)': lambda: SKB_FMLP_SemanticOnly(
        f_cols, dnn_cols, ['item_id'],
        mlp1_hidden_units=SHARED_MLP_UNITS, dnn_dropout=SHARED_DROPOUT, task='binary', device=DEVICE),
    }

    results = []
    for m_name, builder in models.items():

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
                print(f"  验证集 AUC: {curr_val_auc:.4f}")
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
            met['Variant'] = m_name
            results.append(met)

            print(f"✓ {m_name} 最终 AUC: {met['AUC']:.4f}, NDCG@10: {met.get('NDCG@10', 0):.4f}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ {m_name} 崩溃: {e}")
            continue

    final_df = pd.DataFrame(results)
    # 按照 ML-1M 的列名顺序对齐
    cols_order = ['Variant', 'AUC', 'NDCG@5', 'MRR@5', 'HitRate@5', 'NDCG@10', 'MRR@10', 'HitRate@10']
    available_cols = [c for c in cols_order if c in final_df.columns]
    final_df = final_df[available_cols + [c for c in final_df.columns if c not in available_cols]]

    output_file = f'thesis_amazon_ablation_aligned_{datetime.now().strftime("%m%d_%H%M")}.csv'
    final_df.to_csv(output_file, index=False)


    print(final_df[['Variant', 'AUC', 'NDCG@10', 'MRR@10']].to_string(index=False))

if __name__ == "__main__":
    run_amazon_ablation_experiment()