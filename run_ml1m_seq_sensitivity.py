import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import random
import gc
import copy
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss

# DeepCTR-Torch Imports
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, combined_dnn_input
from deepctr_torch.models import *
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN

from sentence_transformers import SentenceTransformer

# ==========================================
# 0. 全局固定配置 (控制变量)
# ==========================================
DATA_PATH = './ml-1m/'
SEED = 2024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔥 固定最佳超参，只变 SEQ_LEN
FIXED_DROPOUT = 0.1
FIXED_PCA_DIM = 128
FIXED_ATT_UNITS = (512, 256)

EPOCHS = 15
NEG_EVAL_COUNT = 99


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def numpy_pad_sequences(sequences, maxlen, padding='post', value=0):
    out = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if not seq: continue
        trunc = seq[:maxlen] if padding == 'post' else seq[-maxlen:]
        if padding == 'post':
            out[i, :len(trunc)] = trunc
        else:
            out[i, -len(trunc):] = trunc
    return out


# ==========================================
# 1. SKB-FMLP v5.3 (动态架构版)
# ==========================================
class SKB_FMLP(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(256, 128), mlp2_hidden_units=(256, 128),
                 att_hidden_units=(128, 64),
                 dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)

        self.history_feat_names = history_feature_list
        first_feat = history_feature_list[0]
        self.embed_dim = self.embedding_dict[first_feat].embedding_dim

        # 动态 Attention
        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim,
                           hidden_units=att_hidden_units,
                           activation='relu', device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)

        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units,
                                dropout_rate=dnn_dropout, device=self.device)

        self.vector_gate = nn.Sequential(
            nn.Linear(mlp1_hidden_units[-1], mlp1_hidden_units[-1]),
            nn.Sigmoid()
        )
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
        return torch.sigmoid(logit)


# ==========================================
# 2. 评估指标
# ==========================================
def get_metrics(y_true, y_pred, k_list=[5, 10]):
    """
    更新版评估指标函数：包含排序指标(NDCG, MRR)与分类指标(Recall, Precision, F1)
    适用于 Leave-one-out (1:N) 负采样协议
    """
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



# ==========================================
# 3. 动态数据处理 (核心：SEQ_LEN 参数化)
# ==========================================
def process_data_for_seq_len(target_seq_len):
    print(f"\n>>> [Preprocessing] Building dataset for SEQ_LEN = {target_seq_len} ...")

    # 1. 加载元数据
    users = pd.read_csv(os.path.join(DATA_PATH, 'users.dat'), sep='::', header=None, engine='python',
                        encoding='latin-1', names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    movies = pd.read_csv(os.path.join(DATA_PATH, 'movies.dat'), sep='::', header=None, engine='python',
                         encoding='latin-1', names=['movie_id', 'title', 'genres'])
    ratings = pd.read_csv(os.path.join(DATA_PATH, 'ratings.dat'), sep='::', header=None, engine='python',
                          encoding='latin-1', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    lbe_map = {f: LabelEncoder().fit(users[f].astype(str)) if f in users else LabelEncoder().fit(movies[f].astype(str))
               for f in ['user_id', 'movie_id', 'gender', 'age', 'occupation']}

    # 2. RAG Embedding (固定使用 64 维)
    print(f"   -> Encoding RAG Vectors (Dim={FIXED_PCA_DIM})...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    raw_emb = encoder.encode([f"{m['title']} {m['genres']}" for _, m in movies.iterrows()], batch_size=512)
    rag_pca = PCA(n_components=FIXED_PCA_DIM).fit_transform(raw_emb)
    rag_pca = normalize(rag_pca, norm='l2', axis=1)
    movie_to_rag = dict(zip(movies['movie_id'], rag_pca))

    genre_set = set('|'.join(movies['genres']).split('|'))
    genre2idx = {g: i + 1 for i, g in enumerate(genre_set)}
    vocab_movie = len(lbe_map['movie_id'].classes_) + 1

    rag_matrix = np.zeros((vocab_movie, FIXED_PCA_DIM))
    encoded_genres = [[] for _ in range(vocab_movie)]
    movie_raw_genres = dict(zip(movies['movie_id'], movies['genres']))

    for i, rid_str in enumerate(lbe_map['movie_id'].classes_):
        rid = int(rid_str)
        rag_matrix[i + 1] = movie_to_rag.get(rid, np.zeros(FIXED_PCA_DIM))
        encoded_genres[i + 1] = [genre2idx[g] for g in movie_raw_genres.get(rid, "").split('|') if g in genre2idx]

    # 3. 按时间切分 (关键：根据 target_seq_len 截取历史)
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    ratings['enc_u'] = lbe_map['user_id'].transform(ratings['user_id'].astype(str)) + 1
    ratings['enc_m'] = lbe_map['movie_id'].transform(ratings['movie_id'].astype(str)) + 1

    hot_items = list(ratings['enc_m'].value_counts().head(500).index)
    all_items = list(range(1, vocab_movie))
    train_l, val_l, test_l = [], [], []

    for uid, group in tqdm(ratings.groupby('enc_u'), desc=f"Splitting (Len={target_seq_len})"):
        items = group['enc_m'].tolist()
        if len(items) < 5: continue
        full_set = set(items)

        # Test (Last Item)
        # 注意：这里取最后 target_seq_len 个作为历史
        test_l.append({'user_id': uid, 'movie_id': items[-1], 'hist': items[:-1][-target_seq_len:], 'label': 1})
        for _ in range(NEG_EVAL_COUNT):
            neg = random.choice(hot_items) if random.random() < 0.5 else random.choice(all_items)
            while neg in full_set: neg = random.choice(all_items)
            test_l.append({'user_id': uid, 'movie_id': neg, 'hist': items[:-1][-target_seq_len:], 'label': 0})

        # Val (Second Last)
        val_l.append({'user_id': uid, 'movie_id': items[-2], 'hist': items[:-2][-target_seq_len:], 'label': 1})
        for _ in range(NEG_EVAL_COUNT):
            neg = random.choice(all_items)
            while neg in full_set: neg = random.choice(all_items)
            val_l.append({'user_id': uid, 'movie_id': neg, 'hist': items[:-2][-target_seq_len:], 'label': 0})

        # Train (Sliding Window)
        train_items = items[:-2]
        cur_set = {items[0]}
        for i in range(1, len(train_items)):
            # 这里的 max(0, i - target_seq_len) 是关键
            hist_slice = train_items[max(0, i - target_seq_len):i]
            train_l.append({'user_id': uid, 'movie_id': train_items[i], 'hist': hist_slice, 'label': 1})

            neg = random.choice(all_items)
            while neg == train_items[i] or neg in cur_set: neg = random.choice(all_items)
            train_l.append({'user_id': uid, 'movie_id': neg, 'hist': hist_slice, 'label': 0})
            cur_set.add(train_items[i])

    def merge(df):
        u_feat = users[['user_id', 'gender', 'age', 'occupation']].copy()
        u_feat['enc_u'] = lbe_map['user_id'].transform(u_feat['user_id'].astype(str)) + 1
        for f in ['gender', 'age', 'occupation']: u_feat[f] = lbe_map[f].transform(u_feat[f].astype(str)) + 1
        return pd.merge(df, u_feat[['enc_u', 'gender', 'age', 'occupation']], left_on='user_id', right_on='enc_u',
                        how='left')

    return merge(pd.DataFrame(train_l)), merge(pd.DataFrame(val_l)), merge(
        pd.DataFrame(test_l)), lbe_map, numpy_pad_sequences(encoded_genres, 5), rag_matrix, genre2idx


# ==========================================
# 4. 主运行循环
# ==========================================
def run_seq_sensitivity_benchmark():
    seed_everything()

    # 待测试的序列长度
    SEQ_LENS_TO_TEST = [10, 20, 50, 100]
    all_results = []

    for seq_len in SEQ_LENS_TO_TEST:
        print(f"\n{'=' * 60}")
        print(f"🚀 Running Sensitivity Test for SEQ_LEN: {seq_len}")
        print(f"{'=' * 60}")

        # 1. 动态生成数据
        train_df, val_df, test_df, lbe_map, pad_genres, rag_vectors, genre2idx = process_data_for_seq_len(seq_len)

        # 2. 动态调整 Batch Size (防止长序列 OOM)
        current_batch_size = 2048 if seq_len <= 50 else 1024
        print(f"⚙️ Adjusted Batch Size to: {current_batch_size}")

        # 3. 特征定义 (maxlen=seq_len)
        profile_cols = [SparseFeat(f, len(lbe_map[f].classes_) + 1, FIXED_PCA_DIM) for f in
                        ['gender', 'age', 'occupation']]
        movie_id_col = SparseFeat('movie_id', len(lbe_map['movie_id'].classes_) + 1, FIXED_PCA_DIM,
                                  embedding_name='movie_id')
        rag_col = DenseFeat('rag_vec', FIXED_PCA_DIM)
        kg_col = VarLenSparseFeat(SparseFeat('genres', len(genre2idx) + 1, FIXED_PCA_DIM), maxlen=5, combiner='mean')

        # 🔥 关键修改：seq_col 的 maxlen 必须与当前循环一致
        seq_col = VarLenSparseFeat(
            SparseFeat('hist_movie_id', len(lbe_map['movie_id'].classes_) + 1, FIXED_PCA_DIM,
                       embedding_name='movie_id'),
            maxlen=seq_len,  # <--- 动态传入
            length_name='sl', combiner='mean')

        linear_cols = profile_cols + [rag_col]
        dnn_cols = profile_cols + [movie_id_col] + [kg_col] + [seq_col] + [rag_col]

        # 4. 构造输入
        def get_input(df):
            x = {f: df[f].values.astype(np.int32) for f in ['gender', 'age', 'occupation', 'movie_id']}
            mids = df['movie_id'].values.astype(int)
            x.update({'rag_vec': rag_vectors[mids], 'genres': pad_genres[mids],
                      'hist_movie_id': numpy_pad_sequences(df['hist'].tolist(), seq_len),  # <--- Pad 到当前长度
                      'sl': np.array([len(h) for h in df['hist']], dtype=np.int32)})
            return x, df['label'].values

        tr_X, tr_y = get_input(train_df)
        va_X, va_y = get_input(val_df)
        te_X, te_y = get_input(test_df)

        # 5. 训练模型 (SKB-FMLP)
        print(f"--- Training SKB-FMLP (Len={seq_len}) ---")
        try:
            model = SKB_FMLP(
                linear_cols, dnn_cols,
                history_feature_list=['movie_id'],
                att_hidden_units=FIXED_ATT_UNITS,
                dnn_dropout=FIXED_DROPOUT,
                task='binary', device=DEVICE
            )

            # 初始化 RAG 权重
            if hasattr(model, 'embedding_dict') and 'movie_id' in model.embedding_dict:
                model.embedding_dict['movie_id'].weight.data.copy_(torch.FloatTensor(rag_vectors).to(DEVICE))
                model.embedding_dict['movie_id'].weight.requires_grad = False

            model.compile("adam", "binary_crossentropy", metrics=["auc"])

            best_auc, patience, counter, best_weights = 0, 3, 0, None
            for epoch in range(EPOCHS):
                model.fit(tr_X, tr_y, batch_size=current_batch_size, epochs=1, verbose=0, validation_data=(va_X, va_y))
                v_pred = model.predict(va_X, current_batch_size).flatten()
                v_auc = roc_auc_score(va_y, v_pred)
                print(f"Len {seq_len} | Ep {epoch + 1} Val AUC: {v_auc:.4f}")

                if v_auc > best_auc:
                    best_auc, counter, best_weights = v_auc, 0, copy.deepcopy(model.state_dict())
                else:
                    counter += 1
                    if counter >= patience: break

            if best_weights: model.load_state_dict(best_weights)

            met = get_metrics(te_y, model.predict(te_X, current_batch_size).flatten())
            met.update({'Model': 'SKB-FMLP', 'Seq_Len': seq_len})
            print(f"✅ SKB-FMLP (Len={seq_len}): AUC={met['AUC']:.4f}, NDCG@10={met['NDCG@10']:.4f}")
            all_results.append(met)

            del model;
            gc.collect();
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"!!! Error in Len {seq_len}: {e}")
            import traceback;
            traceback.print_exc()

    # 6. 保存
    df_res = pd.DataFrame(all_results)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    df_res.to_csv(f'thesis_seqlen_sensitivity_{ts}.csv', index=False)
    print("\nSensitivity Analysis Completed!")
    print(df_res[['Model', 'Seq_Len', 'AUC', 'NDCG@10', 'Recall@10']])


if __name__ == "__main__":
    run_seq_sensitivity_benchmark()