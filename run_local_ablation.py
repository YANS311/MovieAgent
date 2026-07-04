import os
import sys
import django
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import gc
import copy
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. 环境初始化与路径锁定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')
django.setup()

from myapp.models import UserInfo, Movie, UserRating
from sentence_transformers import SentenceTransformer
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, combined_dnn_input
from deepctr_torch.models import DeepFM, DIN, WDL, DCNMix
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN

try:
    from finalmlp import FinalMLP
except ImportError:
    FinalMLP = None


# ==========================================
# 2. 自定义模型集群 (包含第三章基座与第四章全系变体)
# ==========================================

# 🚀 第三章基座模型: 纯 ID + 序列特征的双流网络
class SKB_FMLP_Online(BaseModel):
    """
    SKB-FMLP (第三章提出)：
    验证在多模态特征下，简单的 Early Fusion 不如 MAAN解耦架构。
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(256, 128), mlp2_hidden_units=(256, 128),
                 att_hidden_units=(256, 128), dnn_dropout=0.3, **kwargs):
        super(SKB_FMLP_Online, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.history_feat_names = history_feature_list
        first_feat = history_feature_list[0]
        self.embed_dim = self.embedding_dict[first_feat].embedding_dim

        # Attention 模块
        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim, hidden_units=att_hidden_units, activation='relu', device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)

        # 双流 MLP 主干
        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units, dropout_rate=dnn_dropout, device=self.device)

        # 向量门控
        self.vector_gate = nn.Sequential(nn.Linear(mlp1_hidden_units[-1], mlp1_hidden_units[-1]), nn.Sigmoid())
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        q_name = self.history_feat_names[0]
        query = self.embedding_dict[q_name](X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long())
        keys = self.embedding_dict[q_name](X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

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


# 🚀 多模态安全特征抓取基类
class BaseFusionModel(BaseModel):
    def extract_cf_content_features(self, X):
        u_idx = self.feature_index['user_id']
        m_idx = self.feature_index['movie_id']
        h_idx = self.feature_index['hist_movie_id']

        user_emb = self.embedding_dict['user_id'](X[:, u_idx[0]:u_idx[1]].long()).squeeze(1)
        movie_emb = self.embedding_dict['movie_id'](X[:, m_idx[0]:m_idx[1]].long()).squeeze(1)
        hist_emb = self.embedding_dict['movie_id'](X[:, h_idx[0]:h_idx[1]].long()).mean(dim=1)
        cf_vec = torch.cat([user_emb, movie_emb, hist_emb], dim=-1)

        g_idx = self.feature_index['genres']
        d_idx = self.feature_index['directors']
        genres_emb = self.embedding_dict['genres'](X[:, g_idx[0]:g_idx[1]].long()).mean(dim=1)
        directors_emb = self.embedding_dict['directors'](X[:, d_idx[0]:d_idx[1]].long()).mean(dim=1)

        rag_vals = [X[:, self.feature_index[f'rag_{i}'][0]:self.feature_index[f'rag_{i}'][1]] for i in range(16)]
        rag_emb = torch.cat(rag_vals, dim=-1)
        vis_vals = [X[:, self.feature_index[f'vis_{i}'][0]:self.feature_index[f'vis_{i}'][1]] for i in range(16)]
        vis_emb = torch.cat(vis_vals, dim=-1)

        content_vec = torch.cat([genres_emb, directors_emb, rag_emb, vis_emb], dim=-1)
        return user_emb, movie_emb, cf_vec, genres_emb, directors_emb, rag_emb, vis_emb, content_vec


class DirectFusion(BaseFusionModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, fuse_dim=128, mlp_hidden_units=(256, 128), dnn_dropout=0.3, embed_dim=256, **kwargs):
        super(DirectFusion, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.cf_proj = nn.Linear(embed_dim * 3, fuse_dim)
        self.content_proj = nn.Linear(embed_dim * 2 + 16 * 2, fuse_dim)
        self.bn = nn.BatchNorm1d(fuse_dim * 2)
        self.fusion_mlp = DNN(fuse_dim * 2, mlp_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.dnn_predict = nn.Linear(mlp_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        _, _, cf_vec, _, _, _, _, content_vec = self.extract_cf_content_features(X)
        fused = torch.cat([self.cf_proj(cf_vec), self.content_proj(content_vec)], dim=-1)
        fusion_out = self.fusion_mlp(self.bn(fused))
        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return torch.sigmoid(logit)


class GatedFusion(BaseFusionModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, fuse_dim=128, mlp_hidden_units=(256, 128), dnn_dropout=0.3, embed_dim=256, **kwargs):
        super(GatedFusion, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.cf_proj = nn.Linear(embed_dim * 3, fuse_dim)
        self.content_proj = nn.Linear(embed_dim * 2 + 16 * 2, fuse_dim)
        self.gate = nn.Sequential(nn.Linear(fuse_dim * 2, fuse_dim), nn.Sigmoid())
        self.bn = nn.BatchNorm1d(fuse_dim)
        self.fusion_mlp = DNN(fuse_dim, mlp_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.dnn_predict = nn.Linear(mlp_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        _, _, cf_vec, _, _, _, _, content_vec = self.extract_cf_content_features(X)
        c_proj, cont_proj = self.cf_proj(cf_vec), self.content_proj(content_vec)
        gate_weight = self.gate(torch.cat([c_proj, cont_proj], dim=-1))
        fused = gate_weight * c_proj + (1 - gate_weight) * cont_proj
        fusion_out = self.fusion_mlp(self.bn(fused))
        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return torch.sigmoid(logit)


class CrossAttFusion(BaseFusionModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, fuse_dim=128, mlp_hidden_units=(256, 128), dnn_dropout=0.3, embed_dim=256, **kwargs):
        super(CrossAttFusion, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.cf_proj = nn.Linear(embed_dim * 3, fuse_dim)
        self.gen_proj = nn.Linear(embed_dim, fuse_dim)
        self.dir_proj = nn.Linear(embed_dim, fuse_dim)
        self.rag_proj = nn.Linear(16, fuse_dim)
        self.vis_proj = nn.Linear(16, fuse_dim)

        self.cross_att = nn.MultiheadAttention(fuse_dim, num_heads=4, batch_first=True, dropout=dnn_dropout)
        self.bn = nn.BatchNorm1d(fuse_dim * 2)
        self.fusion_mlp = DNN(fuse_dim * 2, mlp_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.dnn_predict = nn.Linear(mlp_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        _, _, cf_vec, gen_emb, dir_emb, rag_emb, vis_emb, _ = self.extract_cf_content_features(X)
        cf_q = self.cf_proj(cf_vec).unsqueeze(1)

        t_gen, t_dir = self.gen_proj(gen_emb), self.dir_proj(dir_emb)
        t_rag, t_vis = self.rag_proj(rag_emb), self.vis_proj(vis_emb)
        content_tokens = torch.stack([t_gen, t_dir, t_rag, t_vis], dim=1)

        attn_out, _ = self.cross_att(cf_q, content_tokens, content_tokens)
        fused = torch.cat([cf_q.squeeze(1), attn_out.squeeze(1)], dim=-1)
        fusion_out = self.fusion_mlp(self.bn(fused))

        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return torch.sigmoid(logit)


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.linear(x) * self.gate(x)


# 🚀 终极版 MAAN：解耦双流对数几率架构 (支持动态维度)
class BiCrossAttFusion(BaseFusionModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, fuse_dim=64, mlp_hidden_units=(256, 128),
                 dnn_dropout=0.3, embed_dim=256, **kwargs):
        super(BiCrossAttFusion, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        # 🔥 修复: embed_dim 从硬编码 128 改为参数

        # --- Stage 1: ID 基座 ---
        self.att_dnn = DNN(inputs_dim=4 * embed_dim, hidden_units=(256, 128), activation='relu', device=self.device)
        self.att_linear = nn.Linear(128, 1)

        self.bn_id = nn.BatchNorm1d(embed_dim * 3)
        self.mlp_behavior = DNN(embed_dim * 3, mlp_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.vector_gate_sk = nn.Sequential(nn.Linear(mlp_hidden_units[-1], mlp_hidden_units[-1]), nn.Sigmoid())

        # --- 投影池化层 (🔥 改为动态维度) ---
        self.u_proj = GLU(embed_dim, fuse_dim)
        self.m_proj = GLU(embed_dim, fuse_dim)
        self.skb_proj = GLU(embed_dim, fuse_dim)
        self.cf_pool = GLU(embed_dim * 3, fuse_dim)

        self.gen_proj = GLU(embed_dim, fuse_dim)
        self.dir_proj = GLU(embed_dim, fuse_dim)
        self.rag_proj = GLU(16, fuse_dim)
        self.vis_proj = GLU(16, fuse_dim)
        self.content_pool = GLU(embed_dim * 2 + 16 * 2, fuse_dim)

        # ========================================================
        # 流 1: 微观精准狙击流 (Micro-Stream) -> NDCG
        # ========================================================
        self.cf2content = nn.MultiheadAttention(fuse_dim, num_heads=4, batch_first=True, dropout=dnn_dropout)
        self.content2cf = nn.MultiheadAttention(fuse_dim, num_heads=4, batch_first=True, dropout=dnn_dropout)

        self.bn_cross = nn.BatchNorm1d(fuse_dim * 2)
        self.cross_fc = nn.Linear(fuse_dim * 2, mlp_hidden_units[-1])
        self.alpha_gate = nn.Sequential(nn.Linear(mlp_hidden_units[-1] * 2, mlp_hidden_units[-1]), nn.Sigmoid())
        self.dnn_predict_micro = nn.Linear(mlp_hidden_units[-1], 1, bias=False)

        # ========================================================
        # 流 2: 宏观平滑匹配流 (Macro-Stream) -> GAUC
        # ========================================================
        self.macro_gate = nn.Sequential(nn.Linear(fuse_dim * 2, fuse_dim), nn.Sigmoid())
        self.macro_bn = nn.BatchNorm1d(fuse_dim)
        self.macro_mlp = DNN(fuse_dim, (128, 64), dropout_rate=dnn_dropout, device=self.device)
        self.dnn_predict_macro = nn.Linear(64, 1, bias=False)

        # ========================================================
        # 流融合器 (Logit Arbiter)
        # ========================================================
        self.logit_arbiter = nn.Linear(2, 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        u_emb, m_emb, _, gen_emb, dir_emb, rag_emb, vis_emb, content_vec = self.extract_cf_content_features(X)

        # Target Attention
        h_idx = self.feature_index['hist_movie_id']
        keys = self.embedding_dict['movie_id'](X[:, h_idx[0]:h_idx[1]].long())
        query_rep = m_emb.unsqueeze(1).expand(-1, keys.size(1), -1)

        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        # Stage 1 (ID Base)
        id_concat = self.bn_id(torch.cat([u_emb, m_emb, hist_attn], dim=-1))
        beh_out = self.mlp_behavior(id_concat)
        gate_sk = self.vector_gate_sk(beh_out)
        skb_fused = beh_out * gate_sk

        # Tokens Extraction
        cf_tokens = torch.stack([self.u_proj(u_emb), self.m_proj(m_emb), self.skb_proj(hist_attn)], dim=1)
        cf_query = self.cf_pool(id_concat)
        content_tokens = torch.stack([self.gen_proj(gen_emb), self.dir_proj(dir_emb),
                                      self.rag_proj(rag_emb), self.vis_proj(vis_emb)], dim=1)
        content_query = self.content_pool(content_vec)

        # 流 1: Micro
        attn_c2c, _ = self.cf2content(cf_query.unsqueeze(1), content_tokens, content_tokens)
        attn_c2f, _ = self.content2cf(content_query.unsqueeze(1), cf_tokens, cf_tokens)
        cross_out = self.cross_fc(self.bn_cross(torch.cat([attn_c2c.squeeze(1), attn_c2f.squeeze(1)], dim=-1)))
        alpha = self.alpha_gate(torch.cat([skb_fused, cross_out], dim=-1))
        att_final = skb_fused + alpha * cross_out
        logit_micro = self.dnn_predict_micro(att_final)

        # 流 2: Macro
        macro_gate_weight = self.macro_gate(torch.cat([cf_query, content_query], dim=-1))
        macro_fused = macro_gate_weight * cf_query + (1 - macro_gate_weight) * content_query
        macro_out = self.macro_mlp(self.macro_bn(macro_fused))
        logit_macro = self.dnn_predict_macro(macro_out)

        # Logit Arbiter Fusion
        stacked_logits = torch.cat([logit_micro, logit_macro], dim=-1)
        logit = self.logit_arbiter(stacked_logits) + self.linear_model(X)
        return torch.sigmoid(logit)


# ==========================================
# 3. 评估指标 与 数据切分 (复用你原本高效稳定的逻辑)
# ==========================================
def get_rank_metrics(y_true, y_pred, k=5, neg_count=99):
    group_size = neg_count + 1
    num_users = len(y_true) // group_size
    y_true_g = y_true[:num_users * group_size].reshape(num_users, group_size)
    y_pred_g = y_pred[:num_users * group_size].reshape(num_users, group_size)
    gauc_sum, ndcg_sum, mrr_sum, hit_sum, f1_sum, valid_users = 0, 0, 0, 0, 0, 0

    for i in range(num_users):
        if len(np.unique(y_true_g[i])) == 2:
            gauc_sum += roc_auc_score(y_true_g[i], y_pred_g[i])
            valid_users += 1
        pos_score = y_pred_g[i][0]
        # 计算正样本的排名
        rank = (y_pred_g[i] > pos_score).sum() + 1

        if rank <= k:
            ndcg_sum += 1.0 / np.log2(rank + 1)
            mrr_sum += 1.0 / rank
            hit_sum += 1.0
            # 在留一法下, F1@k = 2 / (k + 1) 当命中时
            f1_sum += 2.0 / (k + 1)

    return {
        'GAUC': gauc_sum / valid_users if valid_users > 0 else 0,
        f'NDCG@{k}': ndcg_sum / num_users,
        f'MRR@{k}': mrr_sum / num_users,
        f'Hit@{k}': hit_sum / num_users,
        f'F1@{k}': f1_sum / num_users
    }

def numpy_pad_sequences(sequences, maxlen):
    out = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        trunc = seq[-maxlen:] if len(seq) > 0 else seq
        out[i, :len(trunc)] = trunc
    return out

def load_local_data(device, seq_len, multimodal_dim, embed_dim):
    print(">>> [Data] 正在提取与清理数据...")
    ratings_qs = UserRating.objects.values('user_id', 'movie_id', 'score', 'comment_time')
    df_ratings = pd.DataFrame.from_records(ratings_qs).dropna(subset=['score'])
    df_ratings = df_ratings[df_ratings['score'] >= 7.0].copy()
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['comment_time'], utc=True).astype('int64') // 10 ** 9
    df_ratings = df_ratings.sort_values(['user_id', 'timestamp']).drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

    user_ids = df_ratings['user_id'].unique()
    users_qs = UserInfo.objects.filter(id__in=user_ids).values('id', 'sex', 'age', 'occupation')
    df_users = pd.DataFrame.from_records(users_qs).rename(columns={'id': 'user_id', 'sex': 'gender'})
    df_users.fillna(0, inplace=True)

    movie_ids = df_ratings['movie_id'].unique()
    movies_qs = Movie.objects.filter(id__in=movie_ids).prefetch_related('genres', 'directors')
    movie_dict, rag_texts, visual_vecs, mids_ordered = {}, [], [], []
    for m in tqdm(movies_qs.iterator(chunk_size=2000), total=movies_qs.count(), desc="解析电影多模态"):
        vis_vec = np.array(m.poster_embedding_json) if m.poster_embedding_json else np.zeros(512)
        mids_ordered.append(m.id)
        rag_texts.append(f"{m.title}. {m.summary or ''}")
        visual_vecs.append(vis_vec)
        movie_dict[m.id] = {'genres': list(m.genres.values_list('name', flat=True)),
                            'directors': list(m.directors.values_list('name', flat=True))}

    print(">>> [Data] 抽取语义特征...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    raw_rag = encoder.encode(rag_texts, batch_size=32, show_progress_bar=True)

    from sklearn.preprocessing import normalize as skl_normalize
    rag_init = PCA(n_components=embed_dim).fit_transform(raw_rag)
    rag_init = skl_normalize(rag_init, norm='l2', axis=1)
    rag_pca = PCA(n_components=multimodal_dim).fit_transform(raw_rag)
    visual_pca = PCA(n_components=multimodal_dim).fit_transform(np.array(visual_vecs))

    lbe_map = {f: LabelEncoder().fit(df) for f, df in zip(['user_id', 'movie_id', 'gender', 'age', 'occupation'],
                                                          [df_ratings['user_id'], df_ratings['movie_id'],
                                                           df_users['gender'], df_users['age'], df_users['occupation']])}
    df_ratings['enc_u'] = lbe_map['user_id'].transform(df_ratings['user_id']) + 1
    df_ratings['enc_m'] = lbe_map['movie_id'].transform(df_ratings['movie_id']) + 1

    print(">>> [Data] 严格三段式时间切分 + 混合负采样...")
    train_data, val_data, test_data = [], [], []
    all_encoded = lbe_map['movie_id'].transform(lbe_map['movie_id'].classes_) + 1

    item_counts = df_ratings['enc_m'].value_counts()
    hot_items = item_counts.head(max(100, int(len(item_counts) * 0.1))).index.tolist()
    num_hard, num_rand = int(99 * 0.3), 99 - int(99 * 0.3)

    for uid, group in tqdm(df_ratings.groupby('enc_u'), desc="构建序列集"):
        items = group['enc_m'].tolist()
        if len(items) < 8: continue
        full_set = set(items)

        def get_negs():
            negs = []
            for _ in range(num_rand):
                neg = random.choice(all_encoded)
                while neg in full_set: neg = random.choice(all_encoded)
                negs.append(neg)
            for _ in range(num_hard):
                neg = random.choice(hot_items)
                while neg in full_set: neg = random.choice(hot_items)
                negs.append(neg)
            return negs

        test_m, test_hist = items[-1], items[:-1][-seq_len:]
        test_data.append({'enc_u': uid, 'enc_m': test_m, 'hist': test_hist, 'label': 1})
        for neg in get_negs(): test_data.append({'enc_u': uid, 'enc_m': neg, 'hist': test_hist, 'label': 0})

        val_m, val_hist = items[-2], items[:-2][-seq_len:]
        val_data.append({'enc_u': uid, 'enc_m': val_m, 'hist': val_hist, 'label': 1})
        for neg in get_negs(): val_data.append({'enc_u': uid, 'enc_m': neg, 'hist': val_hist, 'label': 0})

        train_items = items[:-2]
        for i in range(1, len(train_items)):
            train_data.append({'enc_u': uid, 'enc_m': train_items[i], 'hist': train_items[max(0, i - seq_len):i], 'label': 1})
            neg = random.choice(all_encoded)
            while neg in set(train_items[:i + 1]): neg = random.choice(all_encoded)
            train_data.append({'enc_u': uid, 'enc_m': neg, 'hist': train_items[max(0, i - seq_len):i], 'label': 0})

    vocab_movie = len(lbe_map['movie_id'].classes_) + 1
    rag_m, vis_m = np.zeros((vocab_movie, multimodal_dim)), np.zeros((vocab_movie, multimodal_dim))
    rag_init_matrix = np.zeros((vocab_movie, embed_dim), dtype=np.float32)

    all_g, all_d = set(), set()
    [all_g.update(v['genres']) for v in movie_dict.values()]
    [all_d.update(v['directors']) for v in movie_dict.values()]
    g2idx, d2idx = {g: i + 1 for i, g in enumerate(all_g)}, {d: i + 1 for i, d in enumerate(all_d)}

    mid_to_enc = dict(zip(lbe_map['movie_id'].classes_, range(1, vocab_movie)))
    enc_g_list, enc_d_list = [[] for _ in range(vocab_movie)], [[] for _ in range(vocab_movie)]

    for i, mid in enumerate(mids_ordered):
        if mid not in mid_to_enc: continue
        eid = mid_to_enc[mid]
        rag_m[eid], vis_m[eid] = rag_pca[i], visual_pca[i]
        rag_init_matrix[eid] = rag_init[i]
        enc_g_list[eid] = [g2idx[g] for g in movie_dict[mid]['genres']]
        enc_d_list[eid] = [d2idx[d] for d in movie_dict[mid]['directors']]

    return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data), lbe_map, numpy_pad_sequences(
        enc_g_list, 3), numpy_pad_sequences(enc_d_list, 2), rag_m, vis_m, rag_init_matrix, g2idx, d2idx


# ==========================================
# 4. 实验启动与早停监控
# ==========================================
def run_experiments():
    UNIFIED_EMBED_DIM, MULTIMODAL_DIM, BATCH_SIZE = 256, 16, 2048
    SEED, DEVICE, SEQ_LEN = 2026, 'cuda' if torch.cuda.is_available() else 'cpu', 10
    EPOCHS, PATIENCE = 15, 3

    np.random.seed(SEED); torch.manual_seed(SEED)
    train_df, val_df, test_df, lbe_map, pad_g, pad_d, rag_m, vis_m, rag_init_matrix, g2idx, d2idx = load_local_data(
        DEVICE, SEQ_LEN, MULTIMODAL_DIM, UNIFIED_EMBED_DIM)

    linear_cols = [SparseFeat('user_id', len(lbe_map['user_id'].classes_) + 1, UNIFIED_EMBED_DIM),
                   SparseFeat('movie_id', len(lbe_map['movie_id'].classes_) + 1, UNIFIED_EMBED_DIM)]
    base_dnn_cols = linear_cols + [VarLenSparseFeat(
        SparseFeat('hist_movie_id', len(lbe_map['movie_id'].classes_) + 1, UNIFIED_EMBED_DIM, embedding_name='movie_id'), maxlen=SEQ_LEN, combiner='mean', length_name='seq_len')]
    kg_cols = [VarLenSparseFeat(SparseFeat('genres', len(g2idx) + 1, UNIFIED_EMBED_DIM), maxlen=3, combiner='mean'),
               VarLenSparseFeat(SparseFeat('directors', len(d2idx) + 1, UNIFIED_EMBED_DIM), maxlen=2, combiner='mean')]
    rag_cols = [DenseFeat(f'rag_{i}', 1) for i in range(MULTIMODAL_DIM)]
    vis_cols = [DenseFeat(f'vis_{i}', 1) for i in range(MULTIMODAL_DIM)]
    full_cols = base_dnn_cols + kg_cols + rag_cols + vis_cols

    # 实验任务配置
    experiments = {
        "MAAN (Chap.4)": {'cols': full_cols, 'model': 'MAAN', 'type': 'Ablation'},
        "SKB-FMLP (Chap.3)": {'cols': full_cols, 'model': 'SKB_FMLP', 'type': 'Comparison'},
        "w/o Gate": {'cols': full_cols, 'model': 'Direct', 'type': 'Ablation'},
        "w/o Attention": {'cols': full_cols, 'model': 'Gated', 'type': 'Ablation'},
        "w/o Bi-Attn": {'cols': full_cols, 'model': 'CrossAtt', 'type': 'Ablation'},
        # ========== 模态级消融 (Modality Ablation) ==========
        "w/o Visual": {'cols': full_cols, 'model': 'MAAN', 'type': 'Ablation',
                       'ablate': 'visual'},
        "w/o KG": {'cols': full_cols, 'model': 'MAAN', 'type': 'Ablation',
                   'ablate': 'kg'},
        # =====================================================
        "LR": {'cols': linear_cols, 'model': 'LR', 'type': 'Comparison'},
        "WDL": {'cols': full_cols, 'model': 'WDL', 'type': 'Comparison'},
        "DCNMix": {'cols': full_cols, 'model': 'DCNMix', 'type': 'Comparison'},
        "DeepFM": {'cols': full_cols, 'model': 'DeepFM', 'type': 'Comparison'},
        "DIN": {'cols': full_cols, 'model': 'DIN', 'type': 'Comparison'},
        "FinalMLP": {'cols': full_cols, 'model': 'FinalMLP', 'type': 'Comparison'},
    }

    def get_input(df):
        x = {'user_id': df['enc_u'].values.astype(np.int32), 'movie_id': df['enc_m'].values.astype(np.int32)}
        x['hist_movie_id'] = numpy_pad_sequences(df['hist'].tolist(), SEQ_LEN)
        x['seq_len'] = np.array([len(h) for h in df['hist']], dtype=np.int32)
        mids = df['enc_m'].values.astype(int)
        x['genres'], x['directors'] = pad_g[mids], pad_d[mids]
        for i in range(MULTIMODAL_DIM):
            x[f'rag_{i}'] = rag_m[mids, i].astype(np.float32).reshape(-1, 1)
            x[f'vis_{i}'] = vis_m[mids, i].astype(np.float32).reshape(-1, 1)
        return x, df['label'].values

    train_X, train_y = get_input(train_df)
    val_X, val_y = get_input(val_df)
    test_X, test_y = get_input(test_df)
    results_list = []

    print("\n" + "=" * 60)
    print(f"🚀 三段式严格评测启动 | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("=" * 60, flush=True)

    for name, config in experiments.items():
        print(f"\n>>> [RUN] 正在训练: {name}...", flush=True)
        try:
            # ============================================================
            # 🔬 模态级消融：在数据输入阶段将目标模态特征置零
            #    不改模型结构，不改 DataLoader，仅控制输入信号
            # ============================================================
            ablate_mode = config.get('ablate', None)
            cur_train_X = {k: v.copy() if hasattr(v, 'copy') else v for k, v in train_X.items()}
            cur_val_X   = {k: v.copy() if hasattr(v, 'copy') else v for k, v in val_X.items()}
            cur_test_X  = {k: v.copy() if hasattr(v, 'copy') else v for k, v in test_X.items()}

            if ablate_mode == 'visual':
                # 将海报视觉 PCA 向量全部置零（保留语义文本与知识图谱）
                for i in range(MULTIMODAL_DIM):
                    cur_train_X[f'vis_{i}'] = np.zeros_like(cur_train_X[f'vis_{i}'])
                    cur_val_X[f'vis_{i}']   = np.zeros_like(cur_val_X[f'vis_{i}'])
                    cur_test_X[f'vis_{i}']  = np.zeros_like(cur_test_X[f'vis_{i}'])
                print("   [Ablation] 已将视觉特征 (vis_0~vis_15) 全部置零 w/o Visual", flush=True)

            elif ablate_mode == 'kg':
                # 将知识图谱特征（genres + directors）全部置零
                cur_train_X['genres']    = np.zeros_like(cur_train_X['genres'])
                cur_val_X['genres']      = np.zeros_like(cur_val_X['genres'])
                cur_test_X['genres']     = np.zeros_like(cur_test_X['genres'])
                cur_train_X['directors'] = np.zeros_like(cur_train_X['directors'])
                cur_val_X['directors']   = np.zeros_like(cur_val_X['directors'])
                cur_test_X['directors']  = np.zeros_like(cur_test_X['directors'])
                print("   [Ablation] 已将图谱特征 (genres + directors) 全部置零 w/o KG", flush=True)
            # ============================================================

            # 🔥 全系严格对齐 128 维空间，控制变量法，搭配 Dropout与L2 正则抗过拟合
            if config['model'] == 'MAAN':
                model = BiCrossAttFusion(linear_cols, config['cols'], fuse_dim=64, mlp_hidden_units=(256, 128),
                                         dnn_dropout=0.1, l2_reg_embedding=1e-4, device=DEVICE)
            elif config['model'] == 'SKB_FMLP':
                # 第三章基座：参数量公平对齐
                model = SKB_FMLP_Online(linear_cols, config['cols'], history_feature_list=['movie_id'],
                                        mlp1_hidden_units=(256, 128), mlp2_hidden_units=(256, 128),
                                        att_hidden_units=(128, 64), dnn_dropout=0.1, l2_reg_embedding=1e-4,
                                        device=DEVICE)
            elif config['model'] == 'Direct':
                model = DirectFusion(linear_cols, config['cols'], fuse_dim=64, mlp_hidden_units=(256, 128),
                                     dnn_dropout=0.1, l2_reg_embedding=1e-4, embed_dim=UNIFIED_EMBED_DIM, device=DEVICE)
            elif config['model'] == 'Gated':
                model = GatedFusion(linear_cols, config['cols'], fuse_dim=64, mlp_hidden_units=(256, 128),
                                    dnn_dropout=0.1, l2_reg_embedding=1e-4, embed_dim=UNIFIED_EMBED_DIM, device=DEVICE)
            elif config['model'] == 'CrossAtt':
                model = CrossAttFusion(linear_cols, config['cols'], fuse_dim=64, mlp_hidden_units=(256, 128),
                                       dnn_dropout=0.1, l2_reg_embedding=1e-4, embed_dim=UNIFIED_EMBED_DIM, device=DEVICE)
            elif config['model'] == 'LR':
                model = WDL(linear_cols, dnn_feature_columns=[], task='binary', device=DEVICE)
            elif config['model'] == 'WDL':
                model = WDL(linear_cols, config['cols'], task='binary', device=DEVICE,dnn_dropout=0.1)
            elif config['model'] == 'DCNMix':
                model = DCNMix(linear_cols, config['cols'], task='binary', device=DEVICE,dnn_dropout=0.1)
            elif config['model'] == 'DIN':
                model = DIN(config['cols'], history_feature_list=['movie_id'], task='binary', device=DEVICE,dnn_dropout=0.1)
            elif config['model'] == 'FinalMLP':
                if FinalMLP is None: raise ImportError("finalmlp.py 未找到，请确保 FinalMLP 类可用")
                model = FinalMLP(linear_cols, config['cols'], config['cols'], mlp1_hidden_units=(256, 128),
                                 mlp2_hidden_units=(256, 128), task='binary', device=DEVICE,mlp1_dropout=0.1)
            else:
                model = DeepFM(linear_cols, config['cols'], task='binary', device=DEVICE)

            model.compile("adam", "binary_crossentropy", metrics=["auc"])

            # 注入 RAG 初始化并允许微调
            if hasattr(model, 'embedding_dict') and 'movie_id' in model.embedding_dict:
                model.embedding_dict['movie_id'].weight.data.copy_(torch.FloatTensor(rag_init_matrix).to(DEVICE))
                model.embedding_dict['movie_id'].weight.requires_grad = True

            best_val_gauc, best_weights, wait_counter = 0, None, 0

            for epoch in range(EPOCHS):
                model.fit(cur_train_X, train_y, batch_size=BATCH_SIZE, epochs=1, verbose=0)
                val_pred = model.predict(cur_val_X, batch_size=BATCH_SIZE).flatten()
                val_met = get_rank_metrics(val_y, val_pred)
                curr_gauc = val_met['GAUC']
                print(f"   [Epoch {epoch + 1}/{EPOCHS}] Val GAUC: {curr_gauc:.4f} | Val NDCG@5: {val_met['NDCG@5']:.4f}")

                if curr_gauc > best_val_gauc:
                    best_val_gauc = curr_gauc
                    best_weights = copy.deepcopy(model.state_dict())
                    wait_counter = 0
                else:
                    wait_counter += 1
                    if wait_counter >= PATIENCE:
                        print(f"   ⚠️ 触发早停! 回滚权重。")
                        break

            if best_weights: model.load_state_dict(best_weights)
            final_pred = model.predict(cur_test_X, batch_size=BATCH_SIZE).flatten()
            final_met = get_rank_metrics(test_y, final_pred)
            print(f"   🏁 {name} 测试集最终结果: GAUC={final_met['GAUC']:.4f} | NDCG@5={final_met['NDCG@5']:.4f} | F1@5={final_met['F1@5']:.4f}")

            final_met.update({'Exp_Name': name, 'Type': config['type']})
            results_list.append(final_met)
            del model; torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error: {e}")

    if results_list:
        pd.DataFrame(results_list).to_csv(
            os.path.join(BASE_DIR, f"thesis_vfinal_{datetime.now().strftime('%m%d_%H%M')}.csv"), index=False)
        print("\n🎉 实验全部完成，报告已生成！你可以直接把这份 CSV 贴进论文了！")

if __name__ == "__main__":
    run_experiments()