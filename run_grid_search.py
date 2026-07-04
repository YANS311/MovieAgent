#!/usr/bin/env python3
"""
===========================================================================
 MAAN 显存安全网格搜索 (12GB VRAM Safe Grid Search) v2
 严格对齐 run_local_ablation.py 的数据管道与训练流程
===========================================================================
 修复: embed_dim 不再作为模型显式参数 (避免 deepctr BaseModel 构建异常)
 训练: 直接使用 model.fit() + model.predict()，对齐 local_ablation
===========================================================================
"""

import os, sys, gc, random, itertools, copy, traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ==========================================
# 0. 环境初始化
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "movie.settings")
import django
django.setup()

from myapp.models import UserInfo, Movie, UserRating
from sentence_transformers import SentenceTransformer
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN

# ==========================================
# 全局缓存: BGE 编码只做一次
# ==========================================
_CACHED_RAW_RAG = None        # BGE 编码结果 (N, 384)
_CACHED_VISUAL_VECS = None    # CLIP 512 维原始向量 (N, 512)
_CACHED_MOVIE_DICT = None     # {mid: {genres, directors}}
_CACHED_MIDS_ORDERED = None   # 与 raw_rag/visual_vecs 对齐的 mid 顺序列表
_CACHED_ALL_GENRES = None     # 所有 genres 集合
_CACHED_ALL_DIRECTORS = None  # 所有 directors 集合

# ==========================================
# 1. 模型定义 (严格对齐 run_local_ablation.py)
# ==========================================
class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.linear(x) * self.gate(x)


class BaseFusionModel(BaseModel):
    def extract_cf_content_features(self, X):
        u_idx = self.feature_index["user_id"]
        m_idx = self.feature_index["movie_id"]
        h_idx = self.feature_index["hist_movie_id"]
        user_emb  = self.embedding_dict["user_id"](X[:, u_idx[0]:u_idx[1]].long()).squeeze(1)
        movie_emb = self.embedding_dict["movie_id"](X[:, m_idx[0]:m_idx[1]].long()).squeeze(1)
        hist_emb  = self.embedding_dict["movie_id"](X[:, h_idx[0]:h_idx[1]].long()).mean(dim=1)
        cf_vec = torch.cat([user_emb, movie_emb, hist_emb], dim=-1)

        g_idx = self.feature_index["genres"]
        d_idx = self.feature_index["directors"]
        genres_emb    = self.embedding_dict["genres"](X[:, g_idx[0]:g_idx[1]].long()).mean(dim=1)
        directors_emb = self.embedding_dict["directors"](X[:, d_idx[0]:d_idx[1]].long()).mean(dim=1)

        rag_vals = [X[:, self.feature_index[f"rag_{i}"][0]:self.feature_index[f"rag_{i}"][1]] for i in range(16)]
        rag_emb = torch.cat(rag_vals, dim=-1)
        vis_vals = [X[:, self.feature_index[f"vis_{i}"][0]:self.feature_index[f"vis_{i}"][1]] for i in range(16)]
        vis_emb = torch.cat(vis_vals, dim=-1)
        content_vec = torch.cat([genres_emb, directors_emb, rag_emb, vis_emb], dim=-1)
        return user_emb, movie_emb, cf_vec, genres_emb, directors_emb, rag_emb, vis_emb, content_vec


class BiCrossAttFusion(BaseFusionModel):
    """MAAN: 解耦双流对数几率架构 (严格对齐 local_ablation 版本)
       注意: 不再将 embed_dim 作为显式构造参数，由 SparseFeat 决定 embedding 维度
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, fuse_dim=64,
                 mlp_hidden_units=(256, 128), dnn_dropout=0.3, **kwargs):
        super().__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        # 从 feature_columns 推断 embed_dim (取第一个 SparseFeat 的维度)
        embed_dim = dnn_feature_columns[0].embedding_dim

        # --- Stage 1: ID 基座 ---
        self.att_dnn = DNN(4 * embed_dim, (256, 128), activation="relu", device=self.device)
        self.att_linear = nn.Linear(128, 1)
        self.bn_id = nn.BatchNorm1d(embed_dim * 3)
        self.mlp_behavior = DNN(embed_dim * 3, mlp_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.vector_gate_sk = nn.Sequential(nn.Linear(mlp_hidden_units[-1], mlp_hidden_units[-1]), nn.Sigmoid())

        # --- 投影池化层 ---
        self.u_proj, self.m_proj, self.skb_proj = GLU(embed_dim, fuse_dim), GLU(embed_dim, fuse_dim), GLU(embed_dim, fuse_dim)
        self.cf_pool = GLU(embed_dim * 3, fuse_dim)
        self.gen_proj, self.dir_proj = GLU(embed_dim, fuse_dim), GLU(embed_dim, fuse_dim)
        self.rag_proj, self.vis_proj = GLU(16, fuse_dim), GLU(16, fuse_dim)
        self.content_pool = GLU(embed_dim * 2 + 16 * 2, fuse_dim)

        # 流 1: Micro
        self.cf2content = nn.MultiheadAttention(fuse_dim, 4, batch_first=True, dropout=dnn_dropout)
        self.content2cf = nn.MultiheadAttention(fuse_dim, 4, batch_first=True, dropout=dnn_dropout)
        self.bn_cross = nn.BatchNorm1d(fuse_dim * 2)
        self.cross_fc = nn.Linear(fuse_dim * 2, mlp_hidden_units[-1])
        self.alpha_gate = nn.Sequential(nn.Linear(mlp_hidden_units[-1] * 2, mlp_hidden_units[-1]), nn.Sigmoid())
        self.dnn_predict_micro = nn.Linear(mlp_hidden_units[-1], 1, bias=False)

        # 流 2: Macro
        self.macro_gate = nn.Sequential(nn.Linear(fuse_dim * 2, fuse_dim), nn.Sigmoid())
        self.macro_bn = nn.BatchNorm1d(fuse_dim)
        self.macro_mlp = DNN(fuse_dim, (128, 64), dropout_rate=dnn_dropout, device=self.device)
        self.dnn_predict_macro = nn.Linear(64, 1, bias=False)

        # Logit Arbiter
        self.logit_arbiter = nn.Linear(2, 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        u_emb, m_emb, _, gen_emb, dir_emb, rag_emb, vis_emb, content_vec = self.extract_cf_content_features(X)
        h_idx = self.feature_index["hist_movie_id"]
        keys = self.embedding_dict["movie_id"](X[:, h_idx[0]:h_idx[1]].long())
        query_rep = m_emb.unsqueeze(1).expand(-1, keys.size(1), -1)
        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        id_concat = self.bn_id(torch.cat([u_emb, m_emb, hist_attn], dim=-1))
        beh_out = self.mlp_behavior(id_concat)
        gate_sk = self.vector_gate_sk(beh_out)
        skb_fused = beh_out * gate_sk

        cf_tokens = torch.stack([self.u_proj(u_emb), self.m_proj(m_emb), self.skb_proj(hist_attn)], dim=1)
        cf_query = self.cf_pool(id_concat)
        content_tokens = torch.stack([self.gen_proj(gen_emb), self.dir_proj(dir_emb),
                                      self.rag_proj(rag_emb), self.vis_proj(vis_emb)], dim=1)
        content_query = self.content_pool(content_vec)

        attn_c2c, _ = self.cf2content(cf_query.unsqueeze(1), content_tokens, content_tokens)
        attn_c2f, _ = self.content2cf(content_query.unsqueeze(1), cf_tokens, cf_tokens)
        cross_out = self.cross_fc(self.bn_cross(torch.cat([attn_c2c.squeeze(1), attn_c2f.squeeze(1)], dim=-1)))
        alpha = self.alpha_gate(torch.cat([skb_fused, cross_out], dim=-1))
        att_final = skb_fused + alpha * cross_out
        logit_micro = self.dnn_predict_micro(att_final)

        macro_gate_weight = self.macro_gate(torch.cat([cf_query, content_query], dim=-1))
        macro_fused = macro_gate_weight * cf_query + (1 - macro_gate_weight) * content_query
        macro_out = self.macro_mlp(self.macro_bn(macro_fused))
        logit_macro = self.dnn_predict_macro(macro_out)

        stacked_logits = torch.cat([logit_micro, logit_macro], dim=-1)
        logit = self.logit_arbiter(stacked_logits) + self.linear_model(X)
        return torch.sigmoid(logit)


# ==========================================
# 2. 评估指标 (对齐 local_ablation)
# ==========================================
def get_rank_metrics(y_true, y_pred, k=5, neg_count=99):
    group_size = neg_count + 1
    num_users = len(y_true) // group_size
    y_true_g = y_true[:num_users * group_size].reshape(num_users, group_size)
    y_pred_g = y_pred[:num_users * group_size].reshape(num_users, group_size)

    gauc_sum = ndcg_sum = mrr_sum = hit_sum = f1_sum = prec_sum = 0.0
    valid_users = 0.0

    for i in range(num_users):
        if len(np.unique(y_true_g[i])) == 2:
            gauc_sum += roc_auc_score(y_true_g[i], y_pred_g[i])
            valid_users += 1
        pos_score = y_pred_g[i][0]
        rank = (y_pred_g[i] > pos_score).sum() + 1
        if rank <= k:
            ndcg_sum += 1.0 / np.log2(rank + 1)
            mrr_sum  += 1.0 / rank
            hit_sum  += 1.0
            f1_sum   += 2.0 / (k + 1)
        top_k_preds = y_pred_g[i].argsort()[::-1][:k]
        prec_sum += y_true_g[i][top_k_preds].sum() / k

    return {
        "GAUC":       gauc_sum / valid_users if valid_users > 0 else 0.0,
        f"NDCG@{k}":  ndcg_sum / num_users,
        f"MRR@{k}":   mrr_sum  / num_users,
        f"Hit@{k}":   hit_sum  / num_users,
        f"mprec@{k}": prec_sum / num_users,
        f"F1@{k}":    f1_sum   / num_users,
    }


def numpy_pad_sequences(sequences, maxlen):
    out = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        trunc = seq[-maxlen:] if len(seq) > 0 else seq
        out[i, :len(trunc)] = trunc
    return out


# ==========================================
# 3. 数据加载 (严格对齐 run_local_ablation.py 的 load_local_data)
# ==========================================
def _ensure_global_cache(device):
    """首次调用时加载 CLIP 视觉 + BGE 文本 + KG，缓存为全局变量"""
    global _CACHED_RAW_RAG, _CACHED_VISUAL_VECS, _CACHED_MOVIE_DICT
    global _CACHED_MIDS_ORDERED, _CACHED_ALL_GENRES, _CACHED_ALL_DIRECTORS

    if _CACHED_RAW_RAG is not None:
        return  # 已缓存

    print(">>> [Cache] 首次加载多模态数据，仅执行一次...", flush=True)
    movie_ids = UserRating.objects.values_list("movie_id", flat=True).distinct()
    movies_qs = Movie.objects.filter(id__in=movie_ids).prefetch_related("genres", "directors")

    mdict, texts, vvecs, mids_ord = {}, [], [], []
    all_g, all_d = set(), set()
    for m in tqdm(movies_qs.iterator(chunk_size=2000), total=movies_qs.count(), desc="解析电影多模态"):
        vv = np.array(m.poster_embedding_json) if m.poster_embedding_json else np.zeros(512)
        mids_ord.append(m.id)
        texts.append(f"{m.title}. {m.summary or ''}")
        vvecs.append(vv)
        g_list = list(m.genres.values_list("name", flat=True))
        d_list = list(m.directors.values_list("name", flat=True))
        mdict[m.id] = {"genres": g_list, "directors": d_list}
        all_g.update(g_list)
        all_d.update(d_list)

    _CACHED_VISUAL_VECS = np.array(vvecs, dtype=np.float32)
    _CACHED_MOVIE_DICT = mdict
    _CACHED_MIDS_ORDERED = mids_ord
    _CACHED_ALL_GENRES = all_g
    _CACHED_ALL_DIRECTORS = all_d

    # BGE 编码 (CPU)
    print(f"   [Encoder] BGE CPU 编码...", flush=True)
    torch.cuda.empty_cache()
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    _CACHED_RAW_RAG = enc.encode(texts, batch_size=32, show_progress_bar=True).astype(np.float32)
    del enc; gc.collect()
    torch.cuda.empty_cache()

    print("   ✅ 全局缓存就绪", flush=True)


def load_combo_data(seq_len, embed_dim, device):
    """
    完全对齐 run_local_ablation.py 的 load_local_data 函数结构。
    每次调用重新 PCA + 重新构建 train/val/test (开销可控)。
    """
    from sklearn.preprocessing import normalize as skl_normalize
    mm_dim = 16

    _ensure_global_cache(device)

    raw_rag = _CACHED_RAW_RAG
    visual_vecs = _CACHED_VISUAL_VECS
    movie_dict = _CACHED_MOVIE_DICT
    mids_ordered = _CACHED_MIDS_ORDERED
    all_g = _CACHED_ALL_GENRES
    all_d = _CACHED_ALL_DIRECTORS

    # --- 与 load_local_data 完全一致的数据流 ---
    print(">>> [Data] 正在提取与清理数据...", flush=True)
    ratings_qs = UserRating.objects.values("user_id", "movie_id", "score", "comment_time")
    df_ratings = pd.DataFrame.from_records(ratings_qs).dropna(subset=["score"])
    df_ratings = df_ratings[df_ratings["score"] >= 7.0].copy()
    df_ratings["timestamp"] = pd.to_datetime(df_ratings["comment_time"], utc=True).astype("int64") // 10**9
    df_ratings = df_ratings.sort_values(["user_id", "timestamp"]).drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    user_ids = df_ratings["user_id"].unique()
    users_qs = UserInfo.objects.filter(id__in=user_ids).values("id", "sex", "age", "occupation")
    df_users = pd.DataFrame.from_records(users_qs).rename(columns={"id": "user_id", "sex": "gender"})
    df_users.fillna(0, inplace=True)

    print(">>> [Data] 抽取语义特征 (PCA)...", flush=True)
    rag_init = PCA(n_components=embed_dim).fit_transform(raw_rag)
    rag_init = skl_normalize(rag_init, norm="l2", axis=1).astype(np.float32)
    rag_pca = PCA(n_components=mm_dim).fit_transform(raw_rag).astype(np.float32)
    visual_pca = PCA(n_components=mm_dim).fit_transform(visual_vecs).astype(np.float32)

    lbe_map = {
        "user_id":     LabelEncoder().fit(df_ratings["user_id"]),
        "movie_id":    LabelEncoder().fit(df_ratings["movie_id"]),
        "gender":      LabelEncoder().fit(df_users["gender"]),
        "age":         LabelEncoder().fit(df_users["age"]),
        "occupation":  LabelEncoder().fit(df_users["occupation"]),
    }
    df_ratings["enc_u"] = lbe_map["user_id"].transform(df_ratings["user_id"]) + 1
    df_ratings["enc_m"] = lbe_map["movie_id"].transform(df_ratings["movie_id"]) + 1

    print(">>> [Data] 严格三段式时间切分 + 混合负采样...", flush=True)
    train_data, val_data, test_data = [], [], []
    all_encoded = lbe_map["movie_id"].transform(lbe_map["movie_id"].classes_) + 1

    item_counts = df_ratings["enc_m"].value_counts()
    hot_items = item_counts.head(max(100, int(len(item_counts) * 0.1))).index.tolist()
    num_hard, num_rand = int(99 * 0.3), 99 - int(99 * 0.3)

    for uid, group in tqdm(df_ratings.groupby("enc_u"), desc="构建序列集"):
        items = group["enc_m"].tolist()
        if len(items) < 8:
            continue
        full_set = set(items)

        def get_negs():
            negs = []
            for _ in range(num_rand):
                n = random.choice(all_encoded)
                while n in full_set:
                    n = random.choice(all_encoded)
                negs.append(n)
            for _ in range(num_hard):
                n = random.choice(hot_items)
                while n in full_set:
                    n = random.choice(hot_items)
                negs.append(n)
            return negs

        test_m, test_hist = items[-1], items[:-1][-seq_len:]
        test_data.append({"enc_u": uid, "enc_m": test_m, "hist": test_hist, "label": 1})
        for neg in get_negs():
            test_data.append({"enc_u": uid, "enc_m": neg, "hist": test_hist, "label": 0})

        val_m, val_hist = items[-2], items[:-2][-seq_len:]
        val_data.append({"enc_u": uid, "enc_m": val_m, "hist": val_hist, "label": 1})
        for neg in get_negs():
            val_data.append({"enc_u": uid, "enc_m": neg, "hist": val_hist, "label": 0})

        train_items = items[:-2]
        for i in range(1, len(train_items)):
            h = train_items[max(0, i - seq_len):i]
            train_data.append({"enc_u": uid, "enc_m": train_items[i], "hist": h, "label": 1})
            n = random.choice(all_encoded)
            while n in set(train_items[:i + 1]):
                n = random.choice(all_encoded)
            train_data.append({"enc_u": uid, "enc_m": n, "hist": h, "label": 0})

    vocab_movie = len(lbe_map["movie_id"].classes_) + 1
    rag_m = np.zeros((vocab_movie, mm_dim), dtype=np.float32)
    vis_m = np.zeros((vocab_movie, mm_dim), dtype=np.float32)
    rag_init_matrix = np.zeros((vocab_movie, embed_dim), dtype=np.float32)

    g2idx = {g: i + 1 for i, g in enumerate(all_g)}
    d2idx = {d: i + 1 for i, d in enumerate(all_d)}

    mid_to_enc = dict(zip(lbe_map["movie_id"].classes_, range(1, vocab_movie)))
    enc_g_list = [[] for _ in range(vocab_movie)]
    enc_d_list = [[] for _ in range(vocab_movie)]

    for i, mid in enumerate(mids_ordered):
        if mid not in mid_to_enc:
            continue
        eid = mid_to_enc[mid]
        rag_m[eid] = rag_pca[i]
        vis_m[eid] = visual_pca[i]
        rag_init_matrix[eid] = rag_init[i]
        enc_g_list[eid] = [g2idx[g] for g in movie_dict[mid]["genres"]]
        enc_d_list[eid] = [d2idx[d] for d in movie_dict[mid]["directors"]]

    pad_g = numpy_pad_sequences(enc_g_list, 3)
    pad_d = numpy_pad_sequences(enc_d_list, 2)

    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)
    test_df  = pd.DataFrame(test_data)
    del train_data, val_data, test_data; gc.collect()

    # --- get_input (与 run_local_ablation 完全一致) ---
    def get_input(df):
        x = {"user_id": df["enc_u"].values.astype(np.int32),
             "movie_id": df["enc_m"].values.astype(np.int32)}
        x["hist_movie_id"] = numpy_pad_sequences(df["hist"].tolist(), seq_len)
        x["seq_len"] = np.array([len(h) for h in df["hist"]], dtype=np.int32)
        mids = df["enc_m"].values.astype(int)
        x["genres"]    = pad_g[mids]
        x["directors"] = pad_d[mids]
        for i in range(mm_dim):
            x[f"rag_{i}"] = rag_m[mids, i].astype(np.float32).reshape(-1, 1)
            x[f"vis_{i}"] = vis_m[mids, i].astype(np.float32).reshape(-1, 1)
        return x, df["label"].values.astype(np.float32)

    train_X, train_y = get_input(train_df)
    val_X,   val_y   = get_input(val_df)
    test_X,  test_y  = get_input(test_df)

    del train_df, val_df, test_df; gc.collect()
    torch.cuda.empty_cache()

    return (train_X, train_y, val_X, val_y, test_X, test_y,
            lbe_map, g2idx, d2idx, rag_init_matrix)


# ==========================================
# 4. 训练入口 (严格对齐 local_ablation 的 model.fit 风格)
# ==========================================
def train_one_combo(train_X, train_y, val_X, val_y, test_X, test_y,
                    lbe_map, g2idx, d2idx, rag_init_matrix,
                    embed_dim, dnn_dropout, seq_len, device,
                    batch_size=2048, epochs=15, patience=3):
    mm_dim = 16
    linear_cols = [
        SparseFeat("user_id",  len(lbe_map["user_id"].classes_)  + 1, embed_dim),
        SparseFeat("movie_id", len(lbe_map["movie_id"].classes_) + 1, embed_dim),
    ]
    hist_col = VarLenSparseFeat(
        SparseFeat("hist_movie_id", len(lbe_map["movie_id"].classes_) + 1, embed_dim,
                   embedding_name="movie_id"),
        maxlen=seq_len, combiner="mean", length_name="seq_len")
    kg_cols = [
        VarLenSparseFeat(SparseFeat("genres",    len(g2idx) + 1, embed_dim), maxlen=3, combiner="mean"),
        VarLenSparseFeat(SparseFeat("directors", len(d2idx) + 1, embed_dim), maxlen=2, combiner="mean"),
    ]
    rag_cols = [DenseFeat(f"rag_{i}", 1) for i in range(mm_dim)]
    vis_cols = [DenseFeat(f"vis_{i}", 1) for i in range(mm_dim)]
    full_cols = linear_cols + [hist_col] + kg_cols + rag_cols + vis_cols

    l2_strength = 1e-3 if embed_dim >= 256 else 1e-4
    dropout_adjusted = min(dnn_dropout + 0.1, 0.5) if embed_dim >= 256 else dnn_dropout

    # 🔥 不再传 embed_dim 给模型构造器！由 SparseFeat 决定
    model = BiCrossAttFusion(
        linear_cols, full_cols, fuse_dim=64, mlp_hidden_units=(256, 128),
        dnn_dropout=dropout_adjusted,
        l2_reg_embedding=l2_strength, device=device)

    # 注入 RAG 初始化 embedding
    if hasattr(model, "embedding_dict") and "movie_id" in model.embedding_dict:
        model.embedding_dict["movie_id"].weight.data.copy_(
            torch.FloatTensor(rag_init_matrix).to(device))
        model.embedding_dict["movie_id"].weight.requires_grad = True

    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    best_val_gauc, best_weights, wait_counter = 0.0, None, 0

    for ep in range(epochs):
        model.fit(train_X, train_y, batch_size=batch_size, epochs=1, verbose=0)
        val_pred = model.predict(val_X, batch_size=batch_size).flatten()
        val_met = get_rank_metrics(val_y, val_pred)
        curr_gauc = val_met["GAUC"]

        print(f"     [Epoch {ep+1:2d}/{epochs}] "
              f"Val GAUC={curr_gauc:.4f} | NDCG@5={val_met['NDCG@5']:.4f}", flush=True)

        if curr_gauc > best_val_gauc:
            best_val_gauc = curr_gauc
            best_weights = copy.deepcopy(model.state_dict())
            wait_counter = 0
        else:
            wait_counter += 1
            if wait_counter >= patience:
                print(f"     ⚠️ Early stop (patience={patience})", flush=True)
                break

    if best_weights:
        model.load_state_dict(best_weights)
        del best_weights

    # 最佳权重下的预测
    val_pred = model.predict(val_X, batch_size=batch_size).flatten()
    test_pred = model.predict(test_X, batch_size=batch_size).flatten()
    val_met_best = get_rank_metrics(val_y, val_pred)
    test_met = get_rank_metrics(test_y, test_pred)

    del model, val_pred, test_pred
    return best_val_gauc, val_met_best, test_met


def safe_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==========================================
# 5. 网格搜索主引擎
# ==========================================
CSV_COLUMNS = [
    "EmbedDim", "Dropout", "SeqLen",
    "Test GAUC", "Test NDCG@5", "Test MRR@5", "Test Hit@5", "Test mprec@5", "Test F1@5",
    "Val GAUC", "Val NDCG@5", "Val MRR@5", "Status", "ErrorMsg"
]


class GridSearchRunner:
    def __init__(self, csv_path="grid_search_12g_safe.csv"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 2026
        self.batch_size = 2048
        self.epochs = 15
        self.patience = 3
        self.csv_path = os.path.join(BASE_DIR, csv_path)
        self.completed = 0

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # 预热全局缓存
        _ensure_global_cache(self.device)

        pd.DataFrame(columns=CSV_COLUMNS).to_csv(self.csv_path, index=False)
        print(f"📄 CSV 初始化: {self.csv_path}", flush=True)

    def append_row(self, row_dict):
        row = pd.DataFrame([row_dict])
        row.to_csv(self.csv_path, mode="a", header=False, index=False)
        self.completed += 1
        print(f"   💾 已写入 CSV (累计 {self.completed} 组)", flush=True)

    def run(self):
        print("╔" + "=" * 70 + "╗")
        print("║   MAAN 显存安全网格搜索 v2 (对齐 local_ablation 管道)       ║")
        print("╚" + "=" * 70 + "╝")
        print(f"  Device:     {'CUDA' if self.device != 'cpu' else 'CPU'}")
        print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total:      4×5×3 = 60 combos")
        print("-" * 70, flush=True)

        embed_dims    = [32, 64, 128, 256,512]
        dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        seq_lens      = [10]

        total = len(embed_dims) * len(dropout_rates) * len(seq_lens)
        combo_list = list(itertools.product(embed_dims, dropout_rates, seq_lens))

        for idx, (dim, drop, seq) in enumerate(combo_list):
            combo_str = (f"[{idx+1:2d}/{total}] "
                         f"EmbedDim={dim:3d} | Dropout={drop:.1f} | SeqLen={seq:2d}")
            print(f"\n{'='*70}")
            print(f">>> {combo_str}")
            print(f"{'='*70}", flush=True)

            safe_cleanup()

            row = {
                "EmbedDim": dim, "Dropout": drop, "SeqLen": seq,
                "Test GAUC": None, "Test NDCG@5": None, "Test MRR@5": None,
                "Test Hit@5": None, "Test mprec@5": None, "Test F1@5": None,
                "Val GAUC": None, "Val NDCG@5": None, "Val MRR@5": None,
                "Status": "FAILED", "ErrorMsg": ""
            }

            try:
                (train_X, train_y, val_X, val_y, test_X, test_y,
                 lbe_map, g2idx, d2idx, rag_init_matrix) = load_combo_data(
                    seq_len=seq, embed_dim=dim, device=self.device)

                val_gauc, val_met, tmet = train_one_combo(
                    train_X, train_y, val_X, val_y, test_X, test_y,
                    lbe_map, g2idx, d2idx, rag_init_matrix,
                    embed_dim=dim, dnn_dropout=drop, seq_len=seq,
                    device=self.device, batch_size=self.batch_size,
                    epochs=self.epochs, patience=self.patience)

                row.update({
                    "Test GAUC":    round(tmet["GAUC"],      6),
                    "Test NDCG@5":  round(tmet["NDCG@5"],    6),
                    "Test MRR@5":   round(tmet["MRR@5"],     6),
                    "Test Hit@5":   round(tmet["Hit@5"],     6),
                    "Test mprec@5": round(tmet["mprec@5"],   6),
                    "Test F1@5":    round(tmet["F1@5"],      6),
                    "Val GAUC":     round(val_gauc,           6),
                    "Val NDCG@5":   round(val_met["NDCG@5"],  6),
                    "Val MRR@5":    round(val_met["MRR@5"],   6),
                    "Status":       "OK",
                    "ErrorMsg":     ""
                })

                print(f"   ✅ Test GAUC={tmet['GAUC']:.4f} | "
                      f"NDCG@5={tmet['NDCG@5']:.4f} | "
                      f"Hit@5={tmet['Hit@5']:.4f}", flush=True)

                del train_X, train_y, val_X, val_y, test_X, test_y
                del lbe_map, g2idx, d2idx, rag_init_matrix

            except torch.cuda.OutOfMemoryError as oom_e:
                print(f"   💥 OOM! {oom_e}", flush=True)
                row["Status"]   = "OOM"
                row["ErrorMsg"] = "CUDA OutOfMemory"
                safe_cleanup()

            except RuntimeError as rt_e:
                if "out of memory" in str(rt_e).lower():
                    print(f"   💥 OOM (Runtime)! {rt_e}", flush=True)
                    row["Status"]   = "OOM"
                    row["ErrorMsg"] = str(rt_e)[:200]
                else:
                    print(f"   ❌ RuntimeError: {rt_e}", flush=True)
                    row["Status"]   = "ERROR"
                    row["ErrorMsg"] = str(rt_e)[:200]
                safe_cleanup()

            except Exception as exc:
                print(f"   ❌ Exception: {exc}", flush=True)
                traceback.print_exc()
                row["Status"]   = "ERROR"
                row["ErrorMsg"] = str(exc)[:200]
                safe_cleanup()

            finally:
                safe_cleanup()
                self.append_row(row)

        print(f"\n{'='*70}")
        print(f"🎉 网格搜索完成！共 {self.completed}/{total} 组")
        print(f"   结果保存在 {self.csv_path}")
        print(f"{'='*70}", flush=True)


# ==========================================
# 6. 主入口
# ==========================================
if __name__ == "__main__":
    runner = GridSearchRunner()
    runner.run()