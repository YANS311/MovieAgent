#!/usr/bin/env python
"""模态级消融专用脚本：仅跑 MAAN Full / w/o Visual / w/o KG 三项，快速验证"""
import os, sys, django, gc, copy, random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')
django.setup()

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
# 直接从 run_local_ablation 导入所有模型类
from run_local_ablation import (
    BiCrossAttFusion, SKB_FMLP_Online, DirectFusion,
    GatedFusion, CrossAttFusion, numpy_pad_sequences, get_rank_metrics, load_local_data
)

UNIFIED_EMBED_DIM  = 128
MULTIMODAL_DIM     = 16
SEQ_LEN            = 10
BATCH_SIZE         = 2048
EPOCHS             = 15
PATIENCE           = 3
SEED               = 2026
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

# ── 数据加载（复用 run_local_ablation 的函数） ──
train_df, val_df, test_df, lbe_map, pad_g, pad_d, rag_m, vis_m, rag_init_matrix, g2idx, d2idx = \
    load_local_data(DEVICE, SEQ_LEN, MULTIMODAL_DIM, UNIFIED_EMBED_DIM)

linear_cols = [
    SparseFeat('user_id',  len(lbe_map['user_id'].classes_)   + 1, UNIFIED_EMBED_DIM),
    SparseFeat('movie_id', len(lbe_map['movie_id'].classes_)  + 1, UNIFIED_EMBED_DIM),
]
base_dnn_cols = linear_cols + [
    VarLenSparseFeat(SparseFeat('hist_movie_id', len(lbe_map['movie_id'].classes_) + 1,
                                UNIFIED_EMBED_DIM, embedding_name='movie_id'),
                     maxlen=SEQ_LEN, combiner='mean', length_name='seq_len')
]
kg_cols = [
    VarLenSparseFeat(SparseFeat('genres',    len(g2idx) + 1, UNIFIED_EMBED_DIM), maxlen=3, combiner='mean'),
    VarLenSparseFeat(SparseFeat('directors', len(d2idx) + 1, UNIFIED_EMBED_DIM), maxlen=2, combiner='mean'),
]
rag_cols = [DenseFeat(f'rag_{i}', 1) for i in range(MULTIMODAL_DIM)]
vis_cols = [DenseFeat(f'vis_{i}', 1) for i in range(MULTIMODAL_DIM)]
full_cols = base_dnn_cols + kg_cols + rag_cols + vis_cols

def get_input(df):
    x = {'user_id':      df['enc_u'].values.astype(np.int32),
         'movie_id':     df['enc_m'].values.astype(np.int32)}
    x['hist_movie_id'] = numpy_pad_sequences(df['hist'].tolist(), SEQ_LEN)
    x['seq_len']       = np.array([len(h) for h in df['hist']], dtype=np.int32)
    mids = df['enc_m'].values.astype(int)
    x['genres'], x['directors'] = pad_g[mids], pad_d[mids]
    for i in range(MULTIMODAL_DIM):
        x[f'rag_{i}'] = rag_m[mids, i].astype(np.float32).reshape(-1, 1)
        x[f'vis_{i}'] = vis_m[mids, i].astype(np.float32).reshape(-1, 1)
    return x, df['label'].values

train_X, train_y = get_input(train_df)
val_X,   val_y   = get_input(val_df)
test_X,  test_y  = get_input(test_df)

print(f"\n{'='*60}")
print(f"🔬 模态消融专用 | Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)} Device:{DEVICE}")
print(f"{'='*60}\n")

results = []

for exp_name, ablate_mode in [
    ("MAAN Full (ref)", None),
    ("w/o Visual",      "visual"),
    ("w/o KG",          "kg"),
]:
    print(f"\n>>> [RUN] {exp_name} ...", flush=True)

    # ── 数据级消融：深拷贝后置零对应模态 ──
    cur_train = {k: v.copy() for k, v in train_X.items()}
    cur_val   = {k: v.copy() for k, v in val_X.items()}
    cur_test  = {k: v.copy() for k, v in test_X.items()}

    if ablate_mode == "visual":
        for i in range(MULTIMODAL_DIM):
            cur_train[f'vis_{i}'] = np.zeros_like(cur_train[f'vis_{i}'])
            cur_val[f'vis_{i}']   = np.zeros_like(cur_val[f'vis_{i}'])
            cur_test[f'vis_{i}']  = np.zeros_like(cur_test[f'vis_{i}'])
        print("   [Ablation] vis_0~15 → 零向量", flush=True)

    elif ablate_mode == "kg":
        cur_train['genres']    = np.zeros_like(cur_train['genres'])
        cur_val['genres']      = np.zeros_like(cur_val['genres'])
        cur_test['genres']     = np.zeros_like(cur_test['genres'])
        cur_train['directors'] = np.zeros_like(cur_train['directors'])
        cur_val['directors']   = np.zeros_like(cur_val['directors'])
        cur_test['directors']  = np.zeros_like(cur_test['directors'])
        print("   [Ablation] genres + directors → 零向量", flush=True)

    model = BiCrossAttFusion(
        linear_cols, full_cols, fuse_dim=64, mlp_hidden_units=(256, 128),
        dnn_dropout=0.1, l2_reg_embedding=1e-4, device=DEVICE
    )
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    # 注入 RAG 语义 Embedding 初始化
    model.embedding_dict['movie_id'].weight.data.copy_(
        torch.FloatTensor(rag_init_matrix).to(DEVICE))
    model.embedding_dict['movie_id'].weight.requires_grad = True

    best_gauc, best_w, wait = 0.0, None, 0
    for ep in range(EPOCHS):
        model.fit(cur_train, train_y, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        val_pred = model.predict(cur_val, batch_size=BATCH_SIZE).flatten()
        met = get_rank_metrics(val_y, val_pred)
        g = met['GAUC']
        print(f"   [Ep {ep+1:2d}/{EPOCHS}] Val GAUC={g:.5f}  NDCG@5={met['NDCG@5']:.5f}", flush=True)
        if g > best_gauc:
            best_gauc = g; best_w = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: print("   ⏹ 早停"); break

    model.load_state_dict(best_w)
    final_pred = model.predict(cur_test, batch_size=BATCH_SIZE).flatten()
    met = get_rank_metrics(test_y, final_pred)
    met['Exp_Name'] = exp_name
    met['Type']     = 'Ablation'
    results.append(met)
    print(f"   🏁 {exp_name}: GAUC={met['GAUC']:.5f}  NDCG@5={met['NDCG@5']:.5f}  "
          f"MRR@5={met['MRR@5']:.5f}  Hit@5={met['Hit@5']:.5f}  F1@5={met['F1@5']:.5f}", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()

# ── 写出 CSV ──
df_out = pd.DataFrame(results)
csv_name = f"thesis_modality_ablation_{datetime.now().strftime('%m%d_%H%M')}.csv"
df_out.to_csv(os.path.join(BASE_DIR, csv_name), index=False)
print(f"\n✅ 模态消融完成 → {csv_name}")
print(df_out[['Exp_Name', 'GAUC', 'NDCG@5', 'MRR@5', 'Hit@5', 'F1@5']].to_string(index=False))