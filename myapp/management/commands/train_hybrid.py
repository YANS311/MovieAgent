import os
# 🔥 在 CUDA 初始化前设置（防止 CUDA unknown error）
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 禁用 TensorRT，防止兼容性问题
os.environ['TORCH_DISABLE_EXTENSION_IMPORT_ERROR'] = '1'

import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import PCA
from django.core.management.base import BaseCommand
import django

# 设置环境变量与 Django 引擎
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoProject3.settings')
django.setup()

from myapp.models import UserRating, Movie, UserInfo
from sentence_transformers import SentenceTransformer

# DeepCTR-Torch 核心组件
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN

# 导入模型：MMAN (多模态注意力网络) 替代 SKB-FMLP
from myapp.mman_model import MMAN
from myapp.skb_model import SKB_FMLP_Online  # 保留兼容

# ==========================================
# 0. SOTA 参数配置 (MMAN 实验配置)
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UNIFIED_EMBED_DIM = 128   # PCA 统一维度 = 128
TEXT_DIM = 64             # 文本 PCA 维度
VISUAL_DIM = 64           # 视觉特征维度 (= UNIFIED_EMBED_DIM - TEXT_DIM)
SEQ_LEN = 10
FIXED_DROPOUT = 0.1       # MMAN dropout = 0.1
BATCH_SIZE = 256          # 🔥 从1024降到256，避免 CUDA OOM（Django warmup 已占用部分显存）
EPOCHS = 10
USE_MMAN = True           # 🔥 启用 MMAN 模型 (设为 False 则回退到 SKB-FMLP)


def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 🔥 强制重新初始化CUDA，防止未知错误
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def numpy_pad_sequences(sequences, maxlen, padding='post', value=0):
    out = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if not isinstance(seq, (list, np.ndarray)) or len(seq) == 0: continue
        trunc = seq[:maxlen] if padding == 'post' else seq[-maxlen:]
        out[i, :len(trunc)] = trunc if padding == 'post' else trunc
    return out


class Command(BaseCommand):
    help = '全流程：训练 KAG 混合推荐模型(含导演节点) -> 保存模型权重与特征矩阵'

    def handle(self, *args, **options):
        self.stdout.write("--- 🚀 启动 [KAG 完整版: 类型+演员+导演] 离线训练任务 ---")
        seed_everything()

        # =========================================
        # 1. 从 Django 数据库拉取全量数据
        # =========================================
        self.stdout.write("[1/5] 正在从数据库加载数据...")
        ratings_qs = UserRating.objects.all().values('user_id', 'movie_id', 'score', 'comment_time')
        df_ratings = pd.DataFrame.from_records(ratings_qs)
        if df_ratings.empty:
            self.stdout.write(self.style.ERROR("数据为空，请先填充数据库！"))
            return

        users_qs = UserInfo.objects.all().values('id', 'username', 'age', 'occupation', 'sex')
        df_users = pd.DataFrame.from_records(users_qs)
        df_users.rename(columns={'id': 'user_id'}, inplace=True)
        # 填充缺失值：age 用中位数，occupation 用 0（其他），sex 用 1（男）
        df_users['age'] = df_users['age'].fillna(df_users['age'].median() if not df_users['age'].isna().all() else 25).astype(int)
        df_users['occupation'] = df_users['occupation'].fillna(0).astype(int)
        df_users['sex'] = df_users['sex'].fillna(1).astype(int)

        # 🔥 新增 directors 预加载
        # ★ 排除敏感内容：训练阶段直接过滤 is_sensitive=True 的电影，
        #   避免敏感内容进入推荐池，从源头杜绝上线后的审查风险
        self.stdout.write("      -> 正在提取 Movie 及其关联实体(KG)...")
        excluded_sensitive = Movie.objects.filter(is_sensitive=True).count()
        if excluded_sensitive > 0:
            self.stdout.write(f"      ⚠️ 排除 {excluded_sensitive} 部敏感内容电影（is_sensitive=True）")
        movies_qs = Movie.objects.filter(is_sensitive=False).prefetch_related('genres', 'actors', 'directors')

        movie_data = []
        movie_genres_dict, movie_actors_dict, movie_directors_dict = {}, {}, {}
        all_genres, all_actors, all_directors = set(), set(), set()

        for m in movies_qs:
            g_names = [g.name for g in m.genres.all()]
            a_names = [a.name for a in m.actors.all()[:5]]
            d_names = [d.name for d in m.directors.all()[:3]]  # 🔥 提取前3位导演

            movie_data.append({
                'id': str(m.id),
                'title': m.title,
                'summary': m.summary,
                'poster_embedding_json': m.poster_embedding_json
            })
            movie_genres_dict[str(m.id)] = g_names
            movie_actors_dict[str(m.id)] = a_names
            movie_directors_dict[str(m.id)] = d_names  # 🔥 存入字典

            all_genres.update(g_names)
            all_actors.update(a_names)
            all_directors.update(d_names)  # 🔥 收集全部导演

        df_movies = pd.DataFrame(movie_data)

        # =========================================
        # 2. 特征工程 (KG + RAG + 多模态)
        # =========================================
        self.stdout.write("[2/5] 启动 KAG 引擎：提取图谱结构与多模态视觉...")

        lbe_user = LabelEncoder().fit(df_users['user_id'].astype(str))
        lbe_movie = LabelEncoder().fit(df_movies['id'].astype(str))

        df_ratings['enc_u'] = lbe_user.transform(df_ratings['user_id'].astype(str)) + 1
        df_ratings['enc_m'] = lbe_movie.transform(df_ratings['movie_id'].astype(str)) + 1

        vocab_user = len(lbe_user.classes_) + 1
        vocab_movie = len(lbe_movie.classes_) + 1

        # -----------------------------------
        # [A] KG 结构化图谱特征处理 (含导演)
        # -----------------------------------
        self.stdout.write("      -> 正在构建 KG 图谱矩阵 (加入导演节点)...")
        genre2idx = {g: i + 1 for i, g in enumerate(all_genres)}
        actor2idx = {a: i + 1 for i, a in enumerate(all_actors)}
        director2idx = {d: i + 1 for i, d in enumerate(all_directors)}  # 🔥 导演编码

        vocab_genre, vocab_actor, vocab_director = len(genre2idx) + 1, len(actor2idx) + 1, len(director2idx) + 1
        MAX_GENRES, MAX_ACTORS, MAX_DIRECTORS = 5, 5, 3

        genres_matrix = np.zeros((vocab_movie, MAX_GENRES), dtype=np.int32)
        actors_matrix = np.zeros((vocab_movie, MAX_ACTORS), dtype=np.int32)
        directors_matrix = np.zeros((vocab_movie, MAX_DIRECTORS), dtype=np.int32)  # 🔥 导演矩阵

        for raw_mid in lbe_movie.classes_:
            enc_m = lbe_movie.transform([raw_mid])[0] + 1
            g_list = [genre2idx[g] for g in movie_genres_dict.get(raw_mid, [])]
            a_list = [actor2idx[a] for a in movie_actors_dict.get(raw_mid, [])]
            d_list = [director2idx[d] for d in movie_directors_dict.get(raw_mid, [])]  # 🔥

            genres_matrix[enc_m, :min(len(g_list), MAX_GENRES)] = g_list[:MAX_GENRES]
            actors_matrix[enc_m, :min(len(a_list), MAX_ACTORS)] = a_list[:MAX_ACTORS]
            directors_matrix[enc_m, :min(len(d_list), MAX_DIRECTORS)] = d_list[:MAX_DIRECTORS]  # 🔥

        # -----------------------------------
        # [B] RAG 文本与多模态视觉融合
        # -----------------------------------
        # TEXT_DIM 和 VISUAL_DIM 已在顶部配置
        assert TEXT_DIM + VISUAL_DIM == UNIFIED_EMBED_DIM, "TEXT_DIM + VISUAL_DIM 必须等于 UNIFIED_EMBED_DIM"

        self.stdout.write("      -> 正在抽取文本与视觉联合特征...")
        # 🔥 修复 CUDA OOM：强制使用 CPU 编码文本（离线训练不需要 GPU 编码器）
        # 原因：GPU 上已有其他模型占用显存（Django warmup），SentenceTransformer 再加载会导致 OOM
        encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        df_movies['rag_text'] = df_movies['title'].fillna('') + " " + df_movies['summary'].fillna('')
        text_dict = dict(zip(df_movies['id'].astype(str), df_movies['rag_text']))
        ordered_texts = [text_dict.get(mid, "Movie") for mid in lbe_movie.classes_]

        # 🔥 修复：降低 batch_size 避免 CPU 内存压力过大（6500+电影分批编码）
        text_raw = encoder.encode(ordered_texts, batch_size=64, show_progress_bar=False)
        text_pca = PCA(n_components=TEXT_DIM).fit_transform(text_raw)
        text_pca = normalize(text_pca, norm='l2', axis=1)

        visual_dict = dict(zip(df_movies['id'].astype(str), df_movies['poster_embedding_json']))
        visual_features = []
        for mid in lbe_movie.classes_:
            vec = visual_dict.get(mid)
            if vec and isinstance(vec, list) and len(vec) > 0:
                v_arr = np.array(vec)[:VISUAL_DIM]
                v_arr = np.pad(v_arr, (0, max(0, VISUAL_DIM - len(v_arr))))
            else:
                v_arr = np.zeros(VISUAL_DIM)
            visual_features.append(v_arr)

        visual_matrix = normalize(np.array(visual_features), norm='l2', axis=1)
        multimodal_pca = np.concatenate([text_pca, visual_matrix], axis=1)
        rag_matrix = np.zeros((vocab_movie, UNIFIED_EMBED_DIM))
        rag_matrix[1:] = multimodal_pca

        # =========================================
        # 3. 构造滑动窗口训练集
        # =========================================
        self.stdout.write("[3/5] 构建时序训练集...")
        df_ratings = df_ratings.sort_values(['enc_u', 'comment_time'])

        train_l = []
        all_items = list(range(1, vocab_movie))

        for uid, group in tqdm(df_ratings.groupby('enc_u')):
            items = group['enc_m'].tolist()
            if len(items) < 3: continue

            cur_watched = set()
            for i in range(len(items)):
                cur_target = items[i]
                hist = items[max(0, i - SEQ_LEN):i]
                cur_watched.add(cur_target)

                if i < 1: continue

                train_l.append({'user_id': uid, 'movie_id': cur_target, 'hist': hist, 'label': 1})
                while True:
                    neg = random.choice(all_items)
                    if neg not in cur_watched:
                        train_l.append({'user_id': uid, 'movie_id': neg, 'hist': hist, 'label': 0})
                        break

        df_train = pd.DataFrame(train_l)

        # 🔥 新增：合并人口统计学特征到训练集
        user_demo = df_users[['user_id', 'age', 'occupation', 'sex']].drop_duplicates('user_id')
        df_train = df_train.merge(user_demo, on='user_id', how='left')
        df_train['age'] = df_train['age'].fillna(25).astype(int)
        df_train['occupation'] = df_train['occupation'].fillna(0).astype(int)
        df_train['sex'] = df_train['sex'].fillna(1).astype(int)
        # 归一化年龄到 [0, 1]
        age_max = df_train['age'].max() if df_train['age'].max() > 0 else 1
        df_train['age_norm'] = df_train['age'].astype(float) / age_max

        # 🔥 新增：构建 DeepCTR-Torch 结构 (包含导演)
        user_col = SparseFeat('user_id', vocab_user, UNIFIED_EMBED_DIM, embedding_name='user_id')
        movie_col = SparseFeat('movie_id', vocab_movie, UNIFIED_EMBED_DIM, embedding_name='movie_id')

        genre_col = VarLenSparseFeat(SparseFeat('genres', vocab_genre, UNIFIED_EMBED_DIM), maxlen=MAX_GENRES,
                                     combiner='mean')
        actor_col = VarLenSparseFeat(SparseFeat('actors', vocab_actor, UNIFIED_EMBED_DIM), maxlen=MAX_ACTORS,
                                     combiner='mean')
        director_col = VarLenSparseFeat(SparseFeat('directors', vocab_director, UNIFIED_EMBED_DIM),
                                        maxlen=MAX_DIRECTORS, combiner='mean')  # 🔥 导演列

        rag_col = [DenseFeat(f'rag_{i}', 1) for i in range(UNIFIED_EMBED_DIM)]
        seq_col = VarLenSparseFeat(
            SparseFeat('hist_movie_id', vocab_movie, UNIFIED_EMBED_DIM, embedding_name='movie_id'), maxlen=SEQ_LEN,
            length_name='sl', combiner='mean')

        # 🔥 新增：人口统计学特征
        occupation_col = SparseFeat('occupation', 21, 8)   # 0-20 共 21 类，嵌入维度 8
        sex_col = SparseFeat('sex', 3, 4)                  # 1=男, 2=女，嵌入维度 4
        age_col = DenseFeat('age_norm', 1)                 # 归一化年龄，连续特征

        linear_cols = [movie_col] + rag_col
        dnn_cols = [user_col, movie_col, genre_col, actor_col, director_col, seq_col,
                    occupation_col, sex_col] + rag_col + [age_col]  # 🔥 人口统计学特征加入主干，age 放在 rag 之后

        x_train = {
            'user_id': df_train['user_id'].values,
            'movie_id': df_train['movie_id'].values,
            'genres': genres_matrix[df_train['movie_id'].values],
            'actors': actors_matrix[df_train['movie_id'].values],
            'directors': directors_matrix[df_train['movie_id'].values],  # 🔥 送入训练数据
            'hist_movie_id': numpy_pad_sequences(df_train['hist'].tolist(), maxlen=SEQ_LEN),
            'sl': np.array([len(h) for h in df_train['hist']], dtype=np.int32),
            # 🔥 新增：人口统计学特征
            'occupation': df_train['occupation'].values,
            'sex': df_train['sex'].values,
        }
        rag_b = rag_matrix[df_train['movie_id'].values]
        for i in range(UNIFIED_EMBED_DIM):
            x_train[f'rag_{i}'] = rag_b[:, i]
        # 🔥 新增：归一化年龄作为连续特征
        x_train['age_norm'] = df_train['age_norm'].values.astype(np.float32)

        y_train = df_train['label'].values

        # =========================================
        # 4. 训练模型 (MMAN 或 SKB-FMLP)
        # =========================================
        if USE_MMAN:
            self.stdout.write("[4/5] 训练 MMAN 模型 (多模态注意力网络)...")
            self.stdout.write(f"      配置: dropout={FIXED_DROPOUT}, PCA={UNIFIED_EMBED_DIM}, "
                              f"TEXT={TEXT_DIM}, VISUAL={VISUAL_DIM}, demographic=True")
            model = MMAN(
                linear_cols, dnn_cols,
                history_feature_list=['movie_id'],
                text_dim=TEXT_DIM,
                visual_dim=VISUAL_DIM,
                hidden_dim=256,
                num_heads=4,
                dropout=FIXED_DROPOUT,
                use_demographic=True,
                device=DEVICE
            )
        else:
            self.stdout.write("[4/5] 训练 SKB-FMLP 模型...")
            model = SKB_FMLP_Online(linear_cols, dnn_cols, history_feature_list=['movie_id'], device=DEVICE)

        # 🔥 用多模态 RAG 向量初始化 movie_id 嵌入层 (Text+Visual 联合表示)
        if 'movie_id' in model.embedding_dict:
            model.embedding_dict['movie_id'].weight.data.copy_(torch.FloatTensor(rag_matrix).to(DEVICE))
            model.embedding_dict['movie_id'].weight.requires_grad = False

        model.compile("adam", "binary_crossentropy", metrics=["auc"])
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

        # =========================================
        # 5. 导出工程产物
        # =========================================
        # 保存模型类型标记，供线上推理区分
        model_type = 'mman' if USE_MMAN else 'skb_fmlp'
        self.stdout.write(f"[5/5] 保存特征库与权重至 ml_artifacts... (模型类型: {model_type})")
        ARTIFACTS_DIR = os.path.join(django.conf.settings.BASE_DIR, 'ml_artifacts')
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # 保存模型权重（MMAN 和 SKB-FMLP 共用同一路径，但需要标记类型）
        model_path = os.path.join(ARTIFACTS_DIR, 'skb_fmlp_online.pt')
        torch.save(model.state_dict(), model_path)

        # 如果是 MMAN，额外保存一份 MMAN 专用权重
        if USE_MMAN:
            mman_path = os.path.join(ARTIFACTS_DIR, 'mman_online.pt')
            torch.save(model.state_dict(), mman_path)
            self.stdout.write(f"   ✅ MMAN 专用权重已保存至: {mman_path}")

        # 🔥 将导演特征矩阵也丢进 Feature Store，供线上重排切片使用
        feature_store = {
            'enc_movie_ids': np.arange(1, vocab_movie),
            'raw_movie_ids': lbe_movie.inverse_transform(np.arange(1, vocab_movie) - 1),
            'rag_matrix': rag_matrix[1:],
            'genres_matrix': genres_matrix[1:],
            'actors_matrix': actors_matrix[1:],
            'directors_matrix': directors_matrix[1:]  # 🔥 存入字典
        }

        meta_dict = {
            'lbe_user': lbe_user,
            'lbe_movie': lbe_movie,
            'feature_store': feature_store,
            'SEQ_LEN': SEQ_LEN,
            'UNIFIED_EMBED_DIM': UNIFIED_EMBED_DIM,
            'TEXT_DIM': TEXT_DIM,
            'VISUAL_DIM': VISUAL_DIM,
            'FIXED_DROPOUT': FIXED_DROPOUT,
            'model_type': model_type,  # 🔥 新增：标记模型类型
        }

        meta_path = os.path.join(ARTIFACTS_DIR, 'online_features_meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_dict, f)

        self.stdout.write(self.style.SUCCESS(f"✅ 模型权重已保存至: {model_path}"))
        self.stdout.write(self.style.SUCCESS(f"✅ 在线特征元数据已保存至: {meta_path}"))