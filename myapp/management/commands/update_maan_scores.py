"""
MAAN 离线分数增量更新命令
=================================================
当用户产生新评分时，调用 MAAN 模型对该用户做在线推理，
更新 Rec 表中的预测分数。

使用方式：
  python manage.py update_maan_scores --user-id=1
  python manage.py update_maan_scores --all  # 全量更新
"""

import os
import pickle
import logging
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger('movie_agent')


class Command(BaseCommand):
    help = 'MAAN 离线分数增量更新'

    def add_arguments(self, parser):
        parser.add_argument('--user-id', type=int, help='指定用户ID')
        parser.add_argument('--all', action='store_true', help='全量更新所有活跃用户')

    def handle(self, *args, **options):
        import torch
        from myapp.models import UserInfo, UserRating, Movie, Rec

        user_id = options.get('user_id')
        update_all = options.get('all')

        if not user_id and not update_all:
            self.stdout.write(self.style.ERROR('请指定 --user-id 或 --all'))
            return

        # 加载模型
        self.stdout.write('加载 MAAN 模型...')
        model, meta = self._load_model()
        if model is None:
            self.stdout.write(self.style.ERROR('模型加载失败'))
            return

        lbe_user = meta['lbe_user']
        lbe_movie = meta['lbe_movie']
        feature_store = meta['feature_store']
        SEQ_LEN = meta['SEQ_LEN']
        DIM = meta['UNIFIED_EMBED_DIM']

        all_raw_mids = feature_store['raw_movie_ids']
        all_enc_mids = feature_store['enc_movie_ids']
        genres_matrix = feature_store.get('genres_matrix')
        actors_matrix = feature_store.get('actors_matrix')
        directors_matrix = feature_store.get('directors_matrix')
        rag_matrix = feature_store['rag_matrix']

        # 确定目标用户
        if update_all:
            users = UserInfo.objects.filter(is_active=True)
        else:
            users = UserInfo.objects.filter(id=user_id)

        self.stdout.write(f'待更新用户数: {users.count()}')

        device = next(model.parameters()).device

        for user in users:
            # 获取用户历史
            history_raw = list(
                UserRating.objects.filter(user=user)
                .order_by('comment_time')
                .values_list('movie_id', flat=True)
            )
            hist_enc = [
                lbe_movie.transform([str(m)])[0] + 1
                for m in history_raw if str(m) in lbe_movie.classes_
            ]

            if len(hist_enc) == 0:
                hist_padded = np.zeros(SEQ_LEN, dtype=np.int32)
            else:
                hist_padded = np.pad(
                    hist_enc[-SEQ_LEN:],
                    (0, max(0, SEQ_LEN - len(hist_enc))),
                    'constant'
                ) if len(hist_enc) < SEQ_LEN else np.array(hist_enc[-SEQ_LEN:], dtype=np.int32)

            # 排除已评分电影
            user_history_set = set(str(x) for x in history_raw)
            valid_mask = np.array([str(m) not in user_history_set for m in all_raw_mids])

            cand_raw_mids = all_raw_mids[valid_mask]
            cand_enc_mids = all_enc_mids[valid_mask]
            N_cand = len(cand_enc_mids)

            if N_cand == 0:
                self.stdout.write(f'  用户 {user.id}: 无候选电影，跳过')
                continue

            # 构建推理输入
            u_str = str(user.id)
            u_idx = lbe_user.transform([u_str])[0] + 1 if u_str in lbe_user.classes_ else 0

            infer_input = {
                'user_id': np.full(N_cand, u_idx, dtype=np.int32),
                'movie_id': cand_enc_mids,
                'hist_movie_id': np.tile(hist_padded, (N_cand, 1)),
                'sl': np.full(N_cand, min(len(hist_enc), SEQ_LEN), dtype=np.int32)
            }

            if genres_matrix is not None:
                infer_input['genres'] = genres_matrix[valid_mask]
                infer_input['actors'] = actors_matrix[valid_mask]
            if directors_matrix is not None:
                infer_input['directors'] = directors_matrix[valid_mask]

            rag_b = rag_matrix[valid_mask]
            for i in range(DIM):
                infer_input[f'rag_{i}'] = rag_b[:, i]

            # 推理
            with torch.no_grad():
                preds = model.predict(infer_input, batch_size=2048).flatten()

            # 取 Top-100 写入 Rec 表
            top_100_idx = preds.argsort()[::-1][:100]
            top_100_mids = cand_raw_mids[top_100_idx]
            top_100_scores = preds[top_100_idx]

            Rec.objects.filter(user=user).delete()
            rec_objs = [
                Rec(user=user, movie_id=int(mid), rating=float(score))
                for mid, score in zip(top_100_mids, top_100_scores)
            ]
            Rec.objects.bulk_create(rec_objs)

            self.stdout.write(f'  用户 {user.id}: 更新 {len(rec_objs)} 条推荐记录')

        self.stdout.write(self.style.SUCCESS('MAAN 分数增量更新完成'))

    def _load_model(self):
        """加载 MAAN 模型（复用 MAANRerankTool 的加载逻辑）"""
        try:
            import torch
            from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
            from myapp.mman_model import MMAN
            from myapp.skb_model import SKB_FMLP_Online

            artifacts_dir = os.path.join(settings.BASE_DIR, 'ml_artifacts')
            model_path = os.path.join(artifacts_dir, 'skb_fmlp_online.pt')
            meta_path = os.path.join(artifacts_dir, 'online_features_meta.pkl')

            if not os.path.exists(model_path):
                return None, None

            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            state_dict = torch.load(model_path, map_location=device, weights_only=True)

            lbe_user = meta['lbe_user']
            lbe_movie = meta['lbe_movie']
            DIM = meta['UNIFIED_EMBED_DIM']
            SEQ = meta['SEQ_LEN']

            vocab_user = len(lbe_user.classes_) + 1
            vocab_movie = len(lbe_movie.classes_) + 1
            vocab_genre = state_dict['embedding_dict.genres.weight'].shape[0]
            vocab_actor = state_dict['embedding_dict.actors.weight'].shape[0]
            vocab_director = state_dict['embedding_dict.directors.weight'].shape[0]

            user_col = SparseFeat('user_id', vocab_user, DIM, embedding_name='user_id')
            movie_col = SparseFeat('movie_id', vocab_movie, DIM, embedding_name='movie_id')
            genre_col = VarLenSparseFeat(SparseFeat('genres', vocab_genre, DIM), maxlen=5, combiner='mean')
            actor_col = VarLenSparseFeat(SparseFeat('actors', vocab_actor, DIM), maxlen=5, combiner='mean')
            director_col = VarLenSparseFeat(SparseFeat('directors', vocab_director, DIM), maxlen=3, combiner='mean')

            rag_cols = [DenseFeat(f'rag_{i}', 1) for i in range(DIM)]
            seq_col = VarLenSparseFeat(
                SparseFeat('hist_movie_id', vocab_movie, DIM, embedding_name='movie_id'),
                maxlen=SEQ, length_name='sl', combiner='mean')

            linear_cols = [movie_col] + rag_cols
            dnn_cols = [user_col, movie_col, genre_col, actor_col, director_col, seq_col] + rag_cols

            model_type = meta.get('model_type', 'skb_fmlp')
            TEXT_DIM = meta.get('TEXT_DIM', 64)
            VISUAL_DIM = meta.get('VISUAL_DIM', 64)
            DROPOUT = meta.get('FIXED_DROPOUT', 0.1)

            if model_type == 'mman':
                model = MMAN(
                    linear_cols, dnn_cols,
                    history_feature_list=['movie_id'],
                    text_dim=TEXT_DIM, visual_dim=VISUAL_DIM,
                    hidden_dim=256, num_heads=4, dropout=DROPOUT, device=device
                )
            else:
                model = SKB_FMLP_Online(
                    linear_cols, dnn_cols,
                    history_feature_list=['movie_id'],
                    text_dim=TEXT_DIM, visual_dim=VISUAL_DIM,
                    hidden_dim=256, num_heads=4, dropout=DROPOUT, device=device
                )

            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)

            self.stdout.write(f'模型加载成功: {model_type}, device={device}')
            return model, meta

        except Exception as e:
            logger.error(f'MAAN 模型加载失败: {e}')
            return None, None
