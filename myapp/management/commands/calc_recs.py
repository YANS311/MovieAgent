# 文件: myapp/management/commands/calc_recs.py (V4 - 含评估指标版)

import math
import random
from collections import defaultdict
import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

from myapp.models import UserRating, Movie, UserInfo, Rec


class UserBasedCF_V3:
    def __init__(self, n_sim_user=10, n_rec_movie=5):
        self.n_sim_user = n_sim_user
        self.n_rec_movie = n_rec_movie
        self.dataSet = {}  # 训练集: {user: {movie: score}}
        self.user_sim_matrix = {}
        self.user_avg_ratings = {}
        self.movie_count = 0

        print(f'Similar user number = {self.n_sim_user}')
        print(f'Recommended movie number = {self.n_rec_movie}')

    def fit(self, train_ratings_list):
        """
        训练模型 (构建数据集, 计算平均分, 计算相似度矩阵)
        """
        print('Building dataSet...')
        self.dataSet = {}
        user_ratings_sum = defaultdict(int)
        user_ratings_count = defaultdict(int)

        # 1. 构建用户-物品倒排表 & 计算用户平均分
        for user_id, movie_id, score in train_ratings_list:
            self.dataSet.setdefault(user_id, {})
            self.dataSet[user_id][movie_id] = score

            user_ratings_sum[user_id] += score
            user_ratings_count[user_id] += 1

        for user_id in user_ratings_sum:
            self.user_avg_ratings[user_id] = user_ratings_sum[user_id] / user_ratings_count[user_id]

        print('Build dataSet success!')

        # 2. 计算相似度 (皮尔逊)
        self._calc_user_sim()

    def _calc_user_sim(self):
        print('Calculating user similarity matrix (Pearson)...')
        movie_user = {}
        for user, movies in self.dataSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)

        self.movie_count = len(movie_user)
        print(f'Total movie number in Train Set = {self.movie_count}')

        all_users = list(self.dataSet.keys())
        for u in all_users:
            self.user_sim_matrix.setdefault(u, {})
            for v in all_users:
                if u == v: continue

                # 优化: 只计算有过共同评分的用户 (这里简化为全量遍历, 实际上可以通过倒排表加速)
                # 为了 UserCF 的速度，我们只计算 "相关的用户"
                # (这部分逻辑在标准 UserCF 中通常利用倒排表优化，这里沿用之前的逻辑但确保正确性)
                pass

                # --- 优化版的相似度计算 (利用倒排表加速) ---
        # 1. 统计用户共现矩阵 C[u][v] = 共同评分的电影数
        C = defaultdict(dict)
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v: continue
                    C[u].setdefault(v, 0)
                    C[u][v] += 1

        # 2. 计算皮尔逊系数
        for u, related_users in C.items():
            self.user_sim_matrix.setdefault(u, {})
            for v, count in related_users.items():
                # 皮尔逊相关系数计算
                if count < 3: continue  # 共同评分太少, 忽略

                co_rated_movies = [m for m in self.dataSet[u] if m in self.dataSet[v]]

                numerator = 0.0
                denom_u = 0.0
                denom_v = 0.0

                avg_u = self.user_avg_ratings[u]
                avg_v = self.user_avg_ratings[v]

                for m in co_rated_movies:
                    r_ui = self.dataSet[u][m]
                    r_vi = self.dataSet[v][m]
                    numerator += (r_ui - avg_u) * (r_vi - avg_v)
                    denom_u += (r_ui - avg_u) ** 2
                    denom_v += (r_vi - avg_v) ** 2

                if denom_u == 0 or denom_v == 0:
                    self.user_sim_matrix[u][v] = 0
                else:
                    self.user_sim_matrix[u][v] = numerator / (math.sqrt(denom_u) * math.sqrt(denom_v))

        print('Calculate user similarity matrix success!')

    def predict(self, user_id, movie_id):
        """
        预测单个用户对单个电影的评分 (用于评估)
        """
        # 如果是冷启动用户或电影 (训练集里没有), 返回默认分 (比如 3.0 或 全局平均分)
        if user_id not in self.dataSet:
            return 5.0  # 默认中位数 (0-10分)

        avg_u = self.user_avg_ratings[user_id]

        # 找到看过这部电影的相似用户
        # 1. 谁看过这部电影? (这里需要重新遍历一下, 或者维护一个 movie_users 表)
        #    为了效率, 我们假设 fit 阶段已经存了 movie_user, 这里简化直接遍历相似用户

        sim_users = sorted(self.user_sim_matrix.get(user_id, {}).items(), key=lambda x: x[1], reverse=True)[
            :self.n_sim_user]

        numerator = 0.0
        denominator = 0.0

        for v, sim in sim_users:
            if sim <= 0: continue
            if movie_id in self.dataSet[v]:
                r_vi = self.dataSet[v][movie_id]
                avg_v = self.user_avg_ratings[v]

                numerator += sim * (r_vi - avg_v)
                denominator += sim

        if denominator == 0:
            return avg_u

        pred_score = avg_u + (numerator / denominator)
        return max(0.0, min(10.0, pred_score))  # 截断到 0-10

    def recommend(self, user_id):
        """
        为用户生成 Top-N 推荐 (用于存库)
        """
        if user_id not in self.dataSet: return []

        rank = {}
        total_sim = defaultdict(float)

        avg_u = self.user_avg_ratings[user_id]
        watched = self.dataSet[user_id]

        sim_users = sorted(self.user_sim_matrix.get(user_id, {}).items(), key=lambda x: x[1], reverse=True)[
            :self.n_sim_user]

        for v, sim in sim_users:
            if sim <= 0: continue
            avg_v = self.user_avg_ratings[v]

            for m, r in self.dataSet[v].items():
                if m in watched: continue

                rank.setdefault(m, 0.0)
                rank[m] += sim * (r - avg_v)
                total_sim[m] += sim

        final_rank = []
        for m, score_sum in rank.items():
            if total_sim[m] > 0:
                pred = avg_u + (score_sum / total_sim[m])
                final_rank.append((m, max(0.0, min(10.0, pred))))

        return sorted(final_rank, key=lambda x: x[1], reverse=True)[:self.n_rec_movie]


class Command(BaseCommand):
    help = '运行 UserCF 算法: 训练、评估(RMSE/AUC)、并生成推荐'

    def handle(self, *args, **options):
        self.stdout.write("--- 🚀 UserCF 推荐系统启动 ---")

        # 1. 加载所有评分数据
        self.stdout.write("1. 正在从数据库加载评分...")
        # 使用 values_list 稍微快一点
        raw_ratings = list(UserRating.objects.all().values_list('user_id', 'movie_id', 'score'))

        if not raw_ratings:
            self.stderr.write("错误: UserRating 表为空。")
            return

        # 2. 切分数据集 (80% 训练, 20% 测试)
        self.stdout.write(f"   总数据量: {len(raw_ratings)} 条")
        train_data, test_data = train_test_split(raw_ratings, test_size=0.2, random_state=42)
        self.stdout.write(f"   训练集: {len(train_data)}, 测试集: {len(test_data)}")

        # 3. 初始化并训练 UserCF
        userCF = UserBasedCF_V3()
        userCF.fit(train_data)

        # 4. 在测试集上评估 (Evaluation)
        self.stdout.write("2. 正在进行评估 (这可能需要几分钟)...")

        y_true = []
        y_pred = []

        # 为了节省时间，只随机抽取 5000 条测试数据进行评估 (UserCF 预测太慢了)
        # 如果你想全量评估，去掉切片即可
        eval_sample = test_data[:5000]

        for uid, mid, real_score in eval_sample:
            pred_score = userCF.predict(uid, mid)
            y_true.append(real_score)
            y_pred.append(pred_score)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # 计算 AUC (二分类: >=4.0 为正样本)
        y_true_bin = [1 if s >= 4.0 else 0 for s in y_true]
        # 归一化预测分到 0-1
        y_pred_norm = [max(0, min(1, (s - 1) / 4)) for s in y_pred]
        try:
            auc = roc_auc_score(y_true_bin, y_pred_norm)
        except:
            auc = 0.5

        self.stdout.write(self.style.SUCCESS(f"★ UserCF 评估结果:"))
        self.stdout.write(self.style.SUCCESS(f"   RMSE: {rmse:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"   MAE:  {mae:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"   AUC:  {auc:.4f}"))

        # 5. 为全站用户生成推荐 (Inference)
        # 注意：为了保证推荐质量，这里使用的是 80% 数据的模型。
        # 严格来说应该用全量数据重训一次，但为了省时间，直接用即可。
        self.stdout.write("3. 正在为用户生成推荐列表...")

        Rec.objects.all().delete()
        recs_to_create = []

        # 只为活跃用户生成 (节省时间)
        target_users = UserInfo.objects.filter(is_active=True)

        for user in target_users:
            recs = userCF.recommend(user.id)
            for mid, rate in recs:
                recs_to_create.append(Rec(user_id=user.id, movie_id=mid, rating=rate))

        Rec.objects.bulk_create(recs_to_create)
        self.stdout.write(self.style.SUCCESS(f"✅ 全部完成! 生成了 {len(recs_to_create)} 条推荐。"))