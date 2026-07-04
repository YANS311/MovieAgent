"""
FAISS 视觉向量加速检索模块
================================================
替代原 search_visual 中的 O(N) numpy 全量点乘，
使用 FAISS IndexFlatIP 实现近似 O(1) 的内积检索。

升级动机（方案一）：
  - 原实现：np.dot(embs_norm, text_norm) → O(N) 暴搜，10万电影时CPU峰值严重
  - 新实现：faiss.index.search() → FAISS 内建的高效 ANN 检索

使用方式：
    from myapp.utils.visual_faiss import VisualFAISSIndex
    vf = VisualFAISSIndex.instance()
    vf.build(movies_with_embeddings)
    indices, scores = vf.search(text_embedding, top_k=60)
================================================
"""

import numpy as np
import threading
import time
import logging

logger = logging.getLogger('movie_agent')


class VisualFAISSIndex:
    """
    FAISS 视觉向量索引（单例模式）
    使用 IndexFlatIP（精确内积搜索），L2 归一化后等价余弦相似度。
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.index = None
        self.movie_objects = []   # 按 FAISS 索引顺序存储的电影对象
        self.movie_ids = []       # 按 FAISS 索引顺序存储的电影 ID
        self.dimension = 512      # CLIP ViT-B/32 输出维度
        self._built = False

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def build(self, movies_with_embeddings):
        """
        从电影对象列表构建 FAISS 索引。

        Args:
            movies_with_embeddings: list of Movie objects with poster_embedding_json
        """
        try:
            import faiss
        except ImportError:
            logger.warning("[VisualFAISS] faiss 未安装，视觉检索将降级为 NumPy 暴搜")
            return False

        t_start = time.time()
        emb_list = []
        valid_movies = []
        valid_ids = []

        for m in movies_with_embeddings:
            try:
                emb_data = m.poster_embedding_json
                if isinstance(emb_data, str):
                    import json
                    emb_data = json.loads(emb_data)
                vec = np.array(emb_data, dtype=np.float32)
                if vec.shape[0] > 0 and not np.all(vec == 0):
                    # L2 归一化（使内积等价余弦相似度）
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    emb_list.append(vec)
                    valid_movies.append(m)
                    valid_ids.append(m.id)
            except Exception:
                continue

        if not emb_list:
            logger.warning("[VisualFAISS] 无有效视觉向量，索引构建跳过")
            return False

        self.dimension = emb_list[0].shape[0]
        vectors = np.array(emb_list, dtype=np.float32)

        # 构建 FAISS 索引
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

        self.movie_objects = valid_movies
        self.movie_ids = valid_ids
        self._built = True

        elapsed = time.time() - t_start
        logger.info(
            f"[VisualFAISS] 索引构建完成: {len(valid_movies)} 部电影, "
            f"维度={self.dimension}, 耗时={elapsed:.2f}s"
        )
        return True

    def search(self, query_embedding, top_k=60):
        """
        在 FAISS 索引中执行内积搜索。

        Args:
            query_embedding: np.ndarray, 查询向量（需已 L2 归一化）
            top_k: 返回前 K 个最近邻

        Returns:
            tuple: (indices: list[int], scores: list[float], movies: list[Movie])
        """
        if not self._built or self.index is None:
            logger.warning("[VisualFAISS] 索引未构建，无法搜索")
            return [], [], []

        # 确保查询向量已 L2 归一化
        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # FAISS 搜索
        scores, indices = self.index.search(vec, min(top_k, len(self.movie_ids)))

        result_indices = []
        result_scores = []
        result_movies = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.movie_ids):
                continue
            result_indices.append(self.movie_ids[idx])
            result_scores.append(float(score))
            result_movies.append(self.movie_objects[idx])

        return result_indices, result_scores, result_movies

    @property
    def is_ready(self):
        return self._built and self.index is not None

    @property
    def total_movies(self):
        return len(self.movie_ids) if self._built else 0