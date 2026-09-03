"""
BM25 文本检索器 (BM25 Retriever Module)
================================================
基于 jieba 分词构建全量电影文档的倒排 BM25Okapi 索引。

特点：
  1. 字段加权：标题权重放大(x3)，类型/导演/演员加权(x2)，配合剧情简介做全文召回。
  2. 极速计算：基于 NumPy / 原生 Counter 加速，无需第三方复杂依赖。
  3. 动态/单例加载：自动感知 Django Movie 模型数据更新。
================================================
"""

import math
import logging
from collections import Counter, defaultdict
import jieba

from myapp.models import Movie

logger = logging.getLogger('movie_agent')


class BM25Retriever:
    """
    基于 BM25Okapi 算法的电影文本检索器
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []          # 下标到 movie_id 的映射
        self.doc_lengths = []      # 每个文档的词数
        self.avgdl = 0.0           # 平均文档长度
        self.doc_freqs = defaultdict(int)  # 词在多少个文档出现 (df)
        self.tf_list = []          # 每个文档的词频 Counter
        self.num_docs = 0
        self.is_indexed = False

    def build_index(self, movies_qs=None):
        """从 Django Movie QuerySet 构建索引"""
        if movies_qs is None:
            movies_qs = Movie.objects.all().prefetch_related('genres', 'directors', 'actors')

        doc_ids = []
        doc_lengths = []
        tf_list = []
        doc_freqs = defaultdict(int)

        logger.info("[BM25Retriever] 开始构建 BM25 全量文档索引...")

        for m in movies_qs:
            # 权重增强：标题放大 3 次，类型/导演/演员放大 2 次
            title_text = (m.title + " ") * 3 if m.title else ""
            genres_text = (" ".join([g.name for g in m.genres.all()]) + " ") * 2
            directors_text = (" ".join([d.name for d in m.directors.all()]) + " ") * 2
            actors_text = (" ".join([a.name for a in m.actors.all()]) + " ") * 2
            summary_text = m.summary or ""

            full_text = f"{title_text}{genres_text}{directors_text}{actors_text}{summary_text}"

            # jieba 分词
            words = [w.strip().lower() for w in jieba.cut(full_text) if len(w.strip()) > 0]
            if not words:
                continue

            tf = Counter(words)
            doc_ids.append(m.id)
            doc_lengths.append(len(words))
            tf_list.append(tf)

            for word in tf.keys():
                doc_freqs[word] += 1

        self.num_docs = len(doc_ids)
        self.doc_ids = doc_ids
        self.doc_lengths = doc_lengths
        self.avgdl = sum(doc_lengths) / self.num_docs if self.num_docs > 0 else 1.0
        self.doc_freqs = doc_freqs
        self.tf_list = tf_list
        self.is_indexed = True

        logger.info(f"[BM25Retriever] BM25 索引构建完成: 共 {self.num_docs} 部电影，词典大小: {len(doc_freqs)}")

    def _calc_idf(self, word):
        """计算 IDF 分数 (BM25Okapi 变体，非负)"""
        df = self.doc_freqs.get(word, 0)
        if df == 0:
            return 0.0
        # idf = ln((N - n + 0.5) / (n + 0.5) + 1)
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query_text: str, top_k: int = 60):
        """
        检索 Top-K 电影

        Returns:
            List[Dict] - [{'movie_id': int, 'score': float, 'source': 'bm25'}, ...]
        """
        if not self.is_indexed:
            self.build_index()

        if not query_text or self.num_docs == 0:
            return []

        query_words = [w.strip().lower() for w in jieba.cut(query_text) if len(w.strip()) > 0]
        if not query_words:
            return []

        scores = [0.0] * self.num_docs

        for q_word in query_words:
            idf = self._calc_idf(q_word)
            if idf <= 0:
                continue

            for idx in range(self.num_docs):
                tf = self.tf_list[idx].get(q_word, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[idx]
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[idx] += idf * (num / denom)

        # 排序取 Top-K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0.0001:
                results.append({
                    'movie_id': self.doc_ids[idx],
                    'score': float(score),
                    'source': 'bm25'
                })

        return results


_bm25_retriever_instance = None


def get_bm25_retriever(force_rebuild=False):
    """获取 BM25Retriever 单例"""
    global _bm25_retriever_instance
    if _bm25_retriever_instance is None or force_rebuild:
        _bm25_retriever_instance = BM25Retriever()
        _bm25_retriever_instance.build_index()
    return _bm25_retriever_instance
