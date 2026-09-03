"""
混合 RAG 检索器 (Hybrid RAG Retriever with Query Rewrite & RRF)
================================================
整合 Query Rewrite、BM25 倒排检索、FAISS 向量检索，并利用 RRF (Reciprocal Rank Fusion) 融合。

工作流程：
  1. 输入原始 Query (用户输入)
  2. 调用 QueryRewriter 生成拓展后的拓展 Query
  3. 分别并行/依次触发：
     - FAISS 向量语义召回 (vector_recall)
     - BM25 文本匹配召回 (bm25_retriever.search)
  4. RRF 排名融合：
     RRF_Score(d) = w_vector / (k + rank_vector(d)) + w_bm25 / (k + rank_bm25(d))
  5. 返回去重归一化排序后的 Top-K 电影列表
================================================
"""

import time
import logging
from typing import List, Dict, Optional

from myapp.recommender.query_rewriter import get_query_rewriter
from myapp.recommender.bm25_retriever import get_bm25_retriever
from myapp.recommender.recall import vector_recall

logger = logging.getLogger('movie_agent')


class HybridRAGRetriever:
    """
    Query Rewrite + BM25/FAISS + RRF 混合检索器
    """

    def __init__(self, rrf_k: int = 60, w_vector: float = 1.0, w_bm25: float = 1.0):
        self.rrf_k = rrf_k
        self.w_vector = w_vector
        self.w_bm25 = w_bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        excluded_ids: Optional[List[int]] = None,
        rag_resources: Optional[Dict] = None,
        enable_rewrite: bool = True,
        use_llm: bool = True
    ) -> List[Dict]:
        """
        全流程混合检索
        """
        start_time = time.time()
        excluded_ids = set(excluded_ids or [])

        # 1. Query Rewrite
        expanded_query = query
        if enable_rewrite:
            try:
                rewriter = get_query_rewriter()
                rewriter.use_llm = use_llm
                expanded_query = rewriter.rewrite(query)
                if not expanded_query:
                    expanded_query = query
            except Exception as e:
                logger.warning(f"[HybridRAGRetriever] Query Rewrite 失败，回退到原始 Query: {e}")
                expanded_query = query

        # 2. 向量语义检索 (FAISS)
        candidate_limit = top_k * 3
        vec_results = []
        try:
            vec_results = vector_recall(
                expanded_query,
                excluded_ids=list(excluded_ids),
                k=candidate_limit,
                rag_resources=rag_resources
            )
        except Exception as e:
            logger.error(f"[HybridRAGRetriever] 向量检索失败: {e}")

        # 3. BM25 文本检索
        bm25_results = []
        try:
            bm25_retriever = get_bm25_retriever()
            bm25_results = bm25_retriever.search(expanded_query, top_k=candidate_limit)
            if excluded_ids:
                bm25_results = [item for item in bm25_results if item['movie_id'] not in excluded_ids]
        except Exception as e:
            logger.error(f"[HybridRAGRetriever] BM25 检索失败: {e}")

        # 4. RRF (Reciprocal Rank Fusion) 排名融合
        rrf_scores = {}
        movie_ranks = {}  # {mid: {'vector': rank, 'bm25': rank}}

        # 向量召回计分
        for rank, item in enumerate(vec_results, start=1):
            mid = item['movie_id']
            score_contrib = self.w_vector / (self.rrf_k + rank)
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + score_contrib
            if mid not in movie_ranks:
                movie_ranks[mid] = {}
            movie_ranks[mid]['vector'] = rank

        # BM25 召回计分
        for rank, item in enumerate(bm25_results, start=1):
            mid = item['movie_id']
            score_contrib = self.w_bm25 / (self.rrf_k + rank)
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + score_contrib
            if mid not in movie_ranks:
                movie_ranks[mid] = {}
            movie_ranks[mid]['bm25'] = rank

        # 如果两路召回均为空，进行降级兜底 (如使用纯原始 query 再查一次)
        if not rrf_scores and expanded_query != query:
            try:
                bm25_retriever = get_bm25_retriever()
                fallback_bm25 = bm25_retriever.search(query, top_k=top_k)
                for rank, item in enumerate(fallback_bm25, start=1):
                    mid = item['movie_id']
                    rrf_scores[mid] = 1.0 / (self.rrf_k + rank)
            except Exception:
                pass

        # 排序输出 Top-K
        sorted_mids = sorted(rrf_scores.keys(), key=lambda mid: rrf_scores[mid], reverse=True)[:top_k]

        max_score = max(rrf_scores.values()) if rrf_scores else 1.0
        final_results = []
        for mid in sorted_mids:
            ranks = movie_ranks.get(mid, {})
            # 归一化得分 [0, 1]
            norm_score = round(rrf_scores[mid] / max_score, 4)
            final_results.append({
                'movie_id': mid,
                'score': norm_score,
                'rrf_raw_score': round(rrf_scores[mid], 6),
                'source': 'hybrid_rrf',
                'vector_rank': ranks.get('vector', None),
                'bm25_rank': ranks.get('bm25', None),
            })

        latency = round((time.time() - start_time) * 1000, 2)
        logger.debug(
            f"[HybridRAGRetriever] 检索完成: Query='{query}' -> Expanded='{expanded_query}' | "
            f"Vector={len(vec_results)}, BM25={len(bm25_results)} | 融合Top={len(final_results)} | 耗时={latency}ms"
        )

        return final_results


_hybrid_retriever_instance = None


def get_hybrid_retriever():
    """获取单例 HybridRAGRetriever"""
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRAGRetriever()
    return _hybrid_retriever_instance
