"""
Retrieval Benchmark — MovieAgent 检索层专项评估框架
================================================
用于评估与对比不同的检索策略 (FAISS Baseline vs BM25 vs Query Rewrite + BM25/FAISS + RRF)

指标体系:
  - Hit@K (Hit Ratio at K=5, 10, 20): 目标标定电影落在前 K 名的比例
  - MRR@10 (Mean Reciprocal Rank): 目标电影首次出现位置的倒数均值
  - NDCG@10 (Normalized Discounted Cumulative Gain): 排序折扣累计增益
  - Latency (ms): 平均检索耗时

使用方式:
    python3 manage.py shell -c "from myapp.agent.benchmark_retrieval import run_retrieval_benchmark; run_retrieval_benchmark()"
================================================
"""

import os
import json
import math
import time
import logging
from typing import List, Dict, Tuple, Set

from myapp.models import Movie
from myapp.recommender.recall import vector_recall
from myapp.recommender.bm25_retriever import get_bm25_retriever
from myapp.recommender.hybrid_retriever import get_hybrid_retriever

logger = logging.getLogger('movie_agent')

# ── 检索标定评测集 (40 题，涵盖 5 大场景) ──────────────────────────────────
RETRIEVAL_BENCHMARK_SET = [
    # 1. 精确片名/别名 (Exact Match)
    {"id": 1,  "category": "Exact Match",     "query": "盗梦空间",               "targets": ["盗梦空间"]},
    {"id": 2,  "category": "Exact Match",     "query": "星际穿越",               "targets": ["星际穿越"]},
    {"id": 3,  "category": "Exact Match",     "query": "肖申克的救赎",           "targets": ["肖申克的救赎"]},
    {"id": 4,  "category": "Exact Match",     "query": "教父",                   "targets": ["教父"]},
    {"id": 5,  "category": "Exact Match",     "query": "千与千寻",               "targets": ["千与千寻"]},
    {"id": 6,  "category": "Exact Match",     "query": "泰坦尼克号",             "targets": ["泰坦尼克号"]},
    {"id": 7,  "category": "Exact Match",     "query": "楚门的世界",             "targets": ["楚门的世界"]},
    {"id": 8,  "category": "Exact Match",     "query": "黑客帝国",               "targets": ["黑客帝国"]},

    # 2. 实体/导演/演员匹配 (Entity Match)
    {"id": 9,  "category": "Entity Match",    "query": "诺兰导演的电影",         "targets": ["盗梦空间", "星际穿越", "黑暗骑士", "敦刻尔克", "信条", "奥本海默"]},
    {"id": 10, "category": "Entity Match",    "query": "宫崎骏执导的动画片",     "targets": ["千与千寻", "龙猫", "哈尔的移动城堡", "天空之城", "幽灵公主"]},
    {"id": 11, "category": "Entity Match",    "query": "周星驰主演的喜剧",       "targets": ["功夫", "大话西游", "喜剧之王", "少林足球"]},
    {"id": 12, "category": "Entity Match",    "query": "汤姆汉克斯经典电影",     "targets": ["阿甘正传", "拯救大兵瑞恩", "西雅图夜未眠"]},
    {"id": 13, "category": "Entity Match",    "query": "昆汀执导的黑帮犯罪片",   "targets": ["低俗小说", "被解救的姜戈", "杀死比尔"]},
    {"id": 14, "category": "Entity Match",    "query": "大卫芬奇高分悬疑片",     "targets": ["七宗罪", "搏击俱乐部", "消失的爱人"]},

    # 3. 属性与类型组合 (Attribute & Genre)
    {"id": 15, "category": "Attribute Query", "query": "烧脑悬疑推理片",         "targets": ["禁闭岛", "记忆碎片", "致命ID", "盗梦空间", "七宗罪", "看不见的客人"]},
    {"id": 16, "category": "Attribute Query", "query": "高分硬核太空科幻片",     "targets": ["星际穿越", "2001太空漫游", "火星救援", "地心引力"]},
    {"id": 17, "category": "Attribute Query", "query": "温暖治愈高分动画",       "targets": ["龙猫", "千与千寻", "寻梦环游记", "疯狂动物城", "机器人总动员"]},
    {"id": 18, "category": "Attribute Query", "query": "经典黑帮犯罪动作片",     "targets": ["教父", "低俗小说", "美国往事", "教父2"]},
    {"id": 19, "category": "Attribute Query", "query": "适合情侣看的浪漫爱情片", "targets": ["泰坦尼克号", "真爱至上", "爱乐之城", "傲慢与偏见"]},
    {"id": 20, "category": "Attribute Query", "query": "爆笑轻松周末喜剧",       "targets": ["三傻大闹宝莱坞", "疯狂的石头", "宿醉", "触不可及"]},

    # 4. 隐式/剧情模糊描述 (Implicit Plot Search)
    {"id": 21, "category": "Implicit Plot",   "query": "关于梦境植入与多层潜意识的电影",       "targets": ["盗梦空间"]},
    {"id": 22, "category": "Implicit Plot",   "query": "人类在黑洞和五维空间探索救赎的故事",   "targets": ["星际穿越"]},
    {"id": 23, "category": "Implicit Plot",   "query": "在监狱里通过地道逃脱高墙获得自由的剧情", "targets": ["肖申克的救赎"]},
    {"id": 24, "category": "Implicit Plot",   "query": "小男孩在死亡世界寻找音乐梦想的故事",   "targets": ["寻梦环游记"]},
    {"id": 25, "category": "Implicit Plot",   "query": "一个清扫地球垃圾的废土机器人恋爱故事", "targets": ["机器人总动员"]},
    {"id": 26, "category": "Implicit Plot",   "query": "主角发现自己从小生活在真人秀电视节目中", "targets": ["楚门的世界"]},
    {"id": 27, "category": "Implicit Plot",   "query": "主角发现世界是被计算机代码虚拟出来的矩阵", "targets": ["黑客帝国"]},
    {"id": 28, "category": "Implicit Plot",   "query": "精神分裂症患者在孤岛精神病院调查的悬疑片", "targets": ["禁闭岛"]},

    # 5. 相似/对比检索 (Similarity / Comparison Search)
    {"id": 29, "category": "Similarity Search","query": "推荐类似盗梦空间的烧脑电影",         "targets": ["禁闭岛", "记忆碎片", "信条", "源代码", "致命ID"]},
    {"id": 30, "category": "Similarity Search","query": "有没有像银翼杀手那种赛博朋克风格的", "targets": ["黑客帝国", "攻壳机动队", "银翼杀手2049"]},
    {"id": 31, "category": "Similarity Search","query": "推荐和星际穿越风格类似的科幻片",     "targets": ["2001太空漫游", "接触未来", "地心引力", "火星救援"]},
    {"id": 32, "category": "Similarity Search","query": "类似疯狂动物城的动物拟人动画片",     "targets": ["欢乐好声音", "功夫熊猫", "冰川时代"]},
    {"id": 33, "category": "Similarity Search","query": "像楚门的世界那样探讨虚拟与现实的",   "targets": ["黑客帝国", "异次元黑客", "黑镜"]},
    {"id": 34, "category": "Similarity Search","query": "推荐类似阿甘正传的高分励志电影",     "targets": ["当幸福来敲门", "肖申克的救赎", "触不可及", "心灵捕手"]},
    {"id": 35, "category": "Similarity Search","query": "像三傻大闹宝莱坞那样既好笑又有深度的", "targets": ["摔跤吧！爸爸", "小萝莉的猴神大叔", "触不可及"]},
]


class RetrievalEvaluator:
    """
    检索能力度量计算器
    """

    def __init__(self, target_title_map: Dict[str, int]):
        """
        Args:
            target_title_map: 片名到 movie_id 的映射
        """
        self.title_to_id = target_title_map

    def evaluate_query(
        self,
        retrieved_items: List[Dict],
        target_titles: List[str],
        top_ks: List[int] = [5, 10, 20]
    ) -> Dict:
        """评估单个 Query 的检索指标"""
        # 将 target_titles 转化为 target_ids (支持模糊包含)
        target_ids = set()
        for title in target_titles:
            for t_title, mid in self.title_to_id.items():
                if title in t_title or t_title in title:
                    target_ids.add(mid)

        retrieved_ids = [item['movie_id'] for item in retrieved_items]

        # 1. Hit@K
        hits = {}
        for k in top_ks:
            top_k_ids = set(retrieved_ids[:k])
            hits[f"hit@{k}"] = 1.0 if bool(top_k_ids & target_ids) else 0.0

        # 2. MRR@10
        mrr_10 = 0.0
        for rank_idx, mid in enumerate(retrieved_ids[:10], start=1):
            if mid in target_ids:
                mrr_10 = 1.0 / rank_idx
                break

        # 3. NDCG@10
        ndcg_10 = 0.0
        dcg = 0.0
        for rank_idx, mid in enumerate(retrieved_ids[:10], start=1):
            if mid in target_ids:
                dcg += 1.0 / math.log2(rank_idx + 1)

        # 计算 IDCG (理想排序 DCG)
        ideal_hits = min(len(target_ids), 10)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        if idcg > 0:
            ndcg_10 = dcg / idcg

        return {
            **hits,
            'mrr@10': mrr_10,
            'ndcg@10': ndcg_10,
            'target_count': len(target_ids),
        }


def run_retrieval_benchmark(rag_resources=None) -> Dict:
    """
    运行全量检索 Baseline 对比 Benchmark
    """
    print("\n=========================================================")
    print("🚀 开始运行 MovieAgent 检索层 (Hybrid RAG) Benchmark 评估")
    print("=========================================================\n")

    # 构建数据库标题映射
    all_movies = Movie.objects.all().values('id', 'title')
    title_to_id = {m['title'].strip(): m['id'] for m in all_movies if m['title']}

    if rag_resources is None or "vectorstore" not in rag_resources:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'faiss_movie_index')
            if os.path.exists(index_path):
                embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
                vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                rag_resources = {"vectorstore": vector_db}
                print("✅ 成功自动装载 FAISS 向量库向量点集")
        except Exception as e:
            print(f"⚠️ FAISS 自动装载失败: {e}")

    evaluator = RetrievalEvaluator(title_to_id)
    hybrid_retriever = get_hybrid_retriever()
    bm25_retriever = get_bm25_retriever()

    modes = [
        ("Baseline (Pure FAISS)", "faiss"),
        ("BM25 Text Only", "bm25"),
        ("Hybrid (Rule Rewrite + BM25/FAISS + RRF)", "hybrid_rule"),
        ("Hybrid (LLM Rewrite + BM25/FAISS + RRF)", "hybrid_llm"),
    ]

    summary_reports = {}

    for mode_name, mode_key in modes:
        print(f"🔄 正在评测模式: [{mode_name}]...", flush=True)
        t_start = time.time()

        mode_metrics = {
            'hit@5': [], 'hit@10': [], 'hit@20': [],
            'mrr@10': [], 'ndcg@10': [], 'latency_ms': []
        }

        category_metrics = {}

        for idx, item in enumerate(RETRIEVAL_BENCHMARK_SET, 1):
            q = item['query']
            targets = item['targets']
            cat = item['category']

            q_start = time.time()

            # 执行检索
            if mode_key == "faiss":
                retrieved = vector_recall(q, k=20, rag_resources=rag_resources)
            elif mode_key == "bm25":
                retrieved = bm25_retriever.search(q, top_k=20)
            elif mode_key == "hybrid_rule":
                retrieved = hybrid_retriever.retrieve(q, top_k=20, rag_resources=rag_resources, enable_rewrite=True, use_llm=False)
            elif mode_key == "hybrid_llm":
                retrieved = hybrid_retriever.retrieve(q, top_k=20, rag_resources=rag_resources, enable_rewrite=True, use_llm=True)
            else:
                retrieved = []

            q_latency = (time.time() - q_start) * 1000

            res = evaluator.evaluate_query(retrieved, targets)

            mode_metrics['hit@5'].append(res['hit@5'])
            mode_metrics['hit@10'].append(res['hit@10'])
            mode_metrics['hit@20'].append(res['hit@20'])
            mode_metrics['mrr@10'].append(res['mrr@10'])
            mode_metrics['ndcg@10'].append(res['ndcg@10'])
            mode_metrics['latency_ms'].append(q_latency)

            if cat not in category_metrics:
                category_metrics[cat] = {'hit@10': [], 'ndcg@10': []}
            category_metrics[cat]['hit@10'].append(res['hit@10'])
            category_metrics[cat]['ndcg@10'].append(res['ndcg@10'])

        avg_hit5 = sum(mode_metrics['hit@5']) / len(mode_metrics['hit@5'])
        avg_hit10 = sum(mode_metrics['hit@10']) / len(mode_metrics['hit@10'])
        avg_hit20 = sum(mode_metrics['hit@20']) / len(mode_metrics['hit@20'])
        avg_mrr10 = sum(mode_metrics['mrr@10']) / len(mode_metrics['mrr@10'])
        avg_ndcg10 = sum(mode_metrics['ndcg@10']) / len(mode_metrics['ndcg@10'])
        avg_lat = sum(mode_metrics['latency_ms']) / len(mode_metrics['latency_ms'])

        cat_summary = {}
        for cat, cdata in category_metrics.items():
            cat_summary[cat] = {
                'Hit@10': round(sum(cdata['hit@10']) / len(cdata['hit@10']), 4),
                'NDCG@10': round(sum(cdata['ndcg@10']) / len(cdata['ndcg@10']), 4),
            }

        summary_reports[mode_key] = {
            'mode_name': mode_name,
            'Hit@5': round(avg_hit5, 4),
            'Hit@10': round(avg_hit10, 4),
            'Hit@20': round(avg_hit20, 4),
            'MRR@10': round(avg_mrr10, 4),
            'NDCG@10': round(avg_ndcg10, 4),
            'Avg_Latency_ms': round(avg_lat, 2),
            'category_breakdown': cat_summary
        }
        print(f"✅ [{mode_name}] 评估完成: Hit@10={avg_hit10:.4f}, NDCG@10={avg_ndcg10:.4f}, Avg Latency={avg_lat:.1f}ms", flush=True)

    # 打印对比表格
    print("\n=========================================================================================", flush=True)
    print(f"{'检索模式 (Retrieval Strategy)':<42} | {'Hit@5':<7} | {'Hit@10':<7} | {'Hit@20':<7} | {'MRR@10':<7} | {'NDCG@10':<7} | {'Latency':<8}", flush=True)
    print("-----------------------------------------------------------------------------------------", flush=True)
    for key, rep in summary_reports.items():
        print(f"{rep['mode_name']:<42} | {rep['Hit@5']:<7.4f} | {rep['Hit@10']:<7.4f} | {rep['Hit@20']:<7.4f} | {rep['MRR@10']:<7.4f} | {rep['NDCG@10']:<7.4f} | {rep['Avg_Latency_ms']:<6.1f}ms", flush=True)
    print("=========================================================================================\n", flush=True)

    # 保存报告 JSON
    output_file = "retrieval_benchmark_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_reports, f, ensure_ascii=False, indent=2)

    print(f"✅ Benchmark 报告已成功写入到文件: {output_file}\n", flush=True)

    return summary_reports
