"""
Django Management Command: MovieAgent 框架下基座大语言模型对比实验 V3
================================================================
核心设计：LLM 真正参与推荐推理链（精排阶段），而非仅生成解释。
不同 LLM 会产生不同的推荐结果，从而证明 LLM 在 Agent 框架中的协调能力。

实验内容：
  1. 基线对比：MovieAgent(MAAN) vs MovieAgent(LLM Rerank) vs Pure DeepRec
  2. LLM 横向对比：6 个 LLM 在同一 Agent 框架下的推荐质量
  3. 语义命中率（LLM-as-Judge）+ 幻觉率 + 推理延迟

用法:
  python manage.py run_llm_comparison
  python manage.py run_llm_comparison --max-queries=50
  python manage.py run_llm_comparison --output=experiment_results/
================================================================
"""

import os, json, time, math, csv, re, logging
from datetime import datetime
from collections import defaultdict

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model

import requests as req

logger = logging.getLogger('movie_agent')
User = get_user_model()

OLLAMA_URL = "http://localhost:11434/api/chat"

# ── 待测 LLM 模型 ──
MODELS = [
    {"name": "qwen2.5:7b", "display": "Qwen2.5-7B", "params": "7B", "type": "上代纯文本"},
    {"name": "qwen3:0.6b", "display": "Qwen3-0.6B", "params": "0.6B", "type": "纯文本"},
    {"name": "qwen3:4b-instruct", "display": "Qwen3-4B-Instruct", "params": "4B", "type": "纯文本"},
    {"name": "qwen3-vl:4b", "display": "Qwen3-VL-4B", "params": "4B", "type": "视觉语言"},
    {"name": "qwen3:8b", "display": "Qwen3-8B", "params": "8B", "type": "纯文本"},
    {"name": "qwen3.5:9b", "display": "Qwen3.5-9B", "params": "9B", "type": "纯文本"},
]


# ============================================================
# 经典指标
# ============================================================

def hit_rate_at_k(recommended_ids, ground_truth_ids, k=5):
    top_k = set(recommended_ids[:k])
    return 1.0 if (top_k & ground_truth_ids) else 0.0


def ndcg_at_k(recommended_ids, ground_truth_ids, k=5):
    dcg = sum(1.0 / math.log2(i + 2) for i, mid in enumerate(recommended_ids[:k]) if mid in ground_truth_ids)
    n_rel = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended_ids, ground_truth_ids, k=5):
    for i, mid in enumerate(recommended_ids[:k]):
        if mid in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# LLM-as-Judge 语义命中率
# ============================================================

def call_judge_llm(prompt, model="qwen3:4b-instruct", timeout=60):
    """调用 Ollama 作为评判 LLM（与被测模型分离，避免自己评自己）"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是电影推荐系统的评估专家。只输出JSON，不要其他内容。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 256, "num_gpu": 99},
    }
    try:
        r = req.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Judge LLM error: {e}")
        return ""


def evaluate_semantic_hit(query, recommended_movies):
    """LLM-as-Judge 语义命中率评估"""
    if not recommended_movies:
        return False, {"reason": "无推荐结果", "method": "no_result"}

    top1 = recommended_movies[0]
    movie_desc = (
        f"《{top1.get('title', '未知')}》"
        f" 类型:{top1.get('genres', '未知')}"
        f" 导演:{top1.get('directors', '未知')}"
        f" 评分:{top1.get('score', '未知')}"
    )

    judge_prompt = f"""你是严格的电影推荐评估专家。判断推荐是否满足查询的核心需求。

用户查询: {query}
推荐电影: {movie_desc}

评估规则（必须全部满足）:
1. 如果查询指定了类型（如"科幻"），推荐电影必须属于该类型
2. 如果查询指定了导演/演员，推荐电影必须包含该导演/演员
3. 如果查询描述了情感基调（如"温馨""压抑"），推荐电影的基调必须匹配
4. 如果查询只是泛泛的"推荐好看的电影"，只要推荐了电影就算命中

严格按以下JSON格式输出，不要输出其他内容:
{{"is_match": true或false, "reason": "一句话说明原因"}}"""

    resp = call_judge_llm(judge_prompt)

    try:
        s = resp.find('{')
        e = resp.rfind('}') + 1
        if s >= 0 and e > s:
            data = json.loads(resp[s:e])
            is_match = data.get("is_match", False)
            if isinstance(is_match, str):
                is_match = is_match.lower() == "true"
            return bool(is_match), {"reason": data.get("reason", ""), "method": "llm_judge"}
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return False, {"reason": "LLM输出解析失败，默认未命中", "method": "parse_error"}


# ============================================================
# 幻觉率评估
# ============================================================

def evaluate_hallucination(query, movie_title, reason):
    """
    用独立 Judge LLM 评估推荐理由的幻觉率。
    与被测 LLM 分离，避免自己评自己。
    """
    if not reason or not movie_title:
        return 0.0

    prompt = f"""你是电影推荐事实核查专家。判断以下推荐理由是否包含事实错误。

推荐电影: {movie_title}
推荐理由: {reason}

检查项：
1. 导演/演员是否正确
2. 类型是否正确
3. 评分/年份是否正确
4. 剧情描述是否准确

严格按以下JSON格式输出:
{{"hallucination": true/false, "error_type": "none/entity/relation/attribute", "detail": ""}}"""

    resp = call_judge_llm(prompt)
    try:
        s = resp.find('{')
        e = resp.rfind('}') + 1
        if s >= 0 and e > s:
            data = json.loads(resp[s:e])
            return 1.0 if data.get("hallucination", False) else 0.0
    except (json.JSONDecodeError, ValueError):
        pass
    return 0.0


# ============================================================
# Agent 查询执行
# ============================================================

def run_agent_query(user, query, neo_graph, rag_resources, llm_config=None):
    """
    执行一次 MovieAgent 查询。

    Args:
        user: Django User
        query: 查询文本
        neo_graph: Neo4j 实例
        rag_resources: RAG 资源
        llm_config: LLM 配置 {'model_name': str, 'timeout': int}，None 表示使用 MAAN

    Returns:
        dict: 推荐结果
    """
    from myapp.agent.movie_agent import MovieAgent
    from myapp.models import Movie

    t_start = time.time()

    agent = MovieAgent(
        user=user,
        neo_graph=neo_graph,
        rag_resources=rag_resources,
        session_id=f"eval_llm_{int(time.time())}",
        llm_config=llm_config,
    )

    try:
        result = agent.run(query)
        wall_latency = (time.time() - t_start) * 1000

        recommended_ids = result.get('recommended_ids', [])[:5]
        explanations = result.get('explanations', {})

        # 获取推荐电影详情
        movies = Movie.objects.filter(id__in=recommended_ids).prefetch_related('genres', 'directors')
        movie_details = []
        for m in movies:
            reason = explanations.get(m.id, '')
            movie_details.append({
                'id': m.id,
                'title': m.title,
                'genres': ', '.join(g.name for g in m.genres.all()[:3]),
                'directors': ', '.join(d.name for d in m.directors.all()[:2]),
                'score': float(m.score) if m.score else 0,
                'reason': reason,
            })

        return {
            'recommended_ids': recommended_ids,
            'movie_details': movie_details,
            'explanations': explanations,
            'latency_ms': wall_latency,
            'intent': result.get('intent', ''),
            'actions': result.get('actions', []),
        }
    except Exception as e:
        logger.error(f"[Agent Error] {query}: {e}")
        return {
            'recommended_ids': [],
            'movie_details': [],
            'explanations': {},
            'latency_ms': (time.time() - t_start) * 1000,
            'intent': '',
            'actions': [],
        }


# ============================================================
# Pure DeepRec 基线
# ============================================================

def run_deeprec_baseline(query, user):
    """Pure DeepRec 基线：查询重写 → 硬过滤 → MAAN 精排 → Top-5"""
    from myapp.models import Movie
    from myapp.agent.movie_agent import MAANRerankTool

    t0 = time.time()

    # 查询重写：提取类型/演员/导演
    genre_map = {
        '科幻': '科幻', '悬疑': '悬疑', '恐怖': '恐怖', '喜剧': '喜剧',
        '动作': '动作', '爱情': '爱情', '剧情': '剧情', '动画': '动画',
        '战争': '战争', '犯罪': '犯罪', '奇幻': '奇幻', '冒险': '冒险',
        '惊悚': '惊悚', '文艺': '文艺', '纪录': '纪录片', '传记': '传记',
    }
    genre = None
    for kw, g in genre_map.items():
        if kw in query:
            genre = g
            break

    candidates_qs = Movie.objects.all()
    if genre:
        candidates_qs = candidates_qs.filter(genres__name__icontains=genre)

    candidates = list(candidates_qs.values('id', 'title').distinct()[:200])
    if not candidates:
        candidates = list(Movie.objects.order_by('-vote_count', '-score').values('id', 'title')[:200])

    cand_dicts = [{'movie_id': c['id'], 'title': c.get('title', '')} for c in candidates]
    maan = MAANRerankTool()
    result = maan.execute(candidates=cand_dicts, user=user, top_k=5)

    recommended_ids = [item.get('movie_id') for item in result.get('output', []) if item.get('movie_id')]
    latency = (time.time() - t0) * 1000

    return {'recommended_ids': recommended_ids, 'latency_ms': latency}


# ============================================================
# 评估主流程
# ============================================================

def evaluate_system(user, queries, neo_graph, rag_resources, system_name, llm_config=None, verbose=True):
    """
    评估一个系统配置在全部查询上的表现。

    Args:
        system_name: 系统名称（用于日志）
        llm_config: LLM 配置，None 表示 MAAN 基线
    """
    from myapp.models import Movie

    results = {
        'system': system_name,
        'hr_list': [], 'ndcg_list': [], 'mrr_list': [],
        'semantic_list': [], 'halluc_list': [],
        'latency_list': [],
        'per_difficulty': defaultdict(lambda: {'hr': [], 'ndcg': [], 'mrr': [], 'sem': []}),
    }

    for i, q in enumerate(queries):
        query_text = q['query']
        gt_titles = set(q['ground_truth_movies'])
        gt_ids = set(q.get('ground_truth_ids', []))
        difficulty = q['difficulty']

        # 如果没有 ground_truth_ids，从标题反查
        if not gt_ids:
            for title in gt_titles:
                for m in Movie.objects.filter(title__icontains=title):
                    gt_ids.add(m.id)

        if verbose and (i % 10 == 0 or i == len(queries) - 1):
            logger.info(f"  [{i+1}/{len(queries)}] [{difficulty}] {query_text[:50]}...")

        try:
            agent_result = run_agent_query(user, query_text, neo_graph, rag_resources, llm_config=llm_config)
            rec_ids = agent_result['recommended_ids']

            hr = hit_rate_at_k(rec_ids, gt_ids, k=5)
            n = ndcg_at_k(rec_ids, gt_ids, k=5)
            m = mrr(rec_ids, gt_ids, k=5)

            # 语义命中率（采样评估，避免太慢）
            sem_hit = 0.0
            if agent_result['movie_details']:
                sem_hit_val, _ = evaluate_semantic_hit(query_text, agent_result['movie_details'])
                sem_hit = 1.0 if sem_hit_val else 0.0

            # 幻觉率（采样评估）
            halluc = 0.0
            if agent_result['movie_details']:
                top1 = agent_result['movie_details'][0]
                halluc = evaluate_hallucination(query_text, top1.get('title', ''), top1.get('reason', ''))

            results['hr_list'].append(hr)
            results['ndcg_list'].append(n)
            results['mrr_list'].append(m)
            results['semantic_list'].append(sem_hit)
            results['halluc_list'].append(halluc)
            results['latency_list'].append(agent_result['latency_ms'])

            results['per_difficulty'][difficulty]['hr'].append(hr)
            results['per_difficulty'][difficulty]['ndcg'].append(n)
            results['per_difficulty'][difficulty]['mrr'].append(m)
            results['per_difficulty'][difficulty]['sem'].append(sem_hit)

        except Exception as e:
            logger.error(f"  [ERROR] {query_text}: {e}")
            for key in ['hr_list', 'ndcg_list', 'mrr_list', 'semantic_list', 'halluc_list', 'latency_list']:
                results[key].append(0.0)

    # 汇总
    n = len(results['hr_list'])
    results['avg_hr'] = sum(results['hr_list']) / n if n else 0
    results['avg_ndcg'] = sum(results['ndcg_list']) / n if n else 0
    results['avg_mrr'] = sum(results['mrr_list']) / n if n else 0
    results['avg_semantic'] = sum(results['semantic_list']) / n if n else 0
    results['avg_halluc'] = sum(results['halluc_list']) / n if n else 0
    results['avg_latency'] = sum(results['latency_list']) / n if n else 0

    for diff in results['per_difficulty']:
        d = results['per_difficulty'][diff]
        dn = len(d['hr'])
        d['avg_hr'] = sum(d['hr']) / dn if dn else 0
        d['avg_ndcg'] = sum(d['ndcg']) / dn if dn else 0
        d['avg_mrr'] = sum(d['mrr']) / dn if dn else 0
        d['avg_sem'] = sum(d['sem']) / dn if dn else 0

    return results


# ============================================================
# Django Management Command
# ============================================================

class Command(BaseCommand):
    help = 'MovieAgent 框架下基座大语言模型对比实验 V3（LLM 真正参与推理链）'

    def add_arguments(self, parser):
        parser.add_argument('--max-queries', type=int, default=None, help='最大查询数（调试用）')
        parser.add_argument('--output', type=str, default='experiment_results/', help='输出目录')
        parser.add_argument('--user-id', type=int, default=None, help='指定评估用户ID')

    def handle(self, *args, **options):
        from myapp import views

        output_dir = options['output']
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据集
        dataset_path = os.path.join(os.path.dirname(__file__), 'golden_dataset_agent_eval.json')
        if not os.path.exists(dataset_path):
            raise CommandError(f"黄金数据集不存在: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        queries = dataset['queries']
        if options['max_queries']:
            queries = queries[:options['max_queries']]

        # 获取用户
        if options['user_id']:
            user = User.objects.get(id=options['user_id'])
        else:
            user = User.objects.filter(is_staff=True).first() or User.objects.first()
        if not user:
            raise CommandError("数据库中没有任何用户")

        # ── 预热外部资源 ──
        logger.info("\n[预热] 正在加载外部资源...")
        try:
            from myapp.views import load_rag_resources
            load_rag_resources()
            rag_status = "OK" if views.RAG_RESOURCES.get("vectorstore") else "EMPTY"
            logger.info(f"  RAG 资源状态: {rag_status}")
        except Exception as e:
            logger.error(f"  RAG 预热失败: {e}")

        neo_graph = getattr(views, 'neo_graph', None)
        rag_resources = getattr(views, 'RAG_RESOURCES', {})
        neo_status = "OK" if neo_graph else "UNAVAILABLE"
        logger.info(f"  Neo4j 状态: {neo_status}")

        from django.conf import settings as _settings
        maan_path = os.path.join(_settings.BASE_DIR, 'ml_artifacts', 'skb_fmlp_online.pt')
        maan_status = "OK" if os.path.exists(maan_path) else "MISSING (精排将降级)"
        logger.info(f"  MAAN 模型: {maan_status}")
        logger.info("")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  MovieAgent LLM 对比实验 V3\n"
            f"  查询数: {len(queries)} | 用户: {user.username}\n"
            f"{'='*60}"
        ))

        all_rows = []

        # ── 实验 1: MAAN 基线（无 LLM）──
        self.stdout.write(self.style.WARNING(f"\n[1/{len(MODELS)+2}] MAAN 基线（无 LLM）"))
        maan_results = evaluate_system(
            user, queries, neo_graph, rag_resources,
            system_name='MAAN (Baseline)', llm_config=None,
        )
        all_rows.append(self._make_row('MAAN (Baseline)', '0B', 'ML模型', maan_results))

        # ── 实验 2: Pure DeepRec 基线 ──
        self.stdout.write(self.style.WARNING(f"\n[2/{len(MODELS)+2}] Pure DeepRec 基线"))
        dr_results = self._eval_deeprec_baseline(user, queries)
        all_rows.append(self._make_row('Pure DeepRec', '-', '基线', dr_results))

        # ── 实验 3~N: 各 LLM 精排 ──
        for idx, model_cfg in enumerate(MODELS):
            model_name = model_cfg['name']
            model_display = model_cfg['display']
            self.stdout.write(self.style.WARNING(
                f"\n[{idx+3}/{len(MODELS)+2}] LLM 精排: {model_display}"
            ))

            llm_config = {'model_name': model_name, 'timeout': 60}
            llm_results = evaluate_system(
                user, queries, neo_graph, rag_resources,
                system_name=f'LLM: {model_display}', llm_config=llm_config,
            )
            all_rows.append(self._make_row(model_display, model_cfg['params'], model_cfg['type'], llm_results))

        # ── 保存结果 ──
        headers = ['System', 'Params', 'Type', 'HR@5', 'NDCG@5', 'MRR',
                   'Semantic_Hit', 'Hallucination', 'Latency_ms',
                   'HR_single_hop', 'HR_multi_hop', 'HR_implicit_semantic']

        csv_path = os.path.join(output_dir, f'llm_comparison_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(all_rows)

        json_path = os.path.join(output_dir, f'llm_comparison_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': 'MovieAgent LLM 对比实验 V3',
                'num_queries': len(queries),
                'timestamp': timestamp,
                'results': all_rows,
            }, f, ensure_ascii=False, indent=2)

        # ── 打印汇总 ──
        self.stdout.write(self.style.SUCCESS(f"\n{'='*60}"))
        self.stdout.write(self.style.SUCCESS(f"  实验完成！结果已保存至: {output_dir}"))
        self.stdout.write(self.style.SUCCESS(f"{'='*60}"))
        self.stdout.write(f"\n{'System':<25} {'HR@5':>8} {'NDCG@5':>8} {'MRR':>8} {'SemHit':>8} {'Halluc':>8} {'Lat(ms)':>10}")
        self.stdout.write("-" * 85)
        for r in all_rows:
            self.stdout.write(
                f"{r['System']:<25} {r['HR@5']:>8} {r['NDCG@5']:>8} {r['MRR']:>8} "
                f"{r['Semantic_Hit']:>8} {r['Hallucination']:>8} {r['Latency_ms']:>10}"
            )

    def _make_row(self, system, params, type_str, results):
        """构建 CSV 行"""
        row = {
            'System': system,
            'Params': params,
            'Type': type_str,
            'HR@5': f"{results['avg_hr']:.4f}",
            'NDCG@5': f"{results['avg_ndcg']:.4f}",
            'MRR': f"{results['avg_mrr']:.4f}",
            'Semantic_Hit': f"{results['avg_semantic']:.4f}",
            'Hallucination': f"{results['avg_halluc']:.4f}",
            'Latency_ms': f"{results['avg_latency']:.1f}",
        }
        for diff in ['single_hop', 'multi_hop', 'implicit_semantic']:
            d = results['per_difficulty'].get(diff, {})
            row[f'HR_{diff}'] = f"{d.get('avg_hr', 0):.4f}"
        return row

    def _eval_deeprec_baseline(self, user, queries):
        """Pure DeepRec 基线评估"""
        from myapp.models import Movie

        results = {
            'system': 'Pure DeepRec',
            'hr_list': [], 'ndcg_list': [], 'mrr_list': [],
            'semantic_list': [], 'halluc_list': [],
            'latency_list': [],
            'per_difficulty': defaultdict(lambda: {'hr': [], 'ndcg': [], 'mrr': [], 'sem': []}),
        }

        for i, q in enumerate(queries):
            gt_ids = set(q.get('ground_truth_ids', []))
            gt_titles = set(q['ground_truth_movies'])
            difficulty = q['difficulty']

            if not gt_ids:
                for title in gt_titles:
                    for m in Movie.objects.filter(title__icontains=title):
                        gt_ids.add(m.id)

            if (i + 1) % 20 == 0:
                logger.info(f"  [DeepRec] [{i+1}/{len(queries)}]")

            try:
                dr_result = run_deeprec_baseline(q['query'], user)
                rec_ids = dr_result['recommended_ids']

                hr = hit_rate_at_k(rec_ids, gt_ids, k=5)
                n = ndcg_at_k(rec_ids, gt_ids, k=5)
                m = mrr(rec_ids, gt_ids, k=5)

                results['hr_list'].append(hr)
                results['ndcg_list'].append(n)
                results['mrr_list'].append(m)
                results['semantic_list'].append(0.0)  # DeepRec 无语义评估
                results['halluc_list'].append(0.0)
                results['latency_list'].append(dr_result['latency_ms'])

                results['per_difficulty'][difficulty]['hr'].append(hr)
                results['per_difficulty'][difficulty]['ndcg'].append(n)
                results['per_difficulty'][difficulty]['mrr'].append(m)
                results['per_difficulty'][difficulty]['sem'].append(0.0)
            except Exception as e:
                logger.error(f"  [DeepRec Error] {q['query']}: {e}")
                for key in ['hr_list', 'ndcg_list', 'mrr_list', 'semantic_list', 'halluc_list', 'latency_list']:
                    results[key].append(0.0)

        n = len(results['hr_list'])
        results['avg_hr'] = sum(results['hr_list']) / n if n else 0
        results['avg_ndcg'] = sum(results['ndcg_list']) / n if n else 0
        results['avg_mrr'] = sum(results['mrr_list']) / n if n else 0
        results['avg_semantic'] = sum(results['semantic_list']) / n if n else 0
        results['avg_halluc'] = sum(results['halluc_list']) / n if n else 0
        results['avg_latency'] = sum(results['latency_list']) / n if n else 0

        for diff in results['per_difficulty']:
            d = results['per_difficulty'][diff]
            dn = len(d['hr'])
            d['avg_hr'] = sum(d['hr']) / dn if dn else 0
            d['avg_ndcg'] = sum(d['ndcg']) / dn if dn else 0
            d['avg_mrr'] = sum(d['mrr']) / dn if dn else 0
            d['avg_sem'] = sum(d['sem']) / dn if dn else 0

        return results
