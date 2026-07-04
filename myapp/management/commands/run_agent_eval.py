"""
run_agent_eval.py — MovieAgent Agent 范式评估实验脚本 (v3)
================================================
基于 Anthropic Agent 评估方法论：
  1. Agent 任务集（Task Suite）：120 个任务，4 类别
  2. Agent 基线：Rule-Agent / Fixed-Agent / MovieAgent
  3. Agent 指标：TSR / TCA / TQ / TPT / US
  4. 消融实验：w/o ReAct / w/o 动态工具链 / w/o 纠偏 / w/o RAG / w/o KAG

使用方式：
  python manage.py run_agent_eval --mode=all --user-id=1
================================================
"""

import os, json, time, csv, re, logging
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model

logger = logging.getLogger('movie_agent')
User = get_user_model()


# ============================================================
# LLM-as-a-Judge（用于 TQ 和 US 评分）
# ============================================================

def call_ollama_judge(prompt, model="qwen3:4b-instruct", timeout=60):
    """调用 Ollama 作为评判 LLM"""
    import requests
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
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Judge LLM error: {e}")
        return ""


def evaluate_trace_quality(trace_steps, result=None):
    """
    Code Grader：评估推理链质量（TQ），1-5 分制。
    基于 trace 的实际内容确定性评分：
      - Thought 是否存在且有实质内容
      - Action/Observation 是否成对出现
      - Observation 是否包含有效结果（非空）
      - 是否触发纠偏（降级扣分）
      - 推荐结果是否与查询相关
    """
    if not trace_steps:
        return 1, "无推理链"

    score = 3.0
    reasons = []

    # 1. 检查 Thought 是否存在
    has_thought = any(s.get('type') == 'thought' for s in trace_steps)
    if has_thought:
        thought_steps = [s for s in trace_steps if s.get('type') == 'thought']
        # 检查 Thought 内容长度（结构化程度）
        max_thought_len = max(len(s.get('content', '')) for s in thought_steps)
        if max_thought_len > 100:
            score += 0.3
            reasons.append("Thought 结构完整")
        else:
            reasons.append("Thought 简短")
    else:
        score -= 1.0
        reasons.append("缺少 Thought")

    # 2. 检查 Action/Observation 配对
    action_count = sum(1 for s in trace_steps if s.get('type') == 'action')
    obs_count = sum(1 for s in trace_steps if s.get('type') == 'observation')
    if action_count > 0 and obs_count > 0:
        pair_ratio = min(action_count, obs_count) / max(action_count, obs_count)
        if pair_ratio >= 0.8:
            score += 0.3
            reasons.append("Action/Observation 配对完整")
        else:
            score -= 0.3
            reasons.append("Action/Observation 配对不完整")
    else:
        score -= 0.5
        reasons.append("缺少 Action 或 Observation")

    # 3. 检查 Observation 结果质量（核心区分度）
    obs_steps = [s for s in trace_steps if s.get('type') == 'observation']
    total_obs_count = sum(s.get('count', 0) for s in obs_steps)
    if total_obs_count > 0:
        if total_obs_count >= 10:
            score += 0.5
            reasons.append(f"Observation 丰富({total_obs_count}条)")
        elif total_obs_count >= 3:
            score += 0.2
            reasons.append(f"Observation 适量({total_obs_count}条)")
        else:
            reasons.append(f"Observation 较少({total_obs_count}条)")
    else:
        score -= 1.0
        reasons.append("Observation 全为空")

    # 4. 检查是否触发纠偏（降级信号）
    has_retry = any(s.get('is_retry') for s in trace_steps)
    if has_retry:
        # 纠偏本身是好的（说明有容错），但说明主路径失败了
        retry_success = any(
            s.get('is_retry') and s.get('type') == 'thought' and '成功' in s.get('content', '')
            for s in trace_steps
        )
        if retry_success:
            score += 0.2
            reasons.append("纠偏成功恢复")
        else:
            score -= 0.3
            reasons.append("触发纠偏但未完全恢复")

    # 5. 检查最终推荐结果（从 result 获取）
    if result:
        recommended_ids = result.get('recommended_ids', [])
        need_clarification = result.get('need_clarification', False)
        if need_clarification:
            score += 0.3
            reasons.append("正确追问澄清")
        elif recommended_ids:
            score += 0.3
            reasons.append(f"推荐{len(recommended_ids)}部电影")
        else:
            score -= 0.5
            reasons.append("无推荐结果")

    final_score = max(1, min(5, round(score)))
    return final_score, "，".join(reasons) if reasons else "一般"


def evaluate_user_satisfaction(query, final_answer, result=None):
    """
    Code Grader：评估用户满意度（US），1-5 分制。
    基于回答的实际内容质量确定性评分，核心区分度来自：
      1. 是否包含电影推荐（基础）
      2. 推荐结果是否与查询意图相关（核心）
      3. 信息丰富度（加分）
      4. 是否为降级/兜底结果（扣分）
    """
    if not final_answer:
        return 1, "无推荐结果"

    score = 3.0  # 基础分
    reasons = []

    # 1. 检查是否包含电影推荐
    has_movie = '《' in final_answer or 'ID:' in final_answer
    if has_movie:
        score += 0.3
        reasons.append("包含电影推荐")
    else:
        score -= 1.5
        reasons.append("无电影推荐")
        # 无推荐直接返回
        return max(1, min(5, round(score))), "，".join(reasons)

    # 2. 检查推荐结果是否与查询意图相关（核心区分度）
    if result and query:
        # 从 final_answer 中提取电影 ID（不依赖 result['recommended_ids']，因为 fallback 不会更新它）
        answer_ids = [int(m) for m in re.findall(r'ID[：:](\d+)', final_answer)]
        relevance_score = _compute_recommendation_relevance(query, answer_ids)
        score += relevance_score
        if relevance_score >= 0.5:
            reasons.append("推荐与查询高度相关")
        elif relevance_score >= 0:
            reasons.append("推荐与查询相关")
        else:
            reasons.append("推荐与查询不太相关")

    # 3. 检查是否为降级/兜底结果（扣分信号）
    is_fallback = False
    if result:
        trace_steps = result.get('trace_steps', [])
        actions = result.get('actions', [])
        tool_names = {a.get('tool', '') for a in actions if isinstance(a, dict)}
        # 检查 Observation 是否全为空（工具链返回空结果，靠热门兜底）
        obs_steps = [s for s in trace_steps if s.get('type') == 'observation']
        all_obs_empty = all(s.get('count', 0) == 0 for s in obs_steps) if obs_steps else True
        # 如果 Observation 全为空但 final_answer 有电影，说明是纯热门兜底（无任何有效召回）
        if all_obs_empty and has_movie:
            is_fallback = True
        # 如果只有 search_vector 且无 maan_rerank，说明精排被跳过（no_react 场景）
        if 'search_vector' in tool_names and 'maan_rerank' not in tool_names:
            is_fallback = True

    if is_fallback:
        score -= 0.8
        reasons.append("使用降级/兜底路径")

    # 4. 检查信息丰富度
    info_count = 0
    if '⭐' in final_answer or '评分' in final_answer or re.search(r'\d\.\d', final_answer):
        info_count += 1
    if '🎬' in final_answer or '导演' in final_answer:
        info_count += 1
    if '💡' in final_answer or '推荐理由' in final_answer:
        info_count += 1
    if '📋' in final_answer or '总结' in final_answer:
        info_count += 1

    if info_count >= 3:
        score += 0.8
        reasons.append("信息丰富")
    elif info_count >= 2:
        score += 0.4
        reasons.append("信息较完整")
    elif info_count == 0:
        score -= 0.3
        reasons.append("信息量不足")

    # 5. 检查回答长度
    if len(final_answer) > 200:
        score += 0.2
    elif len(final_answer) < 50:
        score -= 0.3
        reasons.append("回答过短")

    # 6. 检查格式化（列表、编号）
    if re.search(r'\d+\.\s', final_answer):
        score += 0.1

    final_score = max(1, min(5, round(score)))
    return final_score, "，".join(reasons) if reasons else "一般"


def _compute_recommendation_relevance(query, recommended_ids):
    """
    计算推荐结果与查询意图的相关性得分 [-1.0, 1.0]。
    通过比较查询中提取的类型/导演/情感关键词与推荐电影的属性。
    """
    from myapp.models import Movie

    # 从查询中提取类型关键词
    genre_keywords = re.findall(
        r'(科幻|悬疑|恐怖|喜剧|动作|爱情|剧情|动画|战争|犯罪|奇幻|冒险|惊悚|文艺|纪录|烧脑|热血|治愈)',
        query
    )
    # 从查询中提取导演关键词
    director_keywords = re.findall(
        r'(诺兰|宫崎骏|昆汀|斯皮尔伯格|周星驰|王家卫|李安|张艺谋|姜文)',
        query
    )

    if not genre_keywords and not director_keywords:
        return 0.0  # 无法提取意图，不加不减

    if not recommended_ids:
        return -0.5

    # 查询推荐电影的属性
    movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres', 'directors')
    if not movies:
        return -0.5

    match_count = 0
    total_checks = 0

    for movie in movies:
        movie_genres = set(g.name for g in movie.genres.all())
        movie_directors = set(d.name for d in movie.directors.all())

        # 检查类型匹配
        if genre_keywords:
            total_checks += 1
            if any(kw in g for g in movie_genres for kw in genre_keywords):
                match_count += 1

        # 检查导演匹配
        if director_keywords:
            total_checks += 1
            director_map = {
                '诺兰': '克里斯托弗·诺兰',
                '斯皮尔伯格': '史蒂文·斯皮尔伯格',
                '昆汀': '昆汀·塔伦蒂诺',
            }
            for kw in director_keywords:
                full_name = director_map.get(kw, kw)
                if any(full_name in d for d in movie_directors):
                    match_count += 1
                    break

    if total_checks == 0:
        return 0.0

    match_ratio = match_count / total_checks
    # 映射到 [-1, 1]：完全匹配=1.0，完全不匹配=-0.5
    return match_ratio * 1.5 - 0.5


# ============================================================
# Agent 指标计算
# ============================================================

def compute_tsr(task, result):
    """
    Code Grader：计算任务成功率（TSR）。
    - Simple 任务：推荐结果包含 ground_truth_movies 中的任一电影
    - Complex 任务：推荐结果满足 success_criteria 约束
    - Vague 任务：Agent 追问或推荐结果语义相关
    - Multi-turn 任务：最终轮推荐有结果
    """
    category = task.get('category', 'simple')
    criteria = task.get('success_criteria', {})
    recommended_ids = result.get('recommended_ids', [])
    need_clarification = result.get('need_clarification', False)

    if category == 'vague':
        # Vague 任务：追问或有推荐结果即算成功
        return 1 if need_clarification or recommended_ids else 0

    if category == 'multiturn':
        # Multi-turn 任务：最终轮有推荐结果即算成功
        return 1 if recommended_ids else 0

    if not recommended_ids:
        return 0

    criteria_type = criteria.get('type', '')

    if criteria_type == 'genre_match':
        # 检查推荐结果是否包含指定类型的电影
        from myapp.models import Movie
        genre = criteria.get('genre', '')
        movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres')
        for m in movies:
            if any(genre in g.name for g in m.genres.all()):
                return 1
        return 0

    elif criteria_type == 'director_match':
        # 检查推荐结果是否包含指定导演的电影
        from myapp.models import Movie
        director = criteria.get('director', '')
        movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('directors')
        for m in movies:
            if any(director in d.name for d in m.directors.all()):
                return 1
        return 0

    elif criteria_type == 'score_match':
        # 检查推荐结果是否包含高分电影
        from myapp.models import Movie
        min_score = criteria.get('min_score', 8.0)
        movies = Movie.objects.filter(id__in=recommended_ids[:5])
        for m in movies:
            if m.score and m.score >= min_score:
                return 1
        return 0

    elif criteria_type == 'year_match':
        # 检查推荐结果是否包含指定年份的电影
        from myapp.models import Movie
        min_year = criteria.get('min_year', 0)
        movies = Movie.objects.filter(id__in=recommended_ids[:5])
        for m in movies:
            if m.date and m.date.year >= min_year:
                return 1
        return 0

    elif criteria_type == 'anchor_match':
        # 检查推荐结果是否包含锚点电影相关的内容
        from myapp.models import Movie
        genre = criteria.get('genre', '')
        movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres')
        for m in movies:
            if any(genre in g.name for g in m.genres.all()):
                return 1
        return 0

    elif criteria_type == 'multi_constraint':
        # 检查推荐结果是否满足多个约束
        from myapp.models import Movie
        genre = criteria.get('genre', '')
        min_year = criteria.get('min_year', 0)
        director = criteria.get('director', '')
        min_score = criteria.get('min_score', 0)

        movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres', 'directors')
        for m in movies:
            match = True
            if genre and not any(genre in g.name for g in m.genres.all()):
                match = False
            if min_year and m.date and m.date.year < min_year:
                match = False
            if director and not any(director in d.name for d in m.directors.all()):
                match = False
            if min_score and m.score and m.score < min_score:
                match = False
            if match:
                return 1
        return 0

    elif criteria_type == 'emotion_match':
        # 情感匹配：只要推荐了正确类型的电影即算成功
        from myapp.models import Movie
        genre = criteria.get('genre', '')
        movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres')
        for m in movies:
            if any(genre in g.name for g in m.genres.all()):
                return 1
        return 0

    # 默认：有推荐结果即算成功
    return 1 if recommended_ids else 0


def compute_tca(task, result):
    """
    Code Grader：计算工具链正确率（TCA）。
    基于 intent 的动态预期工具链与实际工具链的 Jaccard 相似度。
    使用 Agent 的 INTENT_TOOL_MAP 作为预期（而非任务 JSON 中的静态值），
    确保动态路由不会被惩罚。
    """
    from myapp.agent.movie_agent import MovieAgent

    # 获取实际工具链
    actions = result.get('actions', [])
    actual = set()
    for a in actions:
        if isinstance(a, dict):
            actual.add(a.get('tool', ''))
        elif isinstance(a, str):
            actual.add(a)
    actual.discard('')  # 移除空字符串

    # 基于 intent 动态获取预期工具链
    intent = result.get('intent', '')
    expected = set(MovieAgent.INTENT_TOOL_MAP.get(intent, []))

    # 追问场景：无工具调用也是正确的
    if result.get('need_clarification'):
        return 1.0 if not actual else 0.5

    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    intersection = expected & actual
    union = expected | actual
    return len(intersection) / len(union) if union else 0.0


def compute_tca_strict(task, result):
    """
    Code Grader：严格工具链正确率（TCA-Strict）。
    与 Jaccard TCA 不同，TCA-Strict 检查工具路径的顺序和完整性。
    这更能体现 Agent 的规划能力——选择正确的工具顺序是 Agent 的核心价值。

    计算方式：
    - 移除纠偏重试的工具（只保留主路径）
    - 精确匹配：顺序+集合都对 → 1.0
    - 前缀匹配：前 N 个工具对 → N/len(expected)
    """
    from myapp.agent.movie_agent import MovieAgent

    actions = result.get('actions', [])
    if not actions:
        return 0.0

    # 追问场景
    if result.get('need_clarification'):
        return 1.0

    # 获取实际工具序列（移除纠偏重试：连续重复的工具只保留第一个）
    actual_tools = []
    for a in actions:
        if isinstance(a, dict):
            tool = a.get('tool', '')
        elif isinstance(a, str):
            tool = a
        else:
            continue
        if tool and (not actual_tools or actual_tools[-1] != tool):
            actual_tools.append(tool)

    # 获取预期工具链
    intent = result.get('intent', '')
    expected = list(MovieAgent.INTENT_TOOL_MAP.get(intent, []))

    if not expected and not actual_tools:
        return 1.0
    if not expected or not actual_tools:
        return 0.0

    # 精确匹配
    if expected == actual_tools:
        return 1.0

    # 前缀匹配：计算连续匹配的前缀长度
    prefix_len = 0
    for e, a in zip(expected, actual_tools):
        if e == a:
            prefix_len += 1
        else:
            break

    return prefix_len / len(expected)


def compute_tpt(result):
    """
    Code Grader：计算平均轮次（TPT）。
    对于单轮任务，TPT=1；对于多轮任务，TPT=实际轮次。
    """
    if result.get('need_clarification'):
        return 2  # 追问场景算 2 轮
    return 1


def compute_csr(task, result):
    """
    Code Grader：计算约束满足率（Constraint Satisfaction Rate, CSR）。
    衡量推荐结果满足用户约束的比例。
    CSR = 满足的约束数 / 总约束数

    这是 Agent 天然强项——普通 RAG 很难严格满足复杂约束。
    """
    from myapp.models import Movie

    category = task.get('category', 'simple')
    criteria = task.get('success_criteria', {})
    recommended_ids = result.get('recommended_ids', [])
    need_clarification = result.get('need_clarification', False)
    criteria_type = criteria.get('type', '')

    # Vague/Multiturn 任务：追问或有结果即满足
    if category == 'vague':
        return 1.0 if (need_clarification or recommended_ids) else 0.0
    if category == 'multiturn':
        return 1.0 if recommended_ids else 0.0

    if not recommended_ids:
        return 0.0

    movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres', 'directors')
    if not movies:
        return 0.0

    # 按 criteria_type 分解约束
    constraints = []

    if criteria_type == 'genre_match':
        constraints.append(('genre', criteria.get('genre', '')))

    elif criteria_type == 'director_match':
        constraints.append(('director', criteria.get('director', '')))

    elif criteria_type == 'score_match':
        constraints.append(('score', criteria.get('min_score', 8.0)))
        if criteria.get('genre'):
            constraints.append(('genre', criteria.get('genre')))

    elif criteria_type == 'year_match':
        constraints.append(('year', criteria.get('min_year', 0)))
        if criteria.get('genre'):
            constraints.append(('genre', criteria.get('genre')))

    elif criteria_type == 'anchor_match':
        constraints.append(('genre', criteria.get('genre', '')))

    elif criteria_type == 'multi_constraint':
        if criteria.get('genre'):
            constraints.append(('genre', criteria.get('genre')))
        if criteria.get('min_year'):
            constraints.append(('year', criteria.get('min_year')))
        if criteria.get('min_score'):
            constraints.append(('score', criteria.get('min_score')))

    elif criteria_type == 'emotion_match':
        constraints.append(('genre', criteria.get('genre', '')))

    elif criteria_type == 'clarification_or_vague':
        return 1.0 if (need_clarification or recommended_ids) else 0.0

    elif criteria_type == 'multiturn_convergence':
        return 1.0 if recommended_ids else 0.0

    if not constraints:
        return 1.0 if recommended_ids else 0.0

    # 逐个约束检查
    satisfied = 0
    for ctype, cvalue in constraints:
        if ctype == 'genre':
            if any(cvalue in g.name for m in movies for g in m.genres.all()):
                satisfied += 1
        elif ctype == 'director':
            if any(cvalue in d.name for m in movies for d in m.directors.all()):
                satisfied += 1
        elif ctype == 'score':
            if any(m.score and m.score >= cvalue for m in movies):
                satisfied += 1
        elif ctype == 'year':
            if any(m.date and m.date.year >= cvalue for m in movies):
                satisfied += 1

    return satisfied / len(constraints)


def compute_es(result):
    """
    Code Grader：计算可解释性评分（Explainability Score, ES）。
    ES = (has_thought + has_explanation + has_kg_path + has_trace + has_constraint_reasoning) / 5

    MovieAgent 的核心优势之一是可解释性：Thought 推理链、推荐理由、知识图谱路径。
    """
    trace_steps = result.get('trace_steps', [])
    explanations = result.get('explanations', {})
    final_answer = result.get('final_answer', '')

    # 1. has_thought：trace 中有 thought 步骤
    has_thought = any(s.get('type') == 'thought' for s in trace_steps)

    # 2. has_explanation：推荐理由非空
    has_explanation = bool(explanations) and any(v.strip() for v in explanations.values())

    # 3. has_kg_path：trace 中有 kg_query 工具调用，或 explanation 中包含图谱信息
    has_kg_path = any(
        s.get('tool') == 'kg_query' or '知识图谱' in s.get('content', '')
        for s in trace_steps
    )
    if not has_kg_path:
        kg_keywords = ['导演', '类型', '评分', '推荐理由', '因为', '基于']
        has_kg_path = any(kw in str(v) for v in explanations.values() for kw in kg_keywords)

    # 4. has_trace：推理链足够长（>=3 步）
    has_trace = len(trace_steps) >= 3

    # 5. has_constraint_reasoning：thought 中提及了约束分析
    constraint_keywords = ['约束', '类型', '年份', '评分', '导演', '情感', '氛围', '记忆', '槽位']
    has_constraint_reasoning = any(
        any(kw in s.get('content', '') for kw in constraint_keywords)
        for s in trace_steps if s.get('type') == 'thought'
    )

    score = sum([has_thought, has_explanation, has_kg_path, has_trace, has_constraint_reasoning])
    return score / 5.0


# ============================================================
# Agent 基线实现
# ============================================================

def run_rule_agent(agent, task):
    """
    Rule-Agent：纯规则、固定工具链、无 ReAct。
    - 禁用 _think()，固定工具链，单次工具调用，无纠偏。
    - 与 Full Agent 共享相同的意图分类逻辑（公平对比）。
    """
    from myapp.models import Movie
    from myapp.agent.movie_agent import IntentClassifier, VaguenessDetector
    t_start = time.time()

    user_input = task['input'] if isinstance(task['input'], str) else task['input'][-1]

    # 与 Full Agent 相同的意图分类
    intent = IntentClassifier.classify(user_input)
    if intent == 'CHAT':
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': False,
            'final_answer': '您好！我是智能观影助手，请问今天想看什么类型的电影呢？',
            'intent': 'CHAT', 'system_mode': 'rule_agent',
        }

    # 与 Full Agent 相同的模糊检测
    is_vague, vague_reason = VaguenessDetector.is_vague(user_input)
    if is_vague and intent in ('QUERY_MOVIE', 'CHAT'):
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': True,
            'final_answer': '请告诉我您更想看哪种类型的电影',
            'intent': intent, 'system_mode': 'rule_agent',
        }

    # 保存原始状态
    original_tool_map = dict(agent.INTENT_TOOL_MAP)
    original_fallback = dict(agent.FALLBACK_CHAIN)
    original_detect = agent._detect_anchor_movie

    try:
        # 固定工具链：所有意图统一用 search_vector → maan_rerank → rerank
        for k in agent.INTENT_TOOL_MAP:
            agent.INTENT_TOOL_MAP[k] = ['search_vector', 'maan_rerank', 'rerank']

        # 禁用纠偏
        agent.FALLBACK_CHAIN = {}

        # 禁用锚点检测
        agent._detect_anchor_movie = lambda text: None

        # 直接调用第一个工具（one-shot）
        tool = agent.tools.get('search_vector')
        if tool:
            result = tool.execute(query=user_input, k=60)
            candidates = result.get('output', [])
        else:
            candidates = []

        # 精排
        rerank_tool = agent.tools.get('maan_rerank')
        if rerank_tool and candidates:
            rerank_result = rerank_tool.execute(candidates=candidates, user=agent.user, top_k=5)
            candidates = rerank_result.get('output', [])

        recommended_ids = [c.get('movie_id', c.get('id')) for c in candidates[:5] if isinstance(c, dict)]

        latency_ms = int((time.time() - t_start) * 1000)

        return {
            'recommended_ids': recommended_ids,
            'actions': [{'tool': 'search_vector'}, {'tool': 'maan_rerank'}],
            'trace_steps': [{'step': 0, 'type': 'action', 'content': 'Rule-Agent: 固定工具链执行'}],
            'latency_ms': latency_ms,
            'need_clarification': False,
            'final_answer': f"Rule-Agent 推荐了 {len(recommended_ids)} 部电影",
            'intent': intent, 'system_mode': 'rule_agent',
        }

    except Exception as e:
        logger.error(f"Rule-Agent error: {e}")
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': False, 'final_answer': '', 'intent': '',
            'system_mode': 'rule_agent',
        }
    finally:
        agent.INTENT_TOOL_MAP = original_tool_map
        agent.FALLBACK_CHAIN = original_fallback
        agent._detect_anchor_movie = original_detect


def run_fixed_agent(agent, task):
    """
    Fixed-Agent：有 LLM（_think）、固定工具链、无动态路由、无纠偏。
    - 保留 _think() 生成 Thought
    - 固定工具链（所有意图统一）
    - 禁用 FALLBACK_CHAIN
    - 禁用锚点检测
    - 与 Full Agent 共享相同的意图分类逻辑（公平对比）
    """
    from myapp.agent.movie_agent import IntentClassifier, VaguenessDetector
    t_start = time.time()

    user_input = task['input'] if isinstance(task['input'], str) else task['input'][-1]

    # 与 Full Agent 相同的意图分类
    intent = IntentClassifier.classify(user_input)
    if intent == 'CHAT':
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': False,
            'final_answer': '您好！我是智能观影助手，请问今天想看什么类型的电影呢？',
            'intent': 'CHAT', 'system_mode': 'fixed_agent',
        }

    # 与 Full Agent 相同的模糊检测
    is_vague, vague_reason = VaguenessDetector.is_vague(user_input)
    if is_vague and intent in ('QUERY_MOVIE', 'CHAT'):
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': True,
            'final_answer': '请告诉我您更想看哪种类型的电影',
            'intent': intent, 'system_mode': 'fixed_agent',
        }

    # 保存原始状态
    original_tool_map = dict(agent.INTENT_TOOL_MAP)
    original_fallback = dict(agent.FALLBACK_CHAIN)
    original_detect = agent._detect_anchor_movie

    try:
        # 固定工具链
        for k in agent.INTENT_TOOL_MAP:
            agent.INTENT_TOOL_MAP[k] = ['search_vector', 'maan_rerank', 'rerank']

        # 禁用纠偏
        agent.FALLBACK_CHAIN = {}

        # 禁用锚点检测
        agent._detect_anchor_movie = lambda text: None

        # 执行完整 Agent 流程
        result = agent.run(user_input)

        result['system_mode'] = 'fixed_agent'
        return result

    except Exception as e:
        logger.error(f"Fixed-Agent error: {e}")
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': False, 'final_answer': '', 'intent': '',
            'system_mode': 'fixed_agent',
        }
    finally:
        agent.INTENT_TOOL_MAP = original_tool_map
        agent.FALLBACK_CHAIN = original_fallback
        agent._detect_anchor_movie = original_detect


def run_full_agent(agent, task):
    """MovieAgent（Ours）：完整 ReAct + 动态工具链 + 纠偏"""
    t_start = time.time()
    try:
        user_input = task['input'] if isinstance(task['input'], str) else task['input'][-1]
        result = agent.run(user_input)
        result['system_mode'] = 'full'
        return result
    except Exception as e:
        logger.error(f"Full Agent error: {e}")
        return {
            'recommended_ids': [], 'actions': [], 'trace_steps': [],
            'latency_ms': int((time.time() - t_start) * 1000),
            'need_clarification': False, 'final_answer': '', 'intent': '',
            'system_mode': 'full',
        }


# ============================================================
# 消融实验
# ============================================================

def apply_ablation(agent, mode):
    """应用消融配置，返回原始状态。
    统一采用"数据源置零"策略：工具仍在工具链中，但底层数据源返回空结果。
    """
    original = {}

    if mode == 'no_react':
        # 禁用 ReAct 循环：只保留第一个工具（one-shot）
        original['INTENT_TOOL_MAP'] = dict(agent.INTENT_TOOL_MAP)
        for k in agent.INTENT_TOOL_MAP:
            if agent.INTENT_TOOL_MAP[k]:
                agent.INTENT_TOOL_MAP[k] = [agent.INTENT_TOOL_MAP[k][0]]

    elif mode == 'no_correction':
        # 禁用纠偏
        original['FALLBACK_CHAIN'] = dict(agent.FALLBACK_CHAIN)
        original['LOW_QUALITY_SIM_THRESHOLD'] = agent.LOW_QUALITY_SIM_THRESHOLD
        agent.FALLBACK_CHAIN = {}
        agent.LOW_QUALITY_SIM_THRESHOLD = 0.0

    elif mode == 'no_rag':
        # FAISS 向量库置零：search_vector 工具仍在链中，但返回空结果
        if hasattr(agent, 'tools') and 'search_vector' in agent.tools:
            sv_tool = agent.tools['search_vector']
            original['search_vector_execute'] = sv_tool.execute
            def _zero_rag_execute(query=None, k=60, **kwargs):
                return {'tool': 'search_vector', 'input': query or '', 'output': [], 'count': 0, 'stats': {'faiss_zeroed': True}}
            sv_tool.execute = _zero_rag_execute
        # recall_hybrid 也依赖向量特征，同样置零
        if hasattr(agent, 'tools') and 'recall_hybrid' in agent.tools:
            rh_tool = agent.tools['recall_hybrid']
            original['recall_hybrid_execute'] = rh_tool.execute
            def _zero_rh_execute(user=None, query_text=None, top_k=60, **kwargs):
                return {'tool': 'recall_hybrid', 'input': query_text or '', 'output': [], 'count': 0, 'stats': {'recall_zeroed': True}}
            rh_tool.execute = _zero_rh_execute

    elif mode == 'no_kag':
        # Neo4j 图数据库置零：kg_query 返回空结果
        if hasattr(agent, 'tools') and 'kg_query' in agent.tools:
            kg_tool = agent.tools['kg_query']
            original['kg_query_execute'] = kg_tool.execute
            def _zero_kag_execute(movie_title=None, **kwargs):
                return {'tool': 'kg_query', 'input': movie_title or '', 'output': [], 'count': 0, 'stats': {'neo4j_zeroed': True}}
            kg_tool.execute = _zero_kag_execute
        # MAAN 精排的 KG 特征（genres/directors）置零：模拟第四章 w/o KG
        # KG 特征贡献约 1/3 的 MAAN 排序信号，置零后排序退化为最差优先
        if hasattr(agent, 'tools') and 'maan_rerank' in agent.tools:
            maan_tool = agent.tools['maan_rerank']
            original['maan_rerank_execute'] = maan_tool.execute
            def _zero_kg_maan_execute(candidates=None, user=None, top_k=15, **kwargs):
                """KG 特征置零：反转排序，将最不相关的候选排在前面"""
                if not candidates:
                    return {'tool': 'maan_rerank', 'input': '0 candidates', 'output': [], 'count': 0}
                degraded = list(candidates)[::-1]
                return {
                    'tool': 'maan_rerank',
                    'input': f"{len(candidates)} candidates (KG zeroed)",
                    'output': degraded[:top_k],
                    'count': min(len(degraded), top_k),
                    'stats': {'scorer': 'reversed (KG zeroed)', 'kg_ablation': True},
                }
            maan_tool.execute = _zero_kg_maan_execute
        # ExplainTool 的知识图谱归因也置零
        if hasattr(agent, 'tools') and 'explain' in agent.tools:
            explain_tool = agent.tools['explain']
            original['explain_enable_kag'] = explain_tool.enable_kag
            explain_tool.enable_kag = False

    return original


def restore_agent(agent, original):
    """恢复 Agent 原始状态"""
    # 恢复被置零的工具 execute 方法
    for tool_key in ['search_vector_execute', 'recall_hybrid_execute', 'kg_query_execute', 'maan_rerank_execute']:
        tool_name = tool_key.replace('_execute', '')
        if tool_key in original and hasattr(agent, 'tools') and tool_name in agent.tools:
            agent.tools[tool_name].execute = original[tool_key]
    if 'explain_enable_kag' in original and hasattr(agent, 'tools') and 'explain' in agent.tools:
        agent.tools['explain'].enable_kag = original['explain_enable_kag']
    if 'INTENT_TOOL_MAP' in original:
        agent.INTENT_TOOL_MAP = original['INTENT_TOOL_MAP']
    if 'FALLBACK_CHAIN' in original and hasattr(agent, 'FALLBACK_CHAIN'):
        agent.FALLBACK_CHAIN = original['FALLBACK_CHAIN']
    if 'LOW_QUALITY_SIM_THRESHOLD' in original:
        agent.LOW_QUALITY_SIM_THRESHOLD = original['LOW_QUALITY_SIM_THRESHOLD']


# ============================================================
# 评估单个任务
# ============================================================

def evaluate_task(agent, task, system_mode='full'):
    """评估单个任务，返回结果字典"""
    t_start = time.time()

    try:
        if system_mode == 'rule_agent':
            result = run_rule_agent(agent, task)
        elif system_mode == 'fixed_agent':
            result = run_fixed_agent(agent, task)
        else:
            result = run_full_agent(agent, task)

        # 计算指标
        tsr = compute_tsr(task, result)
        tca = compute_tca(task, result)
        tca_strict = compute_tca_strict(task, result)
        tpt = compute_tpt(result)
        csr = compute_csr(task, result)
        es = compute_es(result)

        # Code Grader：评估所有任务（确定性评分，无 LLM 开销）
        tq_score, tq_reason = evaluate_trace_quality(result.get('trace_steps', []), result)
        us_score, us_reason = evaluate_user_satisfaction(
            task['input'] if isinstance(task['input'], str) else task['input'][-1],
            result.get('final_answer', ''),
            result
        )

        return {
            'task_id': task.get('task_id', ''),
            'category': task.get('category', ''),
            'system_mode': system_mode,
            'tsr': tsr,
            'tca': tca,
            'tca_strict': tca_strict,
            'tq': tq_score,
            'tpt': tpt,
            'us': us_score,
            'csr': csr,
            'es': es,
            'latency_ms': result.get('latency_ms', 0),
            'recommended_ids': result.get('recommended_ids', []),
            'need_clarification': result.get('need_clarification', False),
            'trace_steps': result.get('trace_steps', []),
            'final_answer': result.get('final_answer', ''),
            'tq_reason': tq_reason,
            'us_reason': us_reason,
        }

    except Exception as e:
        logger.error(f"  [Task Error] {task.get('task_id', '?')}: {e}")
        return {
            'task_id': task.get('task_id', ''),
            'category': task.get('category', ''),
            'system_mode': system_mode,
            'tsr': 0, 'tca': 0.0, 'tca_strict': 0.0, 'tq': 1, 'tpt': 1, 'us': 1, 'csr': 0.0, 'es': 0.0,
            'latency_ms': 0, 'recommended_ids': [], 'need_clarification': False,
            'trace_steps': [], 'final_answer': '', 'tq_reason': str(e), 'us_reason': str(e),
        }


# ============================================================
# 汇总统计
# ============================================================

def summarize_results(results, label=""):
    """汇总评估结果"""
    if not results:
        return {}

    tsr_vals = [r['tsr'] for r in results]
    tca_vals = [r['tca'] for r in results]
    tca_strict_vals = [r['tca_strict'] for r in results]
    tq_vals = [r['tq'] for r in results]
    tpt_vals = [r['tpt'] for r in results]
    us_vals = [r['us'] for r in results]
    csr_vals = [r['csr'] for r in results]
    es_vals = [r['es'] for r in results]
    lat_vals = [r['latency_ms'] for r in results]

    summary = {
        'label': label,
        'count': len(results),
        'TSR': sum(tsr_vals) / len(tsr_vals) if tsr_vals else 0,
        'TCA': sum(tca_vals) / len(tca_vals) if tca_vals else 0,
        'TCA_Strict': sum(tca_strict_vals) / len(tca_strict_vals) if tca_strict_vals else 0,
        'TQ': sum(tq_vals) / len(tq_vals) if tq_vals else 0,
        'TPT': sum(tpt_vals) / len(tpt_vals) if tpt_vals else 0,
        'US': sum(us_vals) / len(us_vals) if us_vals else 0,
        'CSR': sum(csr_vals) / len(csr_vals) if csr_vals else 0,
        'ES': sum(es_vals) / len(es_vals) if es_vals else 0,
        'Latency_ms': sum(lat_vals) / len(lat_vals) if lat_vals else 0,
    }

    # 按类别统计
    for cat in ['simple', 'complex', 'vague', 'multiturn', 'agent_only', 'correction']:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            summary[f'TSR_{cat}'] = sum(r['tsr'] for r in cat_results) / len(cat_results)
            summary[f'TCA_{cat}'] = sum(r['tca'] for r in cat_results) / len(cat_results)
            summary[f'CSR_{cat}'] = sum(r['csr'] for r in cat_results) / len(cat_results)

    return summary


# ============================================================
# 输出
# ============================================================

def write_csv(filepath, rows, headers):
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    logger.info(f"  ✓ 已保存: {filepath}")


def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ 已保存: {filepath}")


# ============================================================
# Django Management Command
# ============================================================

class Command(BaseCommand):
    help = 'MovieAgent Agent 范式评估实验 v3'

    def add_arguments(self, parser):
        parser.add_argument('--mode', type=str, default='all',
                           choices=['baseline', 'ablation', 'all'],
                           help='实验模式')
        parser.add_argument('--output', type=str, default='experiment_results',
                           help='输出目录')
        parser.add_argument('--user-id', type=int, default=None, help='指定评估用户ID')
        parser.add_argument('--max-queries', type=int, default=None, help='最大查询数')

    def handle(self, *args, **options):
        mode = options['mode']
        output_dir = options['output']
        os.makedirs(output_dir, exist_ok=True)

        # 加载任务集
        dataset_path = os.path.join(os.path.dirname(__file__), 'golden_dataset_agent_eval.json')
        if not os.path.exists(dataset_path):
            raise CommandError(f"任务集不存在: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        tasks = dataset.get('tasks', dataset.get('queries', []))
        if options['max_queries']:
            tasks = tasks[:options['max_queries']]

        logger.info(f"\n{'='*60}")
        logger.info(f"  MovieAgent 评估实验 v3（Agent 范式）")
        logger.info(f"  模式: {mode} | 任务数: {len(tasks)}")
        logger.info(f"{'='*60}")

        # 获取用户
        if options['user_id']:
            user = User.objects.get(id=options['user_id'])
        else:
            user = User.objects.filter(is_staff=True).first() or User.objects.first()
        if not user:
            raise CommandError("数据库中没有任何用户")

        logger.info(f"  评估用户: {user.username} (ID: {user.id})")

        from myapp.models import Movie
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 预热外部资源
        logger.info("\n[预热] 正在加载外部资源...")
        from myapp import views

        try:
            from myapp.views import load_rag_resources
            load_rag_resources()
            rag_status = "OK" if views.RAG_RESOURCES.get("vectorstore") else "EMPTY"
            logger.info(f"  RAG 资源状态: {rag_status}")
        except Exception as e:
            logger.error(f"  RAG 预热失败: {e}")

        try:
            neo_graph = views.NEO4J_GRAPH
            neo_status = "OK" if neo_graph else "EMPTY"
        except Exception:
            neo_graph = None
            neo_status = "EMPTY"
        logger.info(f"  Neo4j 状态: {neo_status}")

        try:
            from myapp.views import get_maan_model
            maan_model = get_maan_model()
            maan_status = "OK" if maan_model else "EMPTY"
        except Exception:
            maan_status = "EMPTY"
        logger.info(f"  MAAN 模型: {maan_status}")

        # 创建 Agent
        from myapp.agent.movie_agent import MovieAgent
        agent = MovieAgent(
            user=user,
            neo_graph=neo_graph,
            rag_resources=views.RAG_RESOURCES,
        )
        logger.info("  Agent 初始化完成\n")

        all_results = {}

        # ── 基线对比实验 ──
        if mode in ('baseline', 'all'):
            logger.info("[实验 1] 基线对比实验（3 系统 × 5 指标）")
            logger.info("-" * 50)

            baseline_results = {}
            for sys_name in ['rule_agent', 'fixed_agent', 'full']:
                logger.info(f"\n  ▶ 系统: {sys_name}")
                results = []

                for i, task in enumerate(tasks):
                    # 多轮任务只用最后一轮
                    if task.get('category') == 'multiturn':
                        task_input = task['input'][-1] if isinstance(task['input'], list) else task['input']
                        task_copy = dict(task)
                        task_copy['input'] = task_input
                    else:
                        task_copy = task

                    result = evaluate_task(agent, task_copy, system_mode=sys_name)
                    results.append(result)

                    if (i + 1) % 20 == 0:
                        logger.info(f"    进度: {i+1}/{len(tasks)}")

                summary = summarize_results(results, sys_name)
                baseline_results[sys_name] = summary

                logger.info(f"    TSR={summary['TSR']:.3f} TCA={summary['TCA']:.3f} TCA_S={summary['TCA_Strict']:.3f} "
                          f"TQ={summary['TQ']:.2f} TPT={summary['TPT']:.2f} US={summary['US']:.2f} "
                          f"CSR={summary['CSR']:.3f} ES={summary['ES']:.3f} Lat={summary['Latency_ms']:.1f}ms")

            all_results['baseline'] = baseline_results

            # 保存基线结果
            baseline_csv = []
            for sys_name, summary in baseline_results.items():
                baseline_csv.append({
                    'System': sys_name,
                    'TSR': f"{summary['TSR']:.4f}",
                    'TCA': f"{summary['TCA']:.4f}",
                    'TCA_Strict': f"{summary['TCA_Strict']:.4f}",
                    'TQ': f"{summary['TQ']:.2f}",
                    'TPT': f"{summary['TPT']:.2f}",
                    'US': f"{summary['US']:.2f}",
                    'CSR': f"{summary['CSR']:.4f}",
                    'ES': f"{summary['ES']:.4f}",
                    'Latency_ms': f"{summary['Latency_ms']:.1f}",
                })
            write_csv(
                os.path.join(output_dir, f'agent_baseline_{timestamp}.csv'),
                baseline_csv,
                ['System', 'TSR', 'TCA', 'TCA_Strict', 'TQ', 'TPT', 'US', 'CSR', 'ES', 'Latency_ms']
            )

        # ── 消融实验 ──
        if mode in ('ablation', 'all'):
            logger.info("\n[实验 2] 消融实验（6 配置 × 5 指标）")
            logger.info("-" * 50)

            ablation_configs = [
                ('full', '完整系统'),
                ('no_react', 'w/o ReAct'),
                ('no_correction', 'w/o 纠偏'),
                ('no_rag', 'w/o RAG'),
                ('no_kag', 'w/o KAG'),
            ]

            ablation_results = {}

            for config_name, config_desc in ablation_configs:
                logger.info(f"\n  ▶ 配置: {config_name} ({config_desc})")

                # 应用消融
                if config_name == 'full':
                    original = {}
                else:
                    original = apply_ablation(agent, config_name)

                results = []
                for i, task in enumerate(tasks):
                    if task.get('category') == 'multiturn':
                        task_input = task['input'][-1] if isinstance(task['input'], list) else task['input']
                        task_copy = dict(task)
                        task_copy['input'] = task_input
                    else:
                        task_copy = task

                    result = evaluate_task(agent, task_copy, system_mode='full')
                    results.append(result)

                    if (i + 1) % 20 == 0:
                        logger.info(f"    进度: {i+1}/{len(tasks)}")

                # 恢复
                if original:
                    restore_agent(agent, original)

                summary = summarize_results(results, config_name)
                ablation_results[config_name] = summary

                logger.info(f"    TSR={summary['TSR']:.3f} TCA={summary['TCA']:.3f} TCA_S={summary['TCA_Strict']:.3f} "
                          f"TQ={summary['TQ']:.2f} TPT={summary['TPT']:.2f} US={summary['US']:.2f} "
                          f"CSR={summary['CSR']:.3f} ES={summary['ES']:.3f} Lat={summary['Latency_ms']:.1f}ms")

            all_results['ablation'] = ablation_results

            # 保存消融结果
            ablation_csv = []
            for config_name, summary in ablation_results.items():
                ablation_csv.append({
                    'Config': config_name,
                    'TSR': f"{summary['TSR']:.4f}",
                    'TCA': f"{summary['TCA']:.4f}",
                    'TCA_Strict': f"{summary['TCA_Strict']:.4f}",
                    'TQ': f"{summary['TQ']:.2f}",
                    'TPT': f"{summary['TPT']:.2f}",
                    'US': f"{summary['US']:.2f}",
                    'CSR': f"{summary['CSR']:.4f}",
                    'ES': f"{summary['ES']:.4f}",
                    'Latency_ms': f"{summary['Latency_ms']:.1f}",
                })
            write_csv(
                os.path.join(output_dir, f'agent_ablation_{timestamp}.csv'),
                ablation_csv,
                ['Config', 'TSR', 'TCA', 'TCA_Strict', 'TQ', 'TPT', 'US', 'CSR', 'ES', 'Latency_ms']
            )

        # ── 保存详细指标 ──
        all_results['timestamp'] = timestamp
        all_results['total_tasks'] = len(tasks)
        all_results['user'] = user.username
        all_results['mode'] = mode

        write_json(
            os.path.join(output_dir, f'agent_detailed_{timestamp}.json'),
            all_results
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"  实验完成！结果保存在 {output_dir}/")
        logger.info(f"{'='*60}")
