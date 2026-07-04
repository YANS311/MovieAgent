"""
可解释性增强模块 (XAI - Explainable AI)
================================================
三个维度的可解释性升级：
  1. 拟人化思考流：将 ReAct 机器步骤翻译为用户友好的白话文
  2. 多维归因雷达：结构化量化归因数据（供前端雷达图渲染）
  3. 推理健康度诊断：Agent 运行质量监控（幻觉风险、工具效率）

使用方式：
    from myapp.utils.xai_explainer import (
        translate_react_to_human,
        build_attribution_radar,
        analyze_trace_health
    )
================================================
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger('movie_agent')


# =============================================================
# 方案一：拟人化思考流翻译器
# =============================================================

# 工具名 → 拟人化描述映射
_TOOL_HUMAN_MAP = {
    'search_vector': '正在语义库中为您寻找风格相近的影片...',
    'recall_hybrid': '正在综合您的观影口味和历史记录进行智能匹配...',
    'kg_query': '正在知识图谱中追溯关联作品和导演风格...',
    'maan_rerank': '正在用深度多模态模型对候选影片进行精排...',
    'rerank': '正在根据多样性和业务规则优化推荐列表...',
    'explain': '正在为每部推荐影片撰写个性化推荐理由...',
}

# 思考阶段关键词 → 拟人化描述
_THOUGHT_PATTERNS = [
    (r'锚点电影.*?《(.+?)》', r'发现您对《\1》评价很高，以此为参考起点'),
    (r'意图.*?QUERY_MOVIE', '理解了您正在寻找好电影推荐'),
    (r'意图.*?QUERY_PROFILE_REC', '理解了您希望基于个人画像获得推荐'),
    (r'意图.*?QUERY_RANK', '理解了您想看看热门榜单'),
    (r'意图.*?QUERY_NEW', '理解了您想了解最新影片'),
    (r'情感约束.*?不要太压抑', '注意到您偏好轻松的观影氛围'),
    (r'年份约束', '留意到您对上映年份有特定要求'),
    (r'追问模式', '发现您想在之前推荐的基础上进一步筛选'),
    (r'自反馈纠偏.*?工具.*?返回空结果', '初次检索未找到匹配结果，正在切换备用策略重新搜索'),
    (r'纠偏成功', '备用策略找到了合适的候选影片'),
    (r'冷启动', '您是新用户，正在为您推荐广受好评的影片'),
]


def translate_react_to_human(react_trace: Dict) -> List[Dict]:
    """
    将 ReAct 推理链翻译为用户友好的"拟人化思考流"。
    
    Args:
        react_trace: 原始 react_trace 字典，含 thought, actions, observations
    
    Returns:
        list of dict: [
            {"step": 1, "type": "thinking", "icon": "🧠", "text": "..."},
            {"step": 2, "type": "searching", "icon": "🔍", "text": "..."},
            {"step": 3, "type": "found", "icon": "✅", "text": "..."},
            {"step": 4, "type": "complete", "icon": "🎬", "text": "..."},
        ]
    """
    if not react_trace:
        return []
    
    human_steps = []
    step_num = 0
    
    # ── Step 1: 理解需求 ──
    thought = react_trace.get('thought', '')
    thought_text = _translate_thought(thought)
    if thought_text:
        step_num += 1
        human_steps.append({
            'step': step_num,
            'type': 'thinking',
            'icon': '🧠',
            'text': thought_text,
        })
    
    # ── Step 2-N: 工具调用过程 ──
    actions = react_trace.get('actions', [])
    observations = react_trace.get('observations', [])
    
    for i, action in enumerate(actions):
        tool_name = action.get('tool', '')
        
        # 翻译工具调用为拟人化描述
        tool_text = _translate_tool_call(tool_name, action, observations, i)
        
        if tool_text:
            step_num += 1
            # 判断是搜索中还是已完成
            obs = observations[i] if i < len(observations) else {}
            obs_count = obs.get('count', 0)
            
            if obs_count > 0:
                human_steps.append({
                    'step': step_num,
                    'type': 'found',
                    'icon': '✅',
                    'text': f"{tool_text}（找到 {obs_count} 条线索）",
                })
            else:
                human_steps.append({
                    'step': step_num,
                    'type': 'searching',
                    'icon': '🔍',
                    'text': tool_text,
                })
    
    # ── 最终步骤：生成推荐 ──
    step_num += 1
    human_steps.append({
        'step': step_num,
        'type': 'complete',
        'icon': '🎬',
        'text': '综合以上分析，为您精心挑选了以下推荐',
    })
    
    return human_steps


def _translate_thought(thought: str) -> str:
    """将原始 Thought 翻译为拟人化描述"""
    if not thought:
        return ""
    
    # 尝试模式匹配
    for pattern, replacement in _THOUGHT_PATTERNS:
        match = re.search(pattern, thought)
        if match:
            return replacement if isinstance(replacement, str) and '\\' not in replacement \
                else match.expand(replacement)
    
    # 提取意图标签
    intent_match = re.search(r'识别用户意图:\s*(\w+)', thought)
    if intent_match:
        intent = intent_match.group(1)
        intent_map = {
            'QUERY_MOVIE': '理解了您正在寻找好电影推荐',
            'QUERY_PROFILE_REC': '理解了您希望基于个人画像获得推荐',
            'QUERY_RANK': '理解了您想看看热门榜单',
            'QUERY_NEW': '理解了您想了解最新影片',
            'QUERY_COMPARISON': '理解了您想对比几部电影',
        }
        return intent_map.get(intent, '正在分析您的需求...')
    
    return '正在理解您的需求...'


def _translate_tool_call(tool_name: str, action: Dict, observations: List, idx: int) -> str:
    """将单个工具调用翻译为拟人化描述"""
    base_text = _TOOL_HUMAN_MAP.get(tool_name, f'正在调用{tool_name}进行分析...')
    
    # 从 action 的 input 中提取上下文信息
    action_input = action.get('input', '')
    
    # 如果是知识图谱查询，尝试提取电影名
    if tool_name == 'kg_query' and action_input:
        movie_match = re.search(r'《(.+?)》', action_input)
        if movie_match:
            return f"正在知识图谱中为您追溯《{movie_match.group(1)}》的关联作品和导演风格..."
    
    return base_text


# =============================================================
# 方案二：多维归因雷达构建器
# =============================================================

def build_attribution_radar(user, movie_id: int) -> Dict[str, Any]:
    """
    构建推荐归因雷达数据（供前端 ECharts 雷达图渲染）。
    
    返回四个维度的量化归因：
    1. semantic_match   — 语义/标签匹配度
    2. graph_path       — 知识图谱路径强度
    3. cf_weight        — 协同过滤/大众评分权重
    4. historical_anchor — 历史锚点触发强度
    
    Returns:
        dict: {
            "radar_data": {"indicators": [...], "values": [...]},
            "attribution_details": [...],
            "primary_anchor": {"title": "...", "movie_id": ..., "reason": "..."},
            "confidence_score": float
        }
    """
    from myapp.models import Movie, UserRating
    from myapp.recommender.explain import (
        _find_best_anchor, _get_visual_embedding, _cosine_similarity,
        _generate_natural_reason, _find_common_elements
    )
    
    target = Movie.objects.filter(id=movie_id).prefetch_related(
        'genres', 'actors', 'directors'
    ).first()
    
    if not target:
        return _empty_radar()
    
    # 获取用户历史高分电影
    history = list(UserRating.objects.filter(
        user=user, score__gte=7.5
    ).select_related('movie').prefetch_related(
        'movie__genres', 'movie__directors', 'movie__actors'
    ).order_by('-comment_time')[:20])
    
    if not history:
        return _cold_start_radar(target)
    
    # ── 维度 1：语义匹配度 ──
    target_genres = set(g.name for g in target.genres.all())
    target_directors = set(d.name for d in target.directors.all())
    
    # 统计用户历史中类型/导演的重合度
    genre_hits = 0
    director_hits = 0
    total_checked = 0
    
    for h in history[:10]:
        h_genres = set(g.name for g in h.movie.genres.all())
        h_directors = set(d.name for d in h.movie.directors.all())
        
        if target_genres & h_genres:
            genre_hits += 1
        if target_directors & h_directors:
            director_hits += 1
        total_checked += 1
    
    semantic_score = min(1.0, (genre_hits + director_hits * 2) / max(total_checked, 1))
    
    # ── 维度 2：知识图谱路径强度 ──
    graph_score = 0.0
    graph_evidence = []
    
    try:
        from myapp import views
        neo_g = getattr(views, 'neo_graph', None)
        if neo_g:
            # 检查是否有导演关联
            dir_res = neo_g.run(
                "MATCH (m:Movie {mid: $mid})<-[:DIRECTED_BY]-(d:Person) "
                "RETURN d.name AS name LIMIT 3",
                mid=target.id
            ).data()
            if dir_res:
                graph_score += 0.4
                graph_evidence.append(f"导演节点: {', '.join(r['name'] for r in dir_res)}")
            
            # 检查是否有类型关联
            gen_res = neo_g.run(
                "MATCH (m:Movie {mid: $mid})-[:BELONGS_TO]->(g:Genre) "
                "RETURN g.name AS name LIMIT 3",
                mid=target.id
            ).data()
            if gen_res:
                graph_score += 0.3
                graph_evidence.append(f"类型节点: {', '.join(r['name'] for r in gen_res)}")
            
            # 检查是否有演员关联
            act_res = neo_g.run(
                "MATCH (m:Movie {mid: $mid})<-[:ACTED_IN]-(a:Person) "
                "RETURN a.name AS name LIMIT 3",
                mid=target.id
            ).data()
            if act_res:
                graph_score += 0.3
                graph_evidence.append(f"演员节点: {', '.join(r['name'] for r in act_res)}")
    except Exception:
        pass
    
    graph_score = min(1.0, graph_score)
    
    # ── 维度 3：协同过滤/大众评分 ──
    cf_score = 0.0
    if target.score:
        cf_score = min(1.0, target.score / 10.0)
    if target.vote_count:
        cf_score = min(1.0, cf_score * (1 + min(target.vote_count / 10000, 0.5)))
    
    # ── 维度 4：历史锚点触发强度 ──
    best_anchor = None
    anchor_score = 0.0
    anchor_reason = ""
    
    if history:
        best_anchor_obj, anchor_type, strength = _find_best_anchor(target, history)
        if best_anchor_obj:
            best_anchor = best_anchor_obj
            anchor_score = min(1.0, strength / 5.5)  # 归一化（最大理论分5.5）
            
            common = _find_common_elements(target, best_anchor_obj)
            if common.get('directors'):
                anchor_reason = f"同导演 {', '.join(common['directors'])}"
            elif common.get('actors'):
                anchor_reason = f"同主演 {', '.join(common['actors'])}"
            elif common.get('genres'):
                anchor_reason = f"同类型 {', '.join(common['genres'])}"
            else:
                anchor_reason = "视觉风格与情感基调相似"
    
    # ── 构建归因详情 ──
    attribution_details = []
    
    if genre_hits > 0:
        common_g = target_genres & set(g.name for h in history[:5] for g in h.movie.genres.all())
        attribution_details.append({
            'dimension': '类型匹配',
            'score': round(semantic_score, 2),
            'evidence': f"与您偏好的{'、'.join(list(common_g)[:2])}类型高度契合",
            'icon': '🎭',
        })
    
    if graph_evidence:
        attribution_details.append({
            'dimension': '知识图谱',
            'score': round(graph_score, 2),
            'evidence': '；'.join(graph_evidence[:2]),
            'icon': '🔗',
        })
    
    if target.score:
        attribution_details.append({
            'dimension': '大众口碑',
            'score': round(cf_score, 2),
            'evidence': f"评分 {target.score} 分{'，' + str(target.vote_count) + '人评价' if target.vote_count else ''}",
            'icon': '⭐',
        })
    
    if best_anchor:
        attribution_details.append({
            'dimension': '历史关联',
            'score': round(anchor_score, 2),
            'evidence': f"与您高分评价的《{best_anchor.title}》{anchor_reason}",
            'icon': '🔗',
        })
    
    # ── 置信度分数 ──
    confidence = round(
        semantic_score * 0.3 + graph_score * 0.25 + cf_score * 0.2 + anchor_score * 0.25,
        2
    )
    
    return {
        'radar_data': {
            'indicators': [
                {'name': '语义匹配', 'max': 1.0},
                {'name': '图谱关联', 'max': 1.0},
                {'name': '大众口碑', 'max': 1.0},
                {'name': '历史锚点', 'max': 1.0},
            ],
            'values': [
                round(semantic_score, 2),
                round(graph_score, 2),
                round(cf_score, 2),
                round(anchor_score, 2),
            ],
        },
        'attribution_details': attribution_details,
        'primary_anchor': {
            'title': best_anchor.title if best_anchor else '',
            'movie_id': best_anchor.id if best_anchor else None,
            'reason': anchor_reason,
        },
        'confidence_score': confidence,
    }


def _empty_radar():
    """空归因数据"""
    return {
        'radar_data': {
            'indicators': [
                {'name': '语义匹配', 'max': 1.0},
                {'name': '图谱关联', 'max': 1.0},
                {'name': '大众口碑', 'max': 1.0},
                {'name': '历史锚点', 'max': 1.0},
            ],
            'values': [0, 0, 0, 0],
        },
        'attribution_details': [],
        'primary_anchor': None,
        'confidence_score': 0,
    }


def _cold_start_radar(target):
    """冷启动归因数据（只有大众口碑维度）"""
    score = min(1.0, (target.score or 5.0) / 10.0)
    return {
        'radar_data': {
            'indicators': [
                {'name': '语义匹配', 'max': 1.0},
                {'name': '图谱关联', 'max': 1.0},
                {'name': '大众口碑', 'max': 1.0},
                {'name': '历史锚点', 'max': 1.0},
            ],
            'values': [0, 0, round(score, 2), 0],
        },
        'attribution_details': [{
            'dimension': '大众口碑',
            'score': round(score, 2),
            'evidence': f"评分 {target.score} 分，优质影片",
            'icon': '⭐',
        }],
        'primary_anchor': None,
        'confidence_score': round(score * 0.2, 2),
    }


# =============================================================
# 方案三：推理健康度诊断器
# =============================================================

def analyze_trace_health(trace) -> Dict[str, Any]:
    """
    分析 AgentTrace 的运行健康度。
    
    Args:
        trace: AgentTrace 数据库对象
    
    Returns:
        dict: {
            "tool_efficiency": float,      # 工具调用效率（结果数/调用次数）
            "hallucination_risk": str,     # 幻觉风险等级: low/medium/high
            "reasoning_depth": int,        # 推理深度（ReAct 循环轮次）
            "health_score": float,         # 综合健康分 0-1
            "diagnostics": [...],          # 诊断详情列表
        }
    """
    diagnostics = []
    
    # ── 1. 工具调用效率 ──
    actions = trace.actions if isinstance(trace.actions, list) else []
    observations = trace.observations if isinstance(trace.observations, list) else []
    recommended = trace.recommended_movies if isinstance(trace.recommended_movies, list) else []
    
    tool_count = len(actions)
    result_count = len(recommended)
    
    if tool_count > 0:
        tool_efficiency = min(1.0, result_count / tool_count)
    else:
        tool_efficiency = 0.0
    
    # 效率诊断
    if tool_count > 5:
        diagnostics.append({
            'level': 'warning',
            'message': f'工具调用次数较多（{tool_count}次），可能存在冗余调用',
            'icon': '⚠️',
        })
    
    # ── 2. 幻觉风险评估 ──
    hallucination_risk = 'low'
    thought = trace.thought or ''
    final_answer = trace.final_answer or ''
    
    # 提取 Thought 和 Final Answer 中的电影名
    thought_movies = set(re.findall(r'《(.+?)》', thought))
    answer_movies = set(re.findall(r'《(.+?)》', final_answer))
    
    # 从 observations 中提取出现过的电影名
    obs_text = str(observations)
    obs_movies = set(re.findall(r'《(.+?)》', obs_text))
    obs_movies.update(str(mid) for mid in recommended)
    
    # 如果最终回答中提到了 Thought 中出现但 observations 中未出现的电影
    phantom_movies = thought_movies - obs_movies
    answer_phantom = answer_movies - obs_movies - thought_movies
    
    if answer_phantom:
        hallucination_risk = 'high'
        diagnostics.append({
            'level': 'error',
            'message': f'最终回答中出现了未经验证的电影：{"、".join(list(answer_phantom)[:3])}',
            'icon': '🚨',
        })
    elif phantom_movies:
        hallucination_risk = 'medium'
        diagnostics.append({
            'level': 'warning',
            'message': f'Thought 中提及但未在 Observation 中验证的电影：{"、".join(list(phantom_movies)[:3])}',
            'icon': '⚠️',
        })
    else:
        diagnostics.append({
            'level': 'success',
            'message': '所有推荐影片均可追溯到检索结果',
            'icon': '✅',
        })
    
    # ── 3. 推理深度 ──
    reasoning_depth = max(1, len(actions))
    
    if reasoning_depth == 1:
        diagnostics.append({
            'level': 'info',
            'message': '单轮推理，直接命中',
            'icon': '⚡',
        })
    elif reasoning_depth <= 3:
        diagnostics.append({
            'level': 'success',
            'message': f'{reasoning_depth}轮推理，流程正常',
            'icon': '✅',
        })
    else:
        diagnostics.append({
            'level': 'warning',
            'message': f'{reasoning_depth}轮推理，可能存在路径回溯',
            'icon': '⚠️',
        })
    
    # ── 4. 响应时间评估 ──
    latency = trace.total_latency_ms or 0
    if latency > 10000:
        diagnostics.append({
            'level': 'warning',
            'message': f'响应时间 {latency}ms 超过 10 秒，建议优化',
            'icon': '⏱️',
        })
    elif latency > 5000:
        diagnostics.append({
            'level': 'info',
            'message': f'响应时间 {latency}ms，可接受',
            'icon': '⏱️',
        })
    else:
        diagnostics.append({
            'level': 'success',
            'message': f'响应时间 {latency}ms，表现优秀',
            'icon': '⚡',
        })
    
    # ── 5. 综合健康分 ──
    efficiency_score = tool_efficiency
    hallucination_score = {'low': 1.0, 'medium': 0.6, 'high': 0.2}[hallucination_risk]
    depth_score = 1.0 if reasoning_depth <= 3 else 0.7 if reasoning_depth <= 5 else 0.4
    latency_score = 1.0 if latency < 5000 else 0.7 if latency < 10000 else 0.4
    
    health_score = round(
        efficiency_score * 0.25 +
        hallucination_score * 0.35 +
        depth_score * 0.2 +
        latency_score * 0.2,
        2
    )
    
    # 健康等级
    if health_score >= 0.8:
        health_grade = 'A'
        health_label = '优秀'
    elif health_score >= 0.6:
        health_grade = 'B'
        health_label = '良好'
    elif health_score >= 0.4:
        health_grade = 'C'
        health_label = '一般'
    else:
        health_grade = 'D'
        health_label = '需优化'
    
    return {
        'tool_efficiency': round(tool_efficiency, 2),
        'hallucination_risk': hallucination_risk,
        'reasoning_depth': reasoning_depth,
        'health_score': health_score,
        'health_grade': health_grade,
        'health_label': health_label,
        'diagnostics': diagnostics,
        'metrics': {
            'tool_calls': tool_count,
            'results': result_count,
            'latency_ms': latency,
        },
    }