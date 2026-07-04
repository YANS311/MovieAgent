"""
Rule Agent — 纯规则基线
================================================
实现一个纯规则驱动的推荐 Agent，
用于与 MovieAgent 和 WorkflowAgent 进行对比实验。

核心特征（Rule Agent 的局限性）：
  1. 纯关键词匹配：无语义理解能力
  2. 固定规则：不可动态调整策略
  3. 无记忆：不融合用户历史
  4. 无推理：无 Thought/Reflection 阶段

对比目的：
  证明 MovieAgent 的 RAG + ReAct 能力优于纯规则方案
================================================
"""

import re
import time
from typing import Dict, List, Optional


class RuleAgent:
    """
    纯规则推荐 Agent（最简 Baseline）

    推理流程（纯规则，无 LLM、无 RAG、无 ReAct）：
        用户输入 → 关键词提取 → 数据库查询 → 返回结果
    """

    # 关键词到类型的映射
    GENRE_KEYWORDS = {
        '科幻': '科幻', 'sci-fi': '科幻', 'science': '科幻',
        '喜剧': '喜剧', '搞笑': '喜剧', '幽默': '喜剧',
        '悬疑': '悬疑', '推理': '悬疑', '烧脑': '悬疑',
        '动画': '动画', '动漫': '动画',
        '动作': '动作', '打斗': '动作',
        '爱情': '爱情', '浪漫': '爱情', '爱情片': '爱情',
        '恐怖': '恐怖', '惊悚': '恐怖', '吓人': '恐怖',
        '战争': '战争', '军事': '战争',
        '犯罪': '犯罪', '警匪': '犯罪', '黑帮': '犯罪',
        '奇幻': '奇幻', '魔幻': '奇幻',
        '纪录片': '纪录片', '记录片': '纪录片',
    }

    # 关键词到导演的映射
    DIRECTOR_KEYWORDS = {
        '诺兰': '诺兰', '克里斯托弗·诺兰': '诺兰',
        '宫崎骏': '宫崎骏',
        '昆汀': '昆汀', '昆汀·塔伦蒂诺': '昆汀',
        '斯皮尔伯格': '斯皮尔伯格', '史蒂文·斯皮尔伯格': '斯皮尔伯格',
        '周星驰': '周星驰',
        '王家卫': '王家卫',
        '李安': '李安',
        '张艺谋': '张艺谋',
    }

    def __init__(self, user=None, **kwargs):
        self.user = user

    def run(self, user_input: str) -> dict:
        """
        执行纯规则推荐
        """
        t_start = time.time()

        # Step 1: 关键词提取
        genre = self._extract_genre(user_input)
        director = self._extract_director(user_input)
        year_min = self._extract_year(user_input)
        score_min = self._extract_score(user_input)

        # Step 2: 构建数据库查询
        from myapp.models import Movie
        qs = Movie.objects.all()

        if genre:
            qs = qs.filter(genres__name__icontains=genre)
        if director:
            qs = qs.filter(directors__name__icontains=director)
        if year_min:
            qs = qs.filter(date__year__gte=year_min)
        if score_min:
            qs = qs.filter(score__gte=score_min)

        # Step 3: 按评分排序，取 Top-5
        movies = qs.order_by('-score', '-vote_count').distinct()[:5]
        recommended_ids = list(movies.values_list('id', flat=True))

        # Step 4: 生成简单回复
        if recommended_ids:
            titles = [m.title for m in movies]
            final_answer = f"为您推荐：{'、'.join(titles)}"
        else:
            final_answer = "抱歉，没有找到符合条件的电影。"

        # Step 5: 生成简单推荐理由
        explanations = {}
        for mid in recommended_ids:
            movie = Movie.objects.filter(id=mid).first()
            if movie:
                explanations[mid] = f"推荐《{movie.title}》——评分{movie.score}分"

        t_total = int((time.time() - t_start) * 1000)

        thought = f"【规则提取】类型={genre}, 导演={director}, 年份≥{year_min}, 评分≥{score_min}"

        return {
            'intent': 'QUERY_MOVIE',
            'thought': thought,
            'actions': [{'tool': 'database_query', 'input': user_input}],
            'observations': [{'tool': 'database_query', 'output': [], 'count': len(recommended_ids)}],
            'final_answer': final_answer,
            'recommended_ids': recommended_ids,
            'explanations': explanations,
            'latency_ms': t_total,
            'need_clarification': False,
            'clarification_options': [],
            'trace_steps': [{'step': 0, 'type': 'thought', 'content': thought}],
        }

    def _extract_genre(self, text):
        for keyword, genre in self.GENRE_KEYWORDS.items():
            if keyword in text:
                return genre
        return None

    def _extract_director(self, text):
        for keyword, director in self.DIRECTOR_KEYWORDS.items():
            if keyword in text:
                return director
        return None

    def _extract_year(self, text):
        match = re.search(r'(\d{4})\s*年[以后以来]', text)
        if match:
            return int(match.group(1))
        match = re.search(r'近(\d+)年', text)
        if match:
            return 2026 - int(match.group(1))
        return None

    def _extract_score(self, text):
        match = re.search(r'(\d(?:\.\d)?)\s*分[以上]', text)
        if match:
            return float(match.group(1))
        return None
