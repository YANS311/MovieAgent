"""
LLM 输出结构化解析器
================================================
替代原 views.py 中脆弱的正则提取逻辑，
使用 JSON Mode 强制 LLM 输出标准格式，正则作兜底。

升级动机（方案二）：
  - 原实现：re.findall(r'《(.*?)》\(ID:(\d+)\)', content) → 模型幻觉导致格式崩溃
  - 新实现：JSON Mode + StructuredOutputParser + 正则兜底

使用方式：
    from myapp.utils.llm_output_parser import parse_recommendations
    results = parse_recommendations(llm_output_text)
    # results = [{"movie_id": 123, "title": "信条", "reason": "..."}]
================================================
"""

import re
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('movie_agent')


# =============================================================
# 方案 A：JSON 模式 Prompt 构建（用于强制 LLM 输出 JSON）
# =============================================================

JSON_OUTPUT_INSTRUCTION = """
【输出格式要求 — 必须严格遵守】
请以如下 JSON 格式输出你的推荐结果，不要输出任何 JSON 之外的文字：

```json
{
  "recommendations": [
    {
      "movie_id": 123,
      "title": "电影名",
      "reason": "推荐理由（50字以内）"
    }
  ]
}
```

⚠️ 注意：
1. movie_id 必须是向量资料或图谱路径中出现的真实 ID
2. 如果资料不足以推荐，recommendations 可以为 []
3. 推荐理由必须自然、有温度，严禁机械词汇
"""


def get_json_output_prompt(base_prompt: str) -> str:
    """
    在原始 Prompt 基础上追加 JSON 输出约束。
    
    Args:
        base_prompt: 原始推荐 Prompt
    
    Returns:
        str: 追加了 JSON 约束的完整 Prompt
    """
    return f"{base_prompt}\n\n{JSON_OUTPUT_INSTRUCTION}"


# =============================================================
# 方案 B：结构化解析器（JSON 优先 + 正则兜底）
# =============================================================

def parse_recommendations(text: str) -> List[Dict]:
    """
    从 LLM 输出中提取结构化推荐结果。
    
    解析优先级：
    1. JSON 解析（最可靠）
    2. 正则兜底（应对模型未遵守 JSON 格式的情况）
    
    Args:
        text: LLM 的原始输出文本
    
    Returns:
        list of dict: [{"movie_id": int, "title": str, "reason": str}]
    """
    if not text:
        return []
    
    # ── 策略 1：JSON 解析 ──
    results = _try_json_parse(text)
    if results:
        logger.info(f"[Parser] JSON 解析成功，提取到 {len(results)} 条推荐")
        return results
    
    # ── 策略 2：正则兜底 ──
    results = _regex_fallback(text)
    if results:
        logger.info(f"[Parser] JSON 解析失败，正则兜底提取到 {len(results)} 条推荐")
        return results
    
    logger.warning(f"[Parser] JSON 和正则均未提取到推荐结果")
    return []


def _try_json_parse(text: str) -> List[Dict]:
    """
    尝试从文本中提取 JSON 并解析。
    支持：完整 JSON、```json``` 包裹、部分 JSON。
    """
    # 1. 尝试直接解析
    results = _extract_json_from_text(text)
    if results:
        return results
    
    # 2. 尝试从 markdown 代码块中提取
    json_patterns = [
        r'```json\s*\n?(.*?)\n?\s*```',  # ```json ... ```
        r'```\s*\n?(.*?)\n?\s*```',       # ``` ... ```
        r'\{[^{}]*"recommendations"[^{}]*\[.*?\][^{}]*\}',  # 内联 JSON
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return _extract_json_from_text(json_str)
            except Exception:
                continue
    
    return []


def _extract_json_from_text(json_str: str) -> List[Dict]:
    """
    从 JSON 字符串中提取 recommendations 列表。
    """
    try:
        data = json.loads(json_str.strip())
        
        # 处理 {"recommendations": [...]} 格式
        if isinstance(data, dict) and 'recommendations' in data:
            recs = data['recommendations']
            return _validate_recommendations(recs)
        
        # 处理直接是列表的情况
        if isinstance(data, list):
            return _validate_recommendations(data)
            
    except json.JSONDecodeError:
        pass
    
    return []


def _validate_recommendations(recs) -> List[Dict]:
    """
    验证推荐列表的格式，过滤无效条目。
    """
    if not isinstance(recs, list):
        return []
    
    valid = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        
        movie_id = r.get('movie_id') or r.get('id')
        title = r.get('title', '')
        reason = r.get('reason', '')
        
        # movie_id 必须是整数
        if movie_id is not None:
            try:
                movie_id = int(movie_id)
            except (ValueError, TypeError):
                continue
        
        if movie_id and title:
            valid.append({
                'movie_id': movie_id,
                'title': str(title).strip(),
                'reason': str(reason).strip(),
            })
    
    return valid


def _regex_fallback(text: str) -> List[Dict]:
    """
    正则兜底提取（兼容原有格式）。
    匹配格式：
      《电影名》(ID:123)：理由
      《电影名》(ID:123)——理由
      1. 《电影名》(ID:123)：理由
    """
    results = []
    
    # 模式 1：带 ID 的格式
    pattern_with_id = r'《([^》]+?)》\s*\(ID:\s*(\d+)\)\s*[:：—–-]\s*(.+?)(?=\n\d+\.|\n《|$)'
    matches = re.findall(pattern_with_id, text, re.DOTALL)
    
    for title, movie_id, reason in matches:
        try:
            mid = int(movie_id.strip())
            results.append({
                'movie_id': mid,
                'title': title.strip(),
                'reason': reason.strip()[:100],  # 截断过长理由
            })
        except ValueError:
            continue
    
    if results:
        return results
    
    # 模式 2：只有 ID 的格式（更宽松）
    pattern_id_only = r'\(ID:\s*(\d+)\)'
    id_matches = re.findall(pattern_id_only, text)
    
    for mid_str in id_matches:
        try:
            mid = int(mid_str)
            # 尝试找到对应的电影名
            title_match = re.search(rf'《([^》]+?)》\s*\(ID:\s*{mid}\)', text)
            title = title_match.group(1) if title_match else f"Movie#{mid}"
            
            # 尝试提取该条推荐的理由
            reason_match = re.search(
                rf'《[^》]+?》\s*\(ID:\s*{mid}\)\s*[:：—–-]\s*(.+?)(?=\n\d+\.|\n《|$)',
                text, re.DOTALL
            )
            reason = reason_match.group(1).strip()[:100] if reason_match else ""
            
            results.append({
                'movie_id': mid,
                'title': title,
                'reason': reason,
            })
        except ValueError:
            continue
    
    return results


def extract_movie_ids_from_text(text: str) -> List[int]:
    """
    从 LLM 输出中提取所有电影 ID。
    兼容：(ID:123)、ID:123、电影ID 123 等格式。
    """
    if not text:
        return []
    
    ids = set()
    
    # 模式 1：(ID:123)
    for m in re.findall(r'\(ID:\s*(\d+)\)', text):
        ids.add(int(m))
    
    # 模式 2：movie_id: 123 (JSON 残留)
    for m in re.findall(r'"movie_id"\s*:\s*(\d+)', text):
        ids.add(int(m))
    
    # 模式 3：ID:123 (无括号)
    for m in re.findall(r'(?<!\()ID:\s*(\d+)', text):
        ids.add(int(m))
    
    return list(ids)


def inject_movie_links_html(text: str) -> str:
    """
    将文本中的 《电影名》(ID:123) 替换为 HTML 可点击链接。
    比原 views.py 中的 inject_movie_links 更健壮。
    """
    from django.urls import reverse
    
    def replace_link(match):
        title = match.group(1).strip()
        movie_id = match.group(2)
        
        try:
            from myapp.models import Movie
            movie = None
            if movie_id:
                movie = Movie.objects.filter(pk=movie_id).first()
            if not movie:
                movie = Movie.objects.filter(title__iexact=title).first()
            
            if movie:
                url = reverse('movie_detail', args=[movie.pk])
                return f'<a href="{url}" target="_blank" class="chat-movie-link">《{title}》</a>'
        except Exception:
            pass
        
        return f'《{title}》'
    
    pattern = re.compile(r'《([^《》]+?)》(?:\(ID:(\d+)\))?')
    return pattern.sub(replace_link, text)