# 文件: myapp/utils/graph_rag.py (V_Keyword_Fix - 修复关键词残留)

import re
import os
import django
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# --- 1. 初始化 Django 环境 ---
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoProject3.settings')
from django.conf import settings

if not settings.configured: django.setup()
from myapp.models import Movie, Genre

# --- 2. 连接 Neo4j ---
try:
    from django.conf import settings as _s
    graph = Neo4jGraph(
        url=getattr(_s, 'NEO4J_URI', 'bolt://localhost:7687'),
        username=getattr(_s, 'NEO4J_USER', 'neo4j'),
        password=getattr(_s, 'NEO4J_PASSWORD', ''),
        sanitize=True
    )
except Exception as e:
    print(f"❌ Neo4j 连接失败: {e}")
    graph = None


# --- 3. 动态获取类型 ---
def get_dynamic_genres():
    try:
        genres = list(Genre.objects.values_list('name', flat=True).distinct())
        if not genres: return "Action, Comedy, Drama, Sci-Fi"
        return ", ".join(genres)
    except:
        return "Action, Comedy, Drama, Sci-Fi"


# --- 4. Prompt ---
CYPHER_GENERATION_TEMPLATE = """
You are a Neo4j Cypher expert.
Task: Generate a Cypher query to answer the question.

Schema:
{schema}

Current Database Genres: [{valid_genres}]

Rules:
1. **Match Logic**: Use `CONTAINS` for fuzzy search on properties (e.g., m.summary, m.title).
2. **Return**: Always return `m.mid` (and `m.score` if available).
3. **Limit**: Limit 5.
4. **Neo4j 5.x Compatibility (CRITICAL)**: 
   - **NEVER use the `ID()` function.** It is deprecated.
   - To check if nodes are different, compare their 'mid' property: `WHERE m.mid <> n.mid`.
   - Or compare the nodes directly: `WHERE m <> n`.
5. **Output**: RAW CYPHER ONLY. No markdown, no explanations.

Question: {question}
Cypher Query:
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question", "valid_genres"],
    template=CYPHER_GENERATION_TEMPLATE
)

if graph:
    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0)
else:
    llm = None


def clean_cypher_query(query_text):
    # 移除 markdown 代码块
    cleaned = re.sub(r"```.*?```", "", query_text, flags=re.DOTALL)
    if not cleaned:  # 如果正则把内容都删空了，说明内容就在代码块里
        match = re.search(r"```(?:cypher)?(.*?)```", query_text, re.DOTALL | re.IGNORECASE)
        if match: cleaned = match.group(1)
    else:
        cleaned = query_text

    # 移除常见前缀
    cleaned = re.sub(r"^(Here is|Cypher|Query|Answer).*?:", "", cleaned, flags=re.IGNORECASE)

    # 强制截取 MATCH 开始
    match_idx = cleaned.upper().find("MATCH")
    if match_idx != -1:
        cleaned = cleaned[match_idx:]

    return cleaned.strip()


# --- 5. 核心修复：智能关键词提取 ---
def extract_keywords_and_map(text):
    # 1. 扩展停用词表 (增加 '片', '影片' 等)
    stop_words = [
        # 基础词
        '推荐', '电影', '几部', '一部', '两部', '三部', '关于', '的', '有没有',
        '想看', '什么', '哪些', '相关', '类型', '片', '影片', '大片', '作品',
        '佳作', '好看', '一下', '帮我', '分析', '我', '喜欢',
        # 上下文与数量词
        '再来', '再', '还有', '没有', '同', '类似', '换', '一批', '个', '点',
        '相似', '像', '风格', '一样', '些', '更多',
        # ✅ 新增口语废话
        '要', '不', '不同', '噢', '哦', '哈', '的', '了', '吧', '吗'
    ]

    clean_text = text

    # 2. 映射字典
    synonym_map = {
        '太空': 'Sci-Fi', '宇宙': 'Sci-Fi', '科幻': 'Sci-Fi',
        '搞笑': 'Comedy', '喜剧': 'Comedy',
        '吓人': 'Horror', '恐怖': 'Horror',
        '打架': 'Action', '动作': 'Action',
        '爱情': 'Romance', '恋爱': 'Romance',
        '卡通': 'Animation', '动画': 'Animation',
        '剧情': 'Drama'
    }

    # 动态注入数据库类型
    db_genres = list(Genre.objects.values_list('name', flat=True))
    for g in db_genres:
        synonym_map[g] = g
        synonym_map[g.lower()] = g

    mapped_genre = None

    # 3. 先提取类型，并从文本中移除类型词
    for k, v in synonym_map.items():
        if k in clean_text:
            if v in db_genres:
                mapped_genre = v
                clean_text = clean_text.replace(k, " ")  # 替换为空格防止粘连
                break

    # 4. 后置清洗：去除停用词 (放在最后，防止 '科幻片' 被切成 '科幻' 和 '片' 后，'片' 残留)
    for sw in stop_words:
        clean_text = clean_text.replace(sw, " ")

    # 5. 去除多余空格
    clean_text = clean_text.strip()

    return clean_text, mapped_genre


def fallback_graph_search(user_input):
    keyword, mapped_genre = extract_keywords_and_map(user_input)

    print(f"   🔄 兜底触发 | 关键词: '{keyword}' | 映射类型: {mapped_genre}")

    results = []

    # 策略 A: 类型搜索
    if mapped_genre:
        cypher = """
        MATCH (m:Movie)-[:BELONGS_TO]->(g:Genre)
        WHERE g.name = $genre
        RETURN m.mid, m.score
        ORDER BY m.score DESC LIMIT 5
        """
        try:
            res = graph.query(cypher, params={'genre': mapped_genre})
            results.extend(res)
        except:
            pass

    # 策略 B: 关键词搜索 (只有当关键词有效且长度>1时才搜)
    # 避免搜 "片" 这种单字
    if keyword and len(keyword) > 1:
        cypher = """
        MATCH (n)-[]-(m:Movie)
        WHERE (n:Person OR n:Movie) AND n.name CONTAINS $kw
        RETURN m.mid, m.score
        ORDER BY m.score DESC LIMIT 5
        """
        try:
            res = graph.query(cypher, params={'kw': keyword})
            existing = {r['m.mid'] for r in results}
            for r in res:
                if r['m.mid'] not in existing:
                    results.append(r)
        except:
            pass

    return results


def query_graph_rag(query_text):
    if not graph or not llm: return None

    try:
        current_valid_genres = get_dynamic_genres()
        schema_str = graph.get_schema

        formatted_prompt = cypher_prompt.format(
            schema=schema_str,
            question=query_text,
            valid_genres=current_valid_genres
        )

        response = llm.invoke(formatted_prompt)
        cleaned_cypher = clean_cypher_query(response.content)

        result_data = None
        if cleaned_cypher:
            try:
                result_data = graph.query(cleaned_cypher)
            except:
                pass

        if not result_data:
            result_data = fallback_graph_search(query_text)

        if not result_data: return None

        formatted_res = []
        seen_ids = set()

        for item in result_data:
            mid = item.get('m.mid') or item.get('mid')
            if not mid or mid in seen_ids: continue
            seen_ids.add(mid)

            try:
                movie_obj = Movie.objects.get(pk=mid)
                title = movie_obj.title
                score = movie_obj.score
                summary = movie_obj.summary[:80] + "..." if movie_obj.summary else "暂无简介"
                info_block = f"电影：《{title}》(ID:{mid}) | 评分：{score} | 简介：{summary}"
                formatted_res.append(info_block)
            except:
                pass

        return "\n".join(formatted_res)

    except Exception as e:
        print(f"GraphRAG Error: {e}")
        return None