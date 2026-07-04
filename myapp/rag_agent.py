# 文件: myapp/rag_agent.py (V16 - Model Adapted Edition)

import os
import django
from datetime import datetime

# --- 1. 环境初始化 ---
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoProject3.settings')
# django.setup()

# --- 2. 第三方库导入 ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    create_tool_calling_agent = None
    AgentExecutor = None
from langchain.tools import tool
from langchain_community.vectorstores import FAISS

# 本地 LLM 支持
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Django 模型
from myapp.models import Movie, UserRating, ChatHistory, UserInfo, Genre, Actor
from django.db.models import Q, Count

# --- 3. 全局配置 ---
FAISS_INDEX_PATH = "faiss_movie_index"  # 向量库路径
MODEL_NAME = "qwen3:4b-instruct"  # 本地 Ollama 模型名
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 向量模型

_AGENT_EXECUTOR_ = None


# --- 4. 核心工具定义 (Tools) ---

@tool
def search_movies_vector(query: str):
    """
    【向量检索】适用于模糊查询、剧情描述或寻找某种"感觉"的电影。
    例如："我想看这种绝望感的科幻片"、"关于时空穿越的故事"。
    它基于语义相似度进行搜索。
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if not os.path.exists(FAISS_INDEX_PATH):
            return "系统提示: 向量数据库尚未构建，无法进行模糊搜索。"

        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        # 检索 Top 4
        docs = vector_db.similarity_search(query, k=4)

        results = []
        for doc in docs:
            # 注意：构建向量库时 metadata 里的字段名可能需要确认，通常假设保持一致
            # 这里对应 Movie 模型
            title = doc.metadata.get('title', '未知标题')
            mid = doc.metadata.get('id', 0)
            score = doc.metadata.get('score', 0.0)

            info = f"电影ID: {mid} | 片名: 《{title}》 | 评分: {score}\n简介: {doc.page_content[:100]}..."
            results.append(info)

        return "\n---\n".join(results)
    except Exception as e:
        return f"向量检索出错: {str(e)}"


@tool
def explore_movie_graph(movie_name: str):
    """
    【全能详情与图谱检索】这是你最强大的工具。
    适用于：
    1. 查询某部电影的**详细信息**（包括**海报视觉风格**、剧情、导演、评分）。
    2. 基于"关系"寻找类似的电影（如"同导演"、"同主演"、"同类型"）。

    当用户问："黑客帝国海报长啥样？"、"这电影讲啥的？"、"推荐类似星际穿越的电影"时，**优先使用此工具**。
    输入：电影名称 (例如 "星际穿越")
    """
    # 1. 锚点定位 (Graph Entry) - 使用 title 字段
    target = Movie.objects.filter(title__icontains=movie_name).prefetch_related('directors', 'actors', 'genres').first()
    if not target:
        return f"数据库中未找到电影: {movie_name}，请尝试使用向量检索工具描述剧情。"

    # --- 🔥 多模态核心修改 Start ---
    # 获取 Qwen-VL 生成的视觉描述
    visual_info = getattr(target, 'poster_caption', "暂无描述")
    # 获取剧情简介
    movie_summary = target.summary if target.summary else "暂无简介"

    # 获取导演信息 (新增)
    directors = target.directors.all()
    director_names = [d.name for d in directors]
    director_str = ", ".join(director_names) if director_names else "未知导演"

    results_text = [
        f"🎬 【电影档案】 《{target.title}》 评分: {target.score}",
        f"🎬 [导演]: {director_str}",  # <-- 🔥 新增导演信息
        f"📖 [剧情简介]: {movie_summary[:300]}...",
        f"🎨 [AI视觉分析]: {visual_info}",
        f"-----------------------------------"
    ]
    # --- 🔥 多模态核心修改 End ---

    # 2. 提取节点属性 (处理 ManyToMany 关系)
    actors = target.actors.all()[:3]  # 取前3位主演
    genres = target.genres.all()  # 取所有类型

    # 3. 路径 A: 导演关联 (Director Relation - 强关联) 🔥 新增
    if directors:
        related_by_director = Movie.objects.filter(directors__in=directors) \
            .exclude(id=target.id) \
            .filter(score__gte=7.5) \
            .order_by('-score')[:4]

        if related_by_director:
            results_text.append(f"🎥 基于同导演 ({director_str}) 的高分作品:")
            for m in related_by_director:
                results_text.append(f"   - [ID:{m.id}] 《{m.title}》 ({m.score}分)")

    # 4. 路径 B: 演员共现 (Co-starring Relation)
    if actors:
        actor_names = [a.name for a in actors]
        related_movies = Movie.objects.filter(actors__in=actors) \
            .exclude(id=target.id) \
            .filter(score__gte=7.5) \
            .annotate(common_actor_count=Count('actors')) \
            .order_by('-score')[:3]  # 稍微减少数量，避免上下文过长

        if related_movies:
            results_text.append(f"🌟 基于共同主演 ({', '.join(actor_names)}):")
            for m in related_movies:
                m_actors = m.actors.all()
                common = [a.name for a in m_actors if a in actors]
                results_text.append(f"   - [ID:{m.id}] 《{m.title}》 ({m.score}分) (同演: {', '.join(common)})")

    # 5. 路径 C: 同类型高分 (Genre Relation)
    if genres:
        main_genre = genres[0]
        genre_movies = Movie.objects.filter(genres__name=main_genre.name) \
            .exclude(id=target.id) \
            .filter(score__gte=8.0) \
            .order_by('-vote_count')[:3]

        if genre_movies:
            results_text.append(f"🏷️ 基于同类型 ({main_genre.name}) 的高分经典:")
            for m in genre_movies:
                results_text.append(f"   - [ID:{m.id}] 《{m.title}》 ({m.score}分)")

    return "\n".join(results_text)


@tool
def recommend_by_history(user_id: str):
    """
    【个性化推荐】当用户说"推荐几部我喜欢的"、"猜我喜欢"时使用。
    基于用户的历史高分记录进行推荐。
    输入: user_id (字符串形式的数字)
    """
    try:
        uid = int(user_id)
        # 获取用户最近好评的电影
        liked_movies = UserRating.objects.filter(user_id=uid, score__gte=4.0).order_by('-comment_time')[:3]

        if not liked_movies.exists():
            return "用户暂无足够的高分观影记录，建议先进行热门推荐。"

        # 获取最近一部喜欢的电影对象
        last_movie = liked_movies[0].movie
        history_names = [r.movie.title for r in liked_movies]

        # 获取该电影的第一个类型
        last_genres = last_movie.genres.all()
        if not last_genres:
            return f"您最近喜欢《{last_movie.title}》，但该电影暂无类型标签。"

        main_genre_name = last_genres[0].name

        # 简单的 ItemCF 替代逻辑：找同类型且没看过的
        watched_ids = UserRating.objects.filter(user_id=uid).values_list('movie_id', flat=True)

        recommendations = Movie.objects.filter(genres__name=main_genre_name) \
            .exclude(id__in=watched_ids) \
            .order_by('-score')[:5]

        res_str = f"基于您最近喜欢的《{last_movie.title}》({main_genre_name})，为您推荐：\n"
        for m in recommendations:
            res_str += f"- [ID:{m.id}] 《{m.title}》 ({m.score}分)\n"

        return res_str
    except Exception as e:
        return f"获取推荐失败: {e}"


# --- 5. 聊天历史管理 (适配新 Model) ---

class DjangoDBChatHistory(BaseChatMessageHistory):
    def __init__(self, user_id: str):
        # 注意：这里的 user_id 对应 UserInfo 表的 id (即 session_id 传进来的值)
        self.user_id = user_id

    @property
    def messages(self):
        # 1. 检查 user_id 是否有效
        if not self.user_id or self.user_id == 'None':
            return []

        # 2. 从 ChatHistory 表加载
        # ChatHistory 表结构: user(FK), role, message, timestamp
        try:
            history = ChatHistory.objects.filter(user_id=self.user_id).order_by('timestamp')
            msgs = []
            for h in history:
                if h.role == 'user':
                    msgs.append(HumanMessage(content=h.message))  # 注意字段是 message
                else:
                    msgs.append(AIMessage(content=h.message))
            return msgs
        except Exception:
            return []

    def add_message(self, message: BaseMessage):
        # 1. 检查 user_id
        if not self.user_id or self.user_id == 'None':
            return

        # 2. 获取 User 实例 (因为是 ForeignKey)
        try:
            user_instance = UserInfo.objects.get(id=self.user_id)
        except UserInfo.DoesNotExist:
            print(f"User ID {self.user_id} not found, skip saving history.")
            return

        # 3. 保存到数据库
        role = 'user' if isinstance(message, HumanMessage) else 'ai'
        ChatHistory.objects.create(
            user=user_instance,
            role=role,
            message=message.content  # 注意字段是 message
        )

    def clear(self):
        if self.user_id and self.user_id != 'None':
            ChatHistory.objects.filter(user_id=self.user_id).delete()


# --- 6. Agent 初始化工厂 ---

def initialize_agent():
    """初始化并返回全局 Agent 执行器"""
    global _AGENT_EXECUTOR_

    if _AGENT_EXECUTOR_ is not None:
        return _AGENT_EXECUTOR_

    print(f"🔄 正在初始化 GraphRAG Agent (Model: {MODEL_NAME})...")

    # 1. 定义 LLM
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.3,
        keep_alive="1h"
    )

    # 2. 工具集
    tools = [
        search_movies_vector,  # Vector RAG
        explore_movie_graph,  # Graph RAG (Adapted)
        recommend_by_history  # Personalization (Adapted)
    ]

    # 3. System Prompt
    SYSTEM_PROMPT = """
    你是一个基于 GraphRAG 的专业电影助手。你不仅懂剧情，还能通过“眼睛”看懂海报。

    【核心决策逻辑】
    1. **模糊感知类** (如"我想看感人的") -> 优先调用 `search_movies_vector`。
    2. **关联/精准类** (如"类似星际穿越的"、"谁演的") -> **必须**优先调用 `explore_movie_graph`。
    3. **个性化推荐** (如"猜我喜欢") -> 调用 `recommend_by_history`。

    【视觉分析与推理要求】
    1. 当用户询问关于海报、画面、颜色或“为什么海报这样设计”时，请结合 [AI视觉分析] 和 [剧情简介] 进行深度回答。
    2. 不要仅仅复述描述。你要尝试解释视觉元素背后的象征意义。
       - 例如：若看到“绿色代码”，结合剧情解释这代表“Matrix虚拟世界的数字本质”。
       - 例如：若看到“冷色调”，结合剧情解释这暗示了“角色的孤独感”或“硬核科幻基调”。
    3. 如果视觉描述与剧情高度吻合，请赞赏该海报的设计巧妙地传达了电影核心。

    【回复规范】
    - **必须**为提到的每部电影生成链接，格式为：`[电影名](/movie/MOVIE_ID/)`。
    - 既然你有图谱工具，当用户问某部电影时，请多提供一点它的关联信息（如"这也同样是诺兰导演的作品"）。
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 6. 绑定历史记录 (关键：映射 session_id 到 user_id)
    _AGENT_EXECUTOR_ = RunnableWithMessageHistory(
        runnable=agent_executor,
        # 这里假设传入的 session_id 就是用户的 user_id (字符串格式)
        get_session_history=lambda session_id: DjangoDBChatHistory(user_id=session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("✅ GraphRAG Agent (Adapted) 初始化完成!")
    return _AGENT_EXECUTOR_


# --- 7. 对外接口 ---

def chat_with_agent(user_input, session_id, user_id=None):
    """
    View 调用的主入口
    """
    # 这里的 session_id 对于 RunnableWithMessageHistory 来说就是 lookup key
    # 在这个实现中，我们需要它等于 user_id，以便查表
    # 如果 user_id 存在，我们优先使用 user_id 作为 history key

    history_key = str(user_id) if user_id else str(session_id)

    agent = initialize_agent()

    context_input = user_input
    if user_id:
        context_input = f"[User ID: {user_id}] {user_input}"

    try:
        response = agent.invoke(
            {"input": context_input},
            config={"configurable": {"session_id": history_key}}
        )
        return response['output']
    except Exception as e:
        print(f"Agent Error: {e}")
        return "抱歉，我的大脑暂时短路了，请稍后再试。"