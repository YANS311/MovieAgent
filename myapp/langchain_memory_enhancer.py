# ========================================================
# LangChain 高级机制集成模块 (LME v1.0)
# 功能：记忆管理、检索优化、逻辑分发
# ========================================================

import json
import time
from typing import List, Dict, Tuple, Optional, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio


# ========================================================
# 1. ConversationSummaryBufferMemory - 记忆管理
# ========================================================

class ConversationSummaryBufferMemory:
    """
    高级记忆管理器：维护短期记忆 + 长期摘要

    核心逻辑：
    - 最近的 N 轮对话保持原样（short_term）
    - 更早之前的对话压缩成摘要（moving_summary）
    - Token 预算受限时自动触发摘要生成

    参数：
    - max_token_limit: Token 预算上限（默认 800）
    - short_term_rounds: 保留的最近轮数（默认 3-5 轮）
    - llm: 用于生成摘要的 LLM 实例
    """

    def __init__(
        self,
        max_token_limit: int = 800,
        short_term_rounds: int = 5,
        llm: Optional[ChatOllama] = None
    ):
        self.max_token_limit = max_token_limit
        self.short_term_rounds = short_term_rounds
        self.chat_history: List[Dict[str, str]] = []  # 完整历史
        self.short_term_memory: List[Dict[str, str]] = []  # 最近 N 轮
        self.moving_summary: str = ""  # 历史摘要
        self.llm = llm or ChatOllama(model="qwen3:4b-instruct", temperature=0.3)
        self._token_count = 0

    def _estimate_tokens(self, text: str) -> int:
        """粗糙的 Token 估计（中文：1字≈1.3 Token，英文：1词≈1.5 Token）"""
        # 中文字数
        cn_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        # 非中文单词数
        en_words = len(text.encode('utf-8').decode('utf-8').split()) - cn_chars
        return int(cn_chars * 1.3 + en_words * 1.5)

    def add_message(self, role: str, content: str) -> None:
        """
        添加新消息到聊天历史

        参数：
        - role: 'user' 或 'ai'
        - content: 消息内容
        """
        message = {'role': role, 'content': content}
        self.chat_history.append(message)
        self._token_count += self._estimate_tokens(content)

        # 动态更新短期记忆（保留最近 N 轮）
        self.short_term_memory = self.chat_history[-self.short_term_rounds * 2:]

        # 触发摘要生成（Token 超过预算时）
        if self._token_count > self.max_token_limit:
            self._summarize_history()

    def _summarize_history(self) -> None:
        """调用 LLM 生成历史摘要"""
        if len(self.chat_history) <= self.short_term_rounds * 2:
            return  # 历史不足，无需摘要

        # 获取待摘要的对话（除去最近 N 轮）
        history_to_summarize = self.chat_history[:-self.short_term_rounds * 2]

        if not history_to_summarize:
            return

        # 构造摘要 Prompt
        dialogue_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history_to_summarize
        ])

        summary_prompt = f"""请用一段简洁的文本总结以下对话的核心要点（不超过 100 字）：

对话如下：
{dialogue_text}

摘要："""

        try:
            t_start = time.time()
            response = self.llm.invoke(summary_prompt)
            self.moving_summary = response.content.strip()
            t_elapsed = time.time() - t_start
            print(f"⏱️ [Memory 摘要] 耗时 {t_elapsed:.2f}s | 摘要长度: {len(self.moving_summary)} 字")
        except Exception as e:
            print(f"❌ [Memory 摘要失败] {e}")

    def get_context(self) -> Tuple[str, str]:
        """
        获取完整的记忆上下文（用于 Prompt 注入）

        返回：(moving_summary, short_term_text)
        """
        short_term_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.short_term_memory
        ])
        return self.moving_summary, short_term_text

    def clear(self) -> None:
        """清空所有记忆"""
        self.chat_history.clear()
        self.short_term_memory.clear()
        self.moving_summary = ""
        self._token_count = 0


# ========================================================
# 2. ContextualCompressionRetriever - 检索优化
# ========================================================

class ContextualCompressionRetriever:
    """
    语义压缩检索器：对 RAG 结果进行二次过滤

    核心逻辑：
    - 计算查询与召回文档的语义相似度
    - 仅保留相似度 >= threshold 的文档
    - 自动去除噪音信息

    参数：
    - similarity_threshold: 相似度阈值（默认 0.75）
    - llm: 用于生成检索证据的 LLM
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self._embedding_cache = {}

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    def _text_to_simple_embedding(self, text: str) -> List[float]:
        """
        简单的文本嵌入（基于字符统计，用于演示）
        在实际环境中应该使用真实的 embedding 模型（如 FAISS + 向量库）
        """
        # 使用字符频率作为简单 embedding
        text_lower = text.lower()
        embedding = [0] * 256

        for char in text_lower:
            if ord(char) < 256:
                embedding[ord(char)] += 1

        # 规范化
        total = sum(embedding)
        if total > 0:
            embedding = [x / total for x in embedding]

        return embedding

    def filter_documents(
        self,
        query: str,
        documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        过滤文档，仅保留相关度高的结果

        参数：
        - query: 用户查询
        - documents: 候选文档列表（每个文档是 {'content': str, 'metadata': dict} 的格式）

        返回：过滤后的文档列表
        """
        if not documents:
            return []

        query_embedding = self._text_to_simple_embedding(query)
        filtered = []

        for doc in documents:
            content = doc.get('content', '')
            doc_embedding = self._text_to_simple_embedding(content)

            # 计算相似度
            similarity = self._cosine_similarity(query_embedding, doc_embedding)

            # 保留高相关度的文档
            if similarity >= self.similarity_threshold:
                doc_with_score = doc.copy()
                doc_with_score['similarity_score'] = similarity
                filtered.append(doc_with_score)

        # 按相似度降序排列
        filtered.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        print(f"🔍 [Compression] 递交 {len(documents)} 文档 → 过滤至 {len(filtered)} "
              f"(threshold={self.similarity_threshold})")

        return filtered


# ========================================================
# 3. RunnableBranch (LCEL) - 逻辑分发
# ========================================================

class ChatBranchRouter:
    """
    基于 LangChain LCEL 的智能路由器

    核心逻辑：
    - 根据意图（intent）自动分配到不同的处理链路
    - 每个分支可以定制化处理逻辑
    - 支持动态扩展新分支

    三个核心分支：
    1. recommend_chain: 电影推荐（QUERY_MOVIE, QUERY_COMPARISON）
    2. fact_chain: 知识图谱查询（QUERY_KG）
    3. casual_chain: 日常闲聊（CHAT）
    """

    def __init__(self):
        self.branches: Dict[str, Any] = {}
        self._setup_branches()

    def _setup_branches(self) -> None:
        """初始化所有处理分支"""
        # 分支 1: 推荐链路（调用向量库 + RAG）
        self.branches['recommend_chain'] = self._recommend_chain_handler

        # 分支 2: 事实/知识链路（调用 Neo4j）
        self.branches['fact_chain'] = self._fact_chain_handler

        # 分支 3: 闲聊链路（直接 LLM）
        self.branches['casual_chain'] = self._casual_chain_handler

        # 分支 4: 视觉搜索链路
        self.branches['visual_chain'] = self._visual_chain_handler

    def _recommend_chain_handler(self, user_input: str, context: Dict) -> Dict[str, str]:
        """
        推荐链路：结合向量库和用户画像推荐电影
        """
        return {
            'route': 'recommend',
            'description': '🎬 电影推荐链路',
            'action': f'基于您的偏好和"{user_input}"进行个性化推荐',
            'use_rag': True,
            'use_graph': True,  # 可选：融合知识图谱
        }

    def _fact_chain_handler(self, user_input: str, context: Dict) -> Dict[str, str]:
        """
        事实链路：基于 Neo4j 图谱的结构化知识检索
        """
        return {
            'route': 'fact',
            'description': '📊 知识图谱查询',
            'action': f'从电影知识图谱中检索"{user_input}"的相关信息',
            'use_rag': False,
            'use_graph': True,
        }

    def _casual_chain_handler(self, user_input: str, context: Dict) -> Dict[str, str]:
        """
        闲聊链路：纯 LLM 对话，无外部知识源
        """
        return {
            'route': 'casual',
            'description': '💬 日常闲聊',
            'action': f'与您就"{user_input}"进行轻松对话',
            'use_rag': False,
            'use_graph': False,
        }

    def _visual_chain_handler(self, user_input: str, context: Dict) -> Dict[str, str]:
        """
        视觉搜索链路：基于海报和视觉特征的搜索
        """
        return {
            'route': 'visual',
            'description': '👁️ 视觉搜索',
            'action': f'基于"{user_input}"的视觉描述搜索电影',
            'use_rag': True,
            'use_graph': False,
        }

    def route(self, intent: str, user_input: str, context: Optional[Dict] = None) -> Dict[str, str]:
        """
        核心路由逻辑

        参数：
        - intent: 意图标签（QUERY_MOVIE, CHAT, QUERY_VISUAL 等）
        - user_input: 用户输入
        - context: 额外上下文（用户画像、历史等）

        返回：路由结果 dict
        """
        context = context or {}

        # 意图 → 分支的映射表
        intent_to_branch = {
            'QUERY_MOVIE': 'recommend_chain',
            'QUERY_COMPARISON': 'recommend_chain',
            'QUERY_PROFILE_REC': 'recommend_chain',
            'QUERY_RANK': 'recommend_chain',
            'QUERY_NEW': 'recommend_chain',
            'QUERY_VISUAL': 'visual_chain',
            'QUERY_VISUAL_RETRY': 'visual_chain',
            'QUERY_KG': 'fact_chain',
            'CHAT': 'casual_chain',
            'QUERY_SELF': 'casual_chain',
        }

        # 获取目标分支
        branch_name = intent_to_branch.get(intent, 'casual_chain')
        handler = self.branches.get(branch_name)

        if handler is None:
            print(f"⚠️ [Router] 未知分支: {branch_name}，降级到 casual_chain")
            handler = self.branches['casual_chain']

        result = handler(user_input, context)
        result['intent'] = intent
        result['branch'] = branch_name

        print(f"🚀 [Router] {result['description']} ← {intent}")

        return result

    def add_branch(
        self,
        branch_name: str,
        handler: callable,
        intent_aliases: List[str] = None
    ) -> None:
        """
        动态添加新分支（支持扩展）

        参数：
        - branch_name: 分支名称
        - handler: 处理函数 (user_input, context) -> Dict
        - intent_aliases: 关联的意图列表
        """
        self.branches[branch_name] = handler
        print(f"✨ [Router] 新增分支: {branch_name}")


# ========================================================
# 4. 集成函数（供 ajax_chat 调用）
# ========================================================

def initialize_chat_memory(
    chat_history: List[Dict[str, str]],
    max_token_limit: int = 800
) -> ConversationSummaryBufferMemory:
    """
    初始化聊天记忆管理器

    参数：
    - chat_history: 现有的聊天历史列表
    - max_token_limit: Token 预算上限

    返回：配置好的 ConversationSummaryBufferMemory 实例
    """
    memory = ConversationSummaryBufferMemory(max_token_limit=max_token_limit)

    # 加载现有历史
    for msg in chat_history:
        memory.add_message(msg['role'], msg['content'])

    return memory


def compress_retrieval_results(
    query: str,
    rag_documents: List[Dict[str, str]],
    similarity_threshold: float = 0.75
) -> List[Dict[str, str]]:
    """
    对 RAG 检索结果进行语义压缩

    参数：
    - query: 用户查询
    - rag_documents: 初始检索结果
    - similarity_threshold: 相似度阈值

    返回：过滤后的文档列表
    """
    compressor = ContextualCompressionRetriever(similarity_threshold=similarity_threshold)
    return compressor.filter_documents(query, rag_documents)


def route_user_intent(
    intent: str,
    user_input: str,
    user_context: Optional[Dict] = None
) -> Dict[str, str]:
    """
    根据用户意图进行智能路由

    参数：
    - intent: 分类出的意图（QUERY_MOVIE, CHAT 等）
    - user_input: 用户输入文本
    - user_context: 用户画像等上下文信息

    返回：路由决策和推荐链路
    """
    router = ChatBranchRouter()
    return router.route(intent, user_input, user_context)


# ========================================================
# 5. 主要 Prompt 构造器（LCEL 风格）
# ========================================================

def build_memory_enhanced_prompt(
    user_input: str,
    memory: ConversationSummaryBufferMemory,
    rag_context: str = "",
    system_role: str = "电影推荐助手"
) -> str:
    """
    构造增强的 Prompt：包含记忆摘要、短期历史和 RAG 知识

    Prompt 结构：
    ┌─ 系统提示
    ├─ 【历史摘要】（如果存在）
    ├─ 【最近对话】
    ├─ 【RAG 检索结果】（如果存在）
    └─ 【当前查询】
    """
    prompt_parts = [
        f"你是专业的{system_role}。请根据以下信息回答用户的问题。\n"
    ]

    # 添加历史摘要（如果存在）
    moving_summary, short_term = memory.get_context()

    if moving_summary:
        prompt_parts.append(f"\n【对话历史摘要】\n{moving_summary}\n")

    # 添加短期历史
    if short_term:
        prompt_parts.append(f"\n【最近对话】\n{short_term}\n")

    # 添加 RAG 检索结果
    if rag_context:
        prompt_parts.append(f"\n【RAG 检索结果】\n{rag_context}\n")

    # 添加当前查询
    prompt_parts.append(f"\n【当前查询】\n用户: {user_input}\n\n请提供专业、有见地的回答：")

    return "".join(prompt_parts)


# ========================================================
# 6. 工具函数
# ========================================================

def format_memory_for_display(memory: ConversationSummaryBufferMemory) -> Dict[str, str]:
    """
    格式化记忆信息用于日志输出
    """
    moving_summary, short_term = memory.get_context()

    return {
        'total_messages': len(memory.chat_history),
        'short_term_rounds': len(memory.short_term_memory),
        'summary_length': len(moving_summary),
        'has_summary': bool(moving_summary),
    }

