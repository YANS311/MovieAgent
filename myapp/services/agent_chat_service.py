"""
Agent 聊天服务层 (AgentChatService)
================================================
从 agent_views.py 的 God Function 中抽离业务逻辑，
遵循单一职责原则 (SRP) 和领域驱动设计 (DDD)。

架构定位：
  View 层 (agent_views.py) → 只负责 HTTP 参数解析和响应
  Service 层 (本文件)      → 负责 Agent 初始化、推理、数据持久化
  Model 层 (models.py)     → 负责数据结构定义

使用方式：
    service = AgentChatService(user, session_key)
    result = service.process_chat(user_input, is_thinking=False)
================================================
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Generator

from django.db import transaction

logger = logging.getLogger('movie_agent')


class AgentChatService:
    """
    Agent 聊天服务（核心业务逻辑封装）
    
    职责：
    1. 初始化 MovieAgent 实例（含外部资源注入）
    2. 执行 ReAct 推理
    3. 持久化 ChatHistory + AgentTrace
    4. 格式化 movie_details（海报 URL 处理）
    5. 排除"不喜欢"电影
    """
    
    def __init__(self, user, session_key: str = 'default'):
        self.user = user
        self.session_id = f"user_{user.id}_{session_key}"
        self._agent = None
    
    def _get_agent(self):
        """懒加载 Agent 实例，自动检测 Ollama 并注入 LLM 配置"""
        if self._agent is None:
            from myapp.agent.movie_agent import MovieAgent
            from myapp import views

            # 检测 Ollama 可用性
            llm_config = self._detect_ollama()

            self._agent = MovieAgent(
                user=self.user,
                neo_graph=getattr(views, 'neo_graph', None),
                rag_resources=getattr(views, 'RAG_RESOURCES', {}),
                session_id=self.session_id,
                llm_config=llm_config,
            )
        return self._agent

    @staticmethod
    def _detect_ollama():
        """检测 Ollama 服务是否可用，返回 llm_config 或 None"""
        import logging
        logger = logging.getLogger(__name__)
        try:
            import requests as req
            from django.conf import settings
            model = getattr(settings, 'AGENT_LLM_MODEL', 'qwen3:4b-instruct')
            enabled = getattr(settings, 'AGENT_LLM_ENABLED', True)
            if not enabled:
                return None
            olla_base = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            resp = req.get(f"{olla_base}/api/tags", timeout=2)
            resp.raise_for_status()
            available = [m.get('name', '') for m in resp.json().get('models', [])]
            # 检查模型是否已拉取（支持 qwen3:4b-instruct 和 qwen3:4b-instruct:latest）
            if any(model in m for m in available):
                logger.info(f"[AgentChatService] Ollama 可用，模型: {model}")
                return {'model_name': model, 'timeout': 30}
            else:
                logger.warning(f"[AgentChatService] Ollama 可用但模型 {model} 未拉取，可用: {available}")
                return None
        except Exception as e:
            logger.info(f"[AgentChatService] Ollama 不可用: {e}")
            return None
    
    def process_chat(self, user_input: str, is_thinking: bool = False) -> Dict[str, Any]:
        """
        处理一次完整的聊天请求。
        
        Args:
            user_input: 安全清洗后的用户输入
            is_thinking: 是否深度思考模式
        
        Returns:
            dict: 完整的响应数据（可直接序列化为 JSON）
        """
        t_start = time.time()
        
        # 1. 执行 Agent ReAct 推理
        agent = self._get_agent()
        result = agent.run(user_input)
        
        # 2. 排除"不喜欢"的电影
        excluded_ids = self._get_excluded_ids()
        if excluded_ids:
            result['recommended_ids'] = [
                mid for mid in result['recommended_ids']
                if mid not in excluded_ids
            ]
        
        # 3. 获取电影详情（标题+海报）
        movie_details = self._format_movie_details(result['recommended_ids'])
        
        # 4. 构建 ReAct Trace 展示
        react_display = agent.get_react_trace(result)
        
        # 5. 持久化（异步安全：在线程池中执行）
        self._save_history(user_input, result['final_answer'])
        self._save_trace(user_input, result)
        
        # 6. 构建响应
        t_total = time.time() - t_start
        
        return {
            'response': result['final_answer'],
            'react_trace': {
                'thought': result['thought'],
                'actions': result['actions'],
                'observations': [
                    {'tool': o.get('tool', ''), 'count': o.get('count', 0)}
                    for o in result['observations']
                ],
                'display_text': react_display,
            },
            'recommended_ids': result['recommended_ids'],
            'movie_details': movie_details,
            'explanations': {str(k): v for k, v in result['explanations'].items()},
            'intent': result['intent'],
            'latency_ms': result['latency_ms'],
            'need_clarification': result.get('need_clarification', False),
            'clarification_options': result.get('clarification_options', []),
            'trace_steps': result.get('trace_steps', []),
        }
    
    def stream_chat(self, user_input: str, is_thinking: bool = False) -> Generator[Dict, None, None]:
        """
        流式处理聊天请求（SSE 模式）— 增强版：逐步骤推送 ReAct 推理链。
        
        Yields:
            dict: SSE 数据块，格式 {"type": str, "content": str, ...}
        
        SSE 事件类型：
        1. intent      → 意图分类结果
        2. thought     → Agent 思考过程（每个 Thought 独立推送）
        3. action      → 工具调用信息（每个 Action 独立推送）
        4. observation  → 工具执行结果（每个 Observation 独立推送）
        5. reflection   → Agent 反思决策
        6. chunk       → 最终答案文本（逐块输出）
        7. done        → 完成信号 + 元数据
        """
        t_start = time.time()
        
        # Phase 0: 意图分类
        yield {
            'type': 'intent',
            'content': '正在分析您的需求...',
        }
        
        # Phase 1: 执行推理（获取完整 trace）
        agent = self._get_agent()
        
        # 使用 MovieAgent 执行（它已经记录了完整的 trace_steps）
        result = agent.run(user_input)
        
        # Phase 2: 逐步推送 ReAct 推理链
        trace_steps = result.get('trace_steps', [])
        
        for step in trace_steps:
            step_type = step.get('type', 'thought')
            content = step.get('content', '')
            tool_name = step.get('tool', '')
            is_retry = step.get('is_retry', False)
            
            if step_type == 'thought':
                yield {
                    'type': 'thought',
                    'content': content,
                    'is_retry': is_retry,
                    'step': step.get('step', 0),
                }
            elif step_type == 'action':
                yield {
                    'type': 'action',
                    'content': content,
                    'tool': tool_name,
                    'input': step.get('input', ''),
                    'is_retry': is_retry,
                    'step': step.get('step', 0),
                }
            elif step_type == 'observation':
                yield {
                    'type': 'observation',
                    'content': content,
                    'tool': tool_name,
                    'count': step.get('count', 0),
                    'is_retry': is_retry,
                    'step': step.get('step', 0),
                }
            elif step_type == 'clarification':
                yield {
                    'type': 'clarification',
                    'content': content,
                    'step': step.get('step', 0),
                }
            elif step_type == 'final_answer':
                # 不在这里推送，后面单独流式输出
                pass
            
            time.sleep(0.05)  # 短延迟，让前端能逐步渲染
        
        # Phase 3: 流式输出最终答案（逐块效果）
        final_text = result['final_answer']
        chunk_size = 5
        
        for i in range(0, len(final_text), chunk_size):
            chunk = final_text[i:i + chunk_size]
            yield {
                'type': 'chunk',
                'content': chunk,
            }
        
        # Phase 4: 完成 + 元数据
        excluded_ids = self._get_excluded_ids()
        if excluded_ids:
            result['recommended_ids'] = [
                mid for mid in result['recommended_ids']
                if mid not in excluded_ids
            ]
        
        movie_details = self._format_movie_details(result['recommended_ids'])
        react_display = agent.get_react_trace(result)
        
        # 持久化
        self._save_history(user_input, result['final_answer'])
        self._save_trace(user_input, result)
        
        t_total = int((time.time() - t_start) * 1000)
        
        yield {
            'type': 'done',
            'content': '',
            'meta': {
                'react_trace': {
                    'thought': result['thought'],
                    'actions': result['actions'],
                    'observations': [
                        {'tool': o.get('tool', ''), 'count': o.get('count', 0)}
                        for o in result['observations']
                    ],
                    'display_text': react_display,
                },
                'trace_steps': trace_steps,
                'recommended_ids': result['recommended_ids'],
                'movie_details': movie_details,
                'explanations': {str(k): v for k, v in result['explanations'].items()},
                'intent': result['intent'],
                'latency_ms': result['latency_ms'],
                'need_clarification': result.get('need_clarification', False),
                'clarification_options': result.get('clarification_options', []),
                'total_latency_ms': t_total,
            }
        }
    
    # ── 内部辅助方法 ──────────────────────────────────────
    
    def _get_excluded_ids(self) -> set:
        """获取用户"不喜欢"的电影ID"""
        try:
            from myapp.views import get_excluded_movie_ids
            return set(get_excluded_movie_ids(self.user))
        except Exception:
            return set()
    
    def _format_movie_details(self, movie_ids: list) -> dict:
        """格式化电影详情（含海报URL处理）"""
        from myapp.models import Movie
        
        details = {}
        if not movie_ids:
            return details
        
        movies = Movie.objects.filter(id__in=movie_ids[:5]).values('id', 'title', 'poster')
        for m in movies:
            poster_url = m.get('poster', '') or ''
            
            # 海报 URL 标准化
            if poster_url.startswith('http'):
                filename = poster_url.rsplit('/', 1)[-1]
                poster_url = f'/media/posters/{filename}'
            elif poster_url and not poster_url.startswith('/'):
                poster_url = f'/media/{poster_url}'
            
            details[str(m['id'])] = {
                'title': m['title'],
                'poster': poster_url,
            }
        
        return details
    
    def _save_history(self, user_input: str, ai_response: str):
        """保存对话历史"""
        try:
            from myapp.models import ChatHistory
            ChatHistory.objects.create(user=self.user, role='user', message=user_input)
            ChatHistory.objects.create(user=self.user, role='ai', message=ai_response)
        except Exception as e:
            logger.error(f"[Service] 保存对话历史失败: {e}")
    
    def _save_trace(self, user_input: str, result: dict):
        """保存 Agent 推理链"""
        try:
            from myapp.models_upgrade import AgentTrace
            AgentTrace.objects.create(
                user=self.user,
                user_input=user_input,
                intent=result['intent'],
                thought=result['thought'],
                actions=result['actions'],
                observations=[
                    {'tool': o.get('tool', ''), 'count': o.get('count', 0)}
                    for o in result['observations']
                ],
                final_answer=result['final_answer'],
                recommended_movies=result['recommended_ids'],
                explanations=result['explanations'],
                total_latency_ms=result['latency_ms'],
            )
        except Exception as e:
            logger.error(f"[Service] 保存 AgentTrace 失败: {e}")


def process_agent_request(user, user_input: str, session_key: str = 'default', is_thinking: bool = False) -> Dict:
    """
    便捷函数：一步完成 Agent 聊天处理。
    适用于不想实例化 Service 的场景。
    
    Returns:
        dict: 完整的 JSON 响应数据
    """
    service = AgentChatService(user, session_key)
    return service.process_chat(user_input, is_thinking)