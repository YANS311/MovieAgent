"""
Agent Tool Registry - 统一工具注册与调度机制
─────────────────────────────────────────────────────────
【论文可写点】：Tool Calling 是 ReAct Agent 的核心机制。本模块实现了统一的
              工具注册表（Tool Registry），支持：
              1. 工具统一注册与发现
              2. 统一异常处理与降级
              3. 工具耗时统计
              4. 便于后续扩展新工具

【修改原因】：原 ajax_chat 中工具调用通过 prompt_builders 字典的 if/else 分支实现，
            结构分散，异常处理不统一。提取为 Registry 模式更规范。
【性能收益】：工具调用统一入口，便于添加缓存、重试、降级策略。
"""
import logging
import time
from typing import Callable, Dict, Optional, Any, Tuple

logger = logging.getLogger('tool_registry')

# ── 全局工具注册表 ──────────────────────────────────────────────
# 【论文可写点】：Tool Registry 模式，每个工具以 intent_key 注册，
#              统一入口 dispatch() 负责路由、异常捕获、耗时统计。
_TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(intent_key: str):
    """
    装饰器：注册一个 Agent 工具
    
    用法：
        @register_tool("QUERY_MOVIE")
        def build_movie_prompt(user, search_query, ...):
            ...
    """
    def decorator(func: Callable):
        _TOOL_REGISTRY[intent_key] = func
        logger.debug(f"[ToolRegistry] 注册工具: {intent_key} -> {func.__name__}")
        return func
    return decorator


def dispatch_tool(intent_key: str, fallback_key: str = "CHAT",
                  *args, **kwargs) -> Tuple[Any, Any, Any]:
    """
    统一工具调度入口
    
    【论文可写点】：dispatch_tool() 实现了工具的统一调度、异常隔离、
                  自动降级（fallback）机制，确保单个工具失败不会影响整个 Agent。
    
    Args:
        intent_key: 意图标识（如 QUERY_MOVIE, QUERY_VISUAL 等）
        fallback_key: 降级意图标识
        *args, **kwargs: 传递给工具函数的参数
    
    Returns:
        tuple: (visual_response, final_prompt, temperature)
    """
    t_start = time.time()
    
    # 1. 查找工具
    tool_func = _TOOL_REGISTRY.get(intent_key)
    if not tool_func:
        logger.warning(f"[ToolRegistry] 未找到工具: {intent_key}，降级到 {fallback_key}")
        tool_func = _TOOL_REGISTRY.get(fallback_key)
    
    if not tool_func:
        logger.error(f"[ToolRegistry] 降级工具也不存在: {fallback_key}")
        return None, None, 0.3
    
    # 2. 执行工具（带异常隔离）
    try:
        result = tool_func(*args, **kwargs)
        latency_ms = round((time.time() - t_start) * 1000, 1)
        logger.info(f"[ToolRegistry] 工具={intent_key} latency={latency_ms}ms status=OK")
        return result
    except Exception as e:
        latency_ms = round((time.time() - t_start) * 1000, 1)
        logger.error(
            f"[ToolRegistry] 工具={intent_key} latency={latency_ms}ms "
            f"ERROR={type(e).__name__}: {e}，降级到 {fallback_key}"
        )
        # 3. 异常时自动降级到 fallback 工具
        if fallback_key and fallback_key != intent_key:
            fb_func = _TOOL_REGISTRY.get(fallback_key)
            if fb_func:
                try:
                    return fb_func(*args, **kwargs)
                except Exception as fb_e:
                    logger.error(f"[ToolRegistry] 降级工具也失败: {fallback_key} ERROR={fb_e}")
        
        # 4. 全部失败，返回安全兜底
        return None, "抱歉，系统暂时无法处理您的请求，请稍后重试。", 0.3


def list_registered_tools() -> list:
    """列出所有已注册的工具（用于调试和论文展示）"""
    return list(_TOOL_REGISTRY.keys())


def get_tool_count() -> int:
    """返回已注册工具数量"""
    return len(_TOOL_REGISTRY)