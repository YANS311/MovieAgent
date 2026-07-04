"""
Agent 智能推荐视图（重构版）
================================================
架构升级：
  - 核心业务逻辑已抽离到 myapp/services/agent_chat_service.py
  - View 层仅负责 HTTP 参数解析和响应（薄 Controller）
  - 支持 SSE 流式输出（打字机效果）
  - 使用 logging 替代 print

视图清单：
  1.  chat_recommend_view     - 智能推荐聊天页
  2.  agent_api_view          - Agent API 接口 (JSON POST)
  2a. agent_stream_view       - Agent SSE 流式接口 (Streaming)
  3.  movie_explain_view      - 电影推荐解释页
  4.  recommend_feedback_view - 推荐反馈收集
  5.  agent_trace_view        - Agent推理链展示
  6.  ajax_agent_kg_query     - 知识图谱查询 API
================================================
"""

import json
import time
import logging
import traceback
from functools import wraps

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.db.models import Q

from myapp.models import Movie, UserRating, ChatHistory, Genre, Region

logger = logging.getLogger('movie_agent')


# ── 辅助函数 ──────────────────────────────────────────────

def _sanitize_input(raw_input):
    """安全清洗（延迟导入避免循环引用）"""
    from myapp.views import sanitize_user_input
    return sanitize_user_input(raw_input)


def _get_session_key(request):
    """获取会话标识"""
    return request.session.session_key or 'default'


def _admin_required_api(view_func):
    """API 接口的管理员权限装饰器，返回 JSON 而非重定向"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_staff:
            return JsonResponse({'status': 'error', 'msg': '仅管理员可访问'}, status=403)
        return view_func(request, *args, **kwargs)
    return wrapper


def _admin_required_view(view_func):
    """页面视图的管理员权限装饰器，无权限时重定向"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_staff:
            messages.error(request, "无权限访问")
            return redirect('front_index')
        return view_func(request, *args, **kwargs)
    return wrapper


# =============================================================
# 视图 1: 智能推荐聊天页
# =============================================================
@login_required
def chat_recommend_view(request):
    """智能推荐聊天页 - 展示 ReAct 推理链 + 推荐结果卡片"""
    from django.utils import timezone
    from datetime import timedelta
    
    time_threshold = timezone.now() - timedelta(hours=24)
    history_qs = ChatHistory.objects.filter(
        user=request.user,
        timestamp__gte=time_threshold
    ).order_by('timestamp')
    
    history_list = [
        {'role': msg.role, 'message': msg.message}
        for msg in history_qs
    ]
    
    context = {
        'chat_history_json': json.dumps(history_list),
        'page_title': '智能推荐助手',
    }
    return render(request, 'agent_chat.html', context)


# =============================================================
# 视图 2: Agent API 接口 (JSON POST)
# ★ 已重构：使用 AgentChatService 服务层
# =============================================================
@csrf_exempt
@login_required
def agent_api_view(request):
    """
    Agent 推荐 API 接口（JSON 响应）
    
    POST 参数: msg, is_thinking (可选)
    返回: {response, react_trace, recommended_ids, movie_details, ...}
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    raw_input = request.POST.get('msg', '').strip()
    if not raw_input:
        return JsonResponse({
            'response': '您好！我是智能推荐助手，请问今天想看什么类型的电影？',
            'react_trace': None,
        })
    
    # 安全清洗
    user_input = _sanitize_input(raw_input)
    if user_input == "MALICIOUS_INJECTION_DETECTED":
        return JsonResponse({
            'response': '🛡️ 系统拦截：检测到不安全的输入内容。',
            'react_trace': None,
        })
    
    try:
        from myapp.services.agent_chat_service import AgentChatService
        
        service = AgentChatService(request.user, _get_session_key(request))
        result = service.process_chat(
            user_input,
            is_thinking=request.POST.get('is_thinking') == 'true'
        )
        
        # ★ XAI 方案一：拟人化思考流
        from myapp.utils.xai_explainer import translate_react_to_human
        result['user_friendly_trace'] = translate_react_to_human(result.get('react_trace', {}))
        
        logger.info(
            f"[Agent] 用户={request.user.username} | "
            f"意图={result.get('intent', '')} | "
            f"耗时={result.get('latency_ms', 0)}ms"
        )
        
        return JsonResponse(result)
    
    except Exception as e:
        logger.exception(f"[Agent] 推理异常: {e}")
        return JsonResponse({
            'response': '智能助手暂时遇到了问题，请稍后重试。',
            'react_trace': None,
            'error': str(e),
        })


# =============================================================
# 视图 2a: Agent SSE 流式接口（打字机效果）
# ★ 新增：Server-Sent Events 实时推送
# =============================================================
@csrf_exempt
@login_required
def agent_stream_view(request):
    """
    Agent SSE 流式接口（打字机效果）
    
    POST 参数: msg, is_thinking (可选)
    响应: SSE 流，每行格式 data: {"type": "...", "content": "..."}\n\n
    
    SSE 事件类型：
    - thought:  Agent 思考过程
    - action:   工具调用信息
    - chunk:    最终答案文本（逐块输出）
    - done:     完成信号 + 元数据（react_trace, recommended_ids 等）
    - error:    错误信息
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    raw_input = request.POST.get('msg', '').strip()
    if not raw_input:
        def empty_stream():
            yield _sse_data({
                'type': 'done',
                'content': '您好！我是智能推荐助手，请问今天想看什么类型的电影？',
            })
        return StreamingHttpResponse(
            empty_stream(),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Nginx 禁用缓冲
            }
        )
    
    # 安全清洗
    user_input = _sanitize_input(raw_input)
    if user_input == "MALICIOUS_INJECTION_DETECTED":
        def blocked_stream():
            yield _sse_data({
                'type': 'error',
                'content': '🛡️ 系统拦截：检测到不安全的输入内容。',
            })
        return StreamingHttpResponse(
            blocked_stream(),
            content_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )
    
    def event_stream():
        try:
            from myapp.services.agent_chat_service import AgentChatService
            
            service = AgentChatService(request.user, _get_session_key(request))
            is_thinking = request.POST.get('is_thinking') == 'true'
            
            for chunk in service.stream_chat(user_input, is_thinking):
                yield _sse_data(chunk)
                time.sleep(0.02)  # 微延迟，避免客户端缓冲区溢出
            
        except Exception as e:
            logger.exception(f"[SSE] 流式推理异常: {e}")
            yield _sse_data({
                'type': 'error',
                'content': '智能助手暂时遇到了问题，请稍后重试。',
            })
    
    return StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


def _sse_data(data: dict) -> str:
    """格式化 SSE 数据块"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# =============================================================
# 视图 3: 电影推荐解释页
# =============================================================
@login_required
def movie_explain_view(request, pk):
    """电影推荐解释页 - 展示推荐某部电影的详细理由（含归因雷达）"""
    movie = get_object_or_404(
        Movie.objects.prefetch_related('genres', 'actors', 'directors'),
        pk=pk
    )
    
    from myapp.recommender.explain import analyze_recommend_reason
    explanation = analyze_recommend_reason(request.user, pk)
    
    # ★ XAI 方案二：多维归因雷达
    from myapp.utils.xai_explainer import build_attribution_radar
    attribution_radar = build_attribution_radar(request.user, pk)
    
    context = {
        'movie': movie,
        'explanation': explanation,
        'attribution_radar': json.dumps(attribution_radar, ensure_ascii=False),
        'confidence_score': attribution_radar.get('confidence_score', 0),
        'page_title': f'推荐理由 - {movie.title}',
    }
    return render(request, 'movie_explain.html', context)


# =============================================================
# 视图 4: 推荐反馈收集 API
# =============================================================
@csrf_exempt
@login_required
@require_POST
def recommend_feedback_view(request):
    """推荐反馈收集接口"""
    movie_id = request.POST.get('movie_id')
    feedback_type = request.POST.get('feedback_type', 'click')
    source = request.POST.get('source', 'recommend_page')
    
    if not movie_id:
        return JsonResponse({'status': 'error', 'msg': '缺少电影ID'})
    
    try:
        from myapp.models_upgrade import UserFeedback
        movie = get_object_or_404(Movie, pk=movie_id)
        
        UserFeedback.objects.create(
            user=request.user,
            movie=movie,
            feedback_type=feedback_type,
            source=source,
        )
        
        return JsonResponse({
            'status': 'success',
            'msg': '感谢您的反馈！',
            'feedback_type': feedback_type,
        })
    except Exception as e:
        logger.error(f"[Feedback] 保存失败: {e}")
        return JsonResponse({'status': 'error', 'msg': str(e)})


# =============================================================
# 视图 5: 对话管理 API
# =============================================================

@csrf_exempt
@login_required
def ajax_clear_chat(request):
    """清除对话历史 + 重置记忆槽位"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    try:
        count = ChatHistory.objects.filter(user=request.user).count()
        ChatHistory.objects.filter(user=request.user).delete()
        
        session_id = f"user_{request.user.id}_{_get_session_key(request)}"
        from myapp.agent.memory import MemoryManager
        memory = MemoryManager(user=request.user, session_id=session_id)
        memory.clear_slots()
        
        return JsonResponse({'status': 'ok', 'cleared_count': count})
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})


@csrf_exempt
@login_required
def ajax_new_chat(request):
    """新建对话：总结当前会话 → 补充画像 → 清除历史"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    try:
        session_id = f"user_{request.user.id}_{_get_session_key(request)}"
        from myapp.agent.memory import MemoryManager
        memory = MemoryManager(user=request.user, session_id=session_id)
        profile_updates = memory.summarize_and_update_profile()
        
        count = ChatHistory.objects.filter(user=request.user).count()
        ChatHistory.objects.filter(user=request.user).delete()
        memory.clear_slots()
        
        return JsonResponse({
            'status': 'ok',
            'profile_updated': profile_updates,
            'cleared_count': count,
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})


@csrf_exempt
@login_required
def ajax_edit_last(request):
    """编辑最近一条用户消息"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    try:
        last_user_msg = ChatHistory.objects.filter(
            user=request.user, role='user'
        ).order_by('-timestamp').first()
        
        if not last_user_msg:
            return JsonResponse({'status': 'error', 'msg': '没有可编辑的消息'})
        
        deleted_count = ChatHistory.objects.filter(
            user=request.user,
            timestamp__gte=last_user_msg.timestamp
        ).delete()[0]
        
        session_id = f"user_{request.user.id}_{_get_session_key(request)}"
        from myapp.agent.memory import MemoryManager
        memory = MemoryManager(user=request.user, session_id=session_id)
        memory.clear_slots()
        
        remaining = ChatHistory.objects.filter(
            user=request.user, role='user'
        ).order_by('timestamp')
        for msg in remaining:
            memory.update_slots(msg.message)
        
        return JsonResponse({
            'status': 'ok',
            'last_message': last_user_msg.message,
            'deleted_count': deleted_count,
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})


@csrf_exempt
@login_required
def ajax_chat_history(request):
    """获取最近N条对话历史"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    try:
        limit = min(int(request.POST.get('limit', 5)), 20)
        
        history_qs = ChatHistory.objects.filter(
            user=request.user
        ).order_by('-timestamp')[:limit]
        
        history_list = [
            {
                'role': msg.role,
                'message': msg.message,
                'timestamp': msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
            }
            for msg in reversed(history_qs)
        ]
        
        return JsonResponse({
            'status': 'ok',
            'history': history_list,
            'total_count': ChatHistory.objects.filter(user=request.user).count(),
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})


@csrf_exempt
@login_required
def ajax_summarize_profile(request):
    """手动触发会话总结 → 画像补充"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    try:
        session_id = f"user_{request.user.id}_{_get_session_key(request)}"
        from myapp.agent.memory import MemoryManager
        memory = MemoryManager(user=request.user, session_id=session_id)
        preferences = memory.summarize_and_update_profile()
        profile_summary = memory.get_profile_summary()
        
        return JsonResponse({
            'status': 'ok',
            'profile_summary': profile_summary,
            'preferences': preferences,
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})


# =============================================================
# 视图 6-8: 管理员 CRUD（UserProfile / AgentTrace / UserFeedback）
# =============================================================

@login_required
@_admin_required_view
def admin_user_profile_list(request):
    from myapp.models_upgrade import UserProfile
    from myapp.utils.pagination import Pagination

    query = request.GET.get('q', '')
    qs = UserProfile.objects.select_related('user').all().order_by('-updated_at')
    if query:
        qs = qs.filter(Q(user__username__icontains=query) | Q(profile_text__icontains=query))

    page_object = Pagination(request, qs, page_size=15)
    return render(request, 'admin_panel/admin_userprofile_list.html', {
        'profiles': page_object.page_queryset,
        'page_string': page_object.html(),
        'query': query,
    })


@login_required
@_admin_required_view
def admin_user_profile_edit(request, pk):
    from myapp.models_upgrade import UserProfile
    profile = get_object_or_404(UserProfile, pk=pk)

    if request.method == 'POST':
        profile.profile_text = request.POST.get('profile_text', '')
        profile.is_cold_start = request.POST.get('is_cold_start') == 'on'
        profile.save()
        messages.success(request, f"用户 {profile.user.username} 的画像已更新")
        return redirect('admin_user_profile_list')

    return render(request, 'admin_panel/admin_userprofile_edit.html', {'profile': profile})


@login_required
@_admin_required_view
def admin_user_profile_delete(request, pk):
    from myapp.models_upgrade import UserProfile
    profile = get_object_or_404(UserProfile, pk=pk)

    if request.method == 'POST':
        username = profile.user.username
        profile.delete()
        messages.success(request, f"用户 {username} 的画像已删除")
        return redirect('admin_user_profile_list')

    return render(request, 'admin_panel/admin_confirm_delete.html', {
        'target_name': f"用户画像: {profile.user.username}",
        'cancel_url': 'admin_user_profile_list'
    })


@login_required
@_admin_required_view
def admin_agent_trace_list(request):
    from myapp.models_upgrade import AgentTrace
    from myapp.utils.pagination import Pagination

    query = request.GET.get('q', '')
    qs = AgentTrace.objects.select_related('user').all().order_by('-created_at')
    if query:
        qs = qs.filter(Q(user__username__icontains=query) | Q(user_input__icontains=query) | Q(intent__icontains=query))

    page_object = Pagination(request, qs, page_size=15)
    return render(request, 'admin_panel/admin_agenttrace_list.html', {
        'traces': page_object.page_queryset,
        'page_string': page_object.html(),
        'query': query,
    })


@login_required
@_admin_required_view
def admin_agent_trace_detail(request, pk):
    from myapp.models_upgrade import AgentTrace
    trace = get_object_or_404(AgentTrace, pk=pk)

    from myapp.utils.xai_explainer import analyze_trace_health
    health_metrics = analyze_trace_health(trace)

    return render(request, 'admin_panel/admin_agenttrace_detail.html', {
        'trace': trace,
        'react_display': trace.to_react_display(),
        'health_metrics': health_metrics,
        'health_json': json.dumps(health_metrics, ensure_ascii=False),
    })


@login_required
@_admin_required_view
def admin_agent_trace_delete(request, pk):
    from myapp.models_upgrade import AgentTrace
    trace = get_object_or_404(AgentTrace, pk=pk)

    if request.method == 'POST':
        trace.delete()
        messages.success(request, "推理链记录已删除")
        return redirect('admin_agent_trace_list')

    return render(request, 'admin_panel/admin_confirm_delete.html', {
        'target_name': f"推理链: {trace.user_input[:30]}",
        'cancel_url': 'admin_agent_trace_list'
    })


@login_required
@_admin_required_view
def admin_user_feedback_list(request):
    from myapp.models_upgrade import UserFeedback
    from myapp.utils.pagination import Pagination

    feedback_type = request.GET.get('type', '')
    qs = UserFeedback.objects.select_related('user', 'movie').all().order_by('-created_at')
    if feedback_type:
        qs = qs.filter(feedback_type=feedback_type)

    page_object = Pagination(request, qs, page_size=20)
    return render(request, 'admin_panel/admin_feedback_list.html', {
        'feedbacks': page_object.page_queryset,
        'page_string': page_object.html(),
        'feedback_type': feedback_type,
        'feedback_types': ['like', 'dislike', 'click', 'skip', 'collect', 'share'],
    })


@login_required
@_admin_required_view
def admin_user_feedback_delete(request, pk):
    from myapp.models_upgrade import UserFeedback
    feedback = get_object_or_404(UserFeedback, pk=pk)

    if request.method == 'POST':
        feedback.delete()
        messages.success(request, "反馈记录已删除")
        return redirect('admin_user_feedback_list')

    return render(request, 'admin_panel/admin_confirm_delete.html', {
        'target_name': f"反馈: {feedback.user.username} - {feedback.movie.title}",
        'cancel_url': 'admin_user_feedback_list'
    })


# =============================================================
# 视图 9: Agent推理链展示页
# =============================================================
@login_required
def agent_trace_view(request, trace_id=None):
    """Agent推理链展示页 - 论文答辩展示用"""
    from myapp.models_upgrade import AgentTrace
    
    if trace_id:
        trace = get_object_or_404(AgentTrace, pk=trace_id)
        if not request.user.is_staff and trace.user != request.user:
            messages.error(request, "无权访问此记录")
            return redirect('agent_trace_list')
        
        return render(request, 'agent_trace_detail.html', {
            'trace': trace,
            'react_display': trace.to_react_display(),
        })
    else:
        if request.user.is_staff:
            traces = AgentTrace.objects.all().order_by('-created_at')[:50]
        else:
            traces = AgentTrace.objects.filter(user=request.user).order_by('-created_at')[:20]
        
        return render(request, 'agent_trace_list.html', {
            'traces': traces,
            'page_title': 'Agent推理链历史',
        })


# =============================================================
# 视图 10: GPU / 推理引擎状态检查 API
# =============================================================
@csrf_exempt
@login_required
@_admin_required_api
def ajax_system_health(request):
    """
    系统健康检查 API — 检查 Ollama / PyTorch / FAISS / Neo4j 状态
    返回各组件的可用性、GPU 信息、模型加载状态等
    """
    health = {
        'timestamp': time.time(),
        'components': {}
    }
    
    # 1. PyTorch + GPU
    try:
        import torch
        cuda_ok = False
        gpu_name = None
        gpu_mem_total = None
        gpu_mem_alloc = None
        cuda_ver = None
        try:
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                cuda_ver = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                # PyTorch 2.x 用 total_memory，旧版用 total_mem
                props = torch.cuda.get_device_properties(0)
                gpu_mem_total = round(getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024 / 1024)
                gpu_mem_alloc = round(torch.cuda.memory_allocated(0) / 1024 / 1024)
        except Exception as cuda_err:
            # CUDA 初始化失败（常见于服务器环境切换后），降级为 CPU 模式
            cuda_ok = False
            logger.warning(f"[Health] CUDA 初始化失败，降级为 CPU: {cuda_err}")

        torch_info = {
            'available': True,
            'version': torch.__version__,
            'cuda_available': cuda_ok,
            'cuda_version': cuda_ver,
            'gpu_name': gpu_name,
            'gpu_memory_total_mb': gpu_mem_total,
            'gpu_memory_allocated_mb': gpu_mem_alloc,
            'device': 'cuda' if cuda_ok else 'cpu',
        }
        health['components']['pytorch'] = torch_info
    except ImportError:
        health['components']['pytorch'] = {'available': False, 'error': 'PyTorch 未安装'}
    except Exception as e:
        health['components']['pytorch'] = {'available': False, 'error': str(e)}
    
    # 2. Ollama
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            health['components']['ollama'] = {
                'available': True,
                'models': models,
                'model_count': len(models),
            }
        else:
            health['components']['ollama'] = {'available': False, 'error': result.stderr[:200]}
    except FileNotFoundError:
        health['components']['ollama'] = {'available': False, 'error': 'Ollama 未安装'}
    except subprocess.TimeoutExpired:
        health['components']['ollama'] = {'available': False, 'error': 'Ollama 响应超时'}
    except Exception as e:
        health['components']['ollama'] = {'available': False, 'error': str(e)}
    
    # 3. FAISS
    try:
        import faiss
        health['components']['faiss'] = {
            'available': True,
            'version': faiss.__version__ if hasattr(faiss, '__version__') else 'unknown',
            'gpu_support': hasattr(faiss, 'GpuIndexFlatL2'),
        }
        # 检查 FAISS 索引文件
        import os
        index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'faiss_movie_index')
        if os.path.exists(index_path):
            files = os.listdir(index_path)
            health['components']['faiss']['index_files'] = files
            health['components']['faiss']['index_path'] = index_path
    except ImportError:
        health['components']['faiss'] = {'available': False, 'error': 'FAISS 未安装'}
    except Exception as e:
        health['components']['faiss'] = {'available': False, 'error': str(e)}
    
    # 4. Neo4j
    try:
        from myapp import views
        neo_graph = getattr(views, 'neo_graph', None)
        if neo_graph:
            # 尝试简单查询
            result = neo_graph.run("MATCH (n) RETURN count(n) AS cnt LIMIT 1").data()
            node_count = result[0]['cnt'] if result else 0
            health['components']['neo4j'] = {
                'available': True,
                'node_count': node_count,
            }
        else:
            health['components']['neo4j'] = {'available': False, 'error': 'Neo4j 未连接'}
    except Exception as e:
        health['components']['neo4j'] = {'available': False, 'error': str(e)}
    
    # 5. MAAN / SKB-FMLP 模型
    try:
        from myapp.agent.movie_agent import MAANRerankTool
        model, meta, _ = MAANRerankTool._load_model()
        health['components']['maan_model'] = {
            'available': model is not None,
            'model_type': type(model).__name__ if model else None,
            'device': str(next(model.parameters()).device) if model and hasattr(model, 'parameters') else 'N/A',
        }
    except Exception as e:
        health['components']['maan_model'] = {'available': False, 'error': str(e)}
    
    # 6. RAG 资源
    try:
        from myapp import views
        rag = getattr(views, 'RAG_RESOURCES', {})
        health['components']['rag'] = {
            'available': bool(rag),
            'keys': list(rag.keys()) if rag else [],
        }
    except Exception as e:
        health['components']['rag'] = {'available': False, 'error': str(e)}
    
    # 总体状态
    available_count = sum(1 for c in health['components'].values() if c.get('available'))
    total_count = len(health['components'])
    health['summary'] = {
        'available': available_count,
        'total': total_count,
        'status': 'healthy' if available_count >= total_count - 1 else 'degraded',
    }
    
    return JsonResponse(health)


# =============================================================
# 视图 11: Agent Debug 模式 API
# =============================================================
@csrf_exempt
@login_required
@_admin_required_api
def ajax_agent_debug(request):
    """
    Agent Debug API — 返回详细的推理诊断信息
    包含：记忆槽位、候选池统计、工具调用详情、性能指标
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
    
    raw_input = request.POST.get('msg', '').strip()
    if not raw_input:
        return JsonResponse({'error': '缺少输入'}, status=400)
    
    user_input = _sanitize_input(raw_input)
    
    try:
        from myapp.services.agent_chat_service import AgentChatService
        from myapp.agent.memory import MemoryManager
        
        service = AgentChatService(request.user, _get_session_key(request))
        
        # 获取记忆状态
        session_id = f"user_{request.user.id}_{_get_session_key(request)}"
        memory = MemoryManager(user=request.user, session_id=session_id)
        slots = memory.get_slots()
        
        # 执行推理并获取完整trace
        agent = service._get_agent()
        result = agent.run(user_input)
        
        # 构建debug信息
        debug_info = {
            'input': user_input,
            'intent': result['intent'],
            'memory_slots': slots,
            'memory_summary': memory.get_memory_summary(),
            'trace_steps': result.get('trace_steps', []),
            'react_iterations': result.get('react_iterations', 0),
            'candidate_pool_size': len([
                s for s in result.get('trace_steps', [])
                if s.get('type') == 'observation'
            ]),
            'tools_used': list(set(
                s.get('tool', '') for s in result.get('trace_steps', [])
                if s.get('tool')
            )),
            'actions': result.get('actions', []),
            'observations': [
                {'tool': o.get('tool', ''), 'count': o.get('count', 0), 'stats': o.get('stats', {})}
                for o in result.get('observations', [])
            ],
            'recommended_ids': result.get('recommended_ids', []),
            'latency_ms': result.get('latency_ms', 0),
            'final_answer': result.get('final_answer', ''),
            'need_clarification': result.get('need_clarification', False),
        }
        
        return JsonResponse(debug_info)
    
    except Exception as e:
        logger.exception(f"[Debug] Agent调试异常: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# =============================================================
# 视图 12: Agent 前端 - 知识图谱可视化页
# =============================================================
@login_required
def agent_kg_view(request):
    """Agent 前端知识图谱可视化页面"""
    query = request.GET.get('q', '').strip()
    movie_id = request.GET.get('movie_id', '').strip()
    return render(request, 'agent_kg.html', {
        'query': query,
        'movie_id': movie_id,
    })


@csrf_exempt
@login_required
def ajax_agent_kg_query(request):
    """
    Agent 前端知识图谱查询 API
    模式: overview | query | movie
    """
    from myapp import views
    _g = getattr(views, 'neo_graph', None)
    if _g is None:
        return JsonResponse({
            "nodes": [], "links": [], "categories": [],
            "triples": [], "error": "Neo4j 未连接"
        })

    try:
        query = request.GET.get('q', '').strip()
        movie_id = request.GET.get('movie_id', '').strip()
        mode = request.GET.get('mode', 'overview')

        if movie_id:
            mode = 'movie'
        elif query:
            mode = 'query'

        nodes_dict = {}
        links_set = set()
        triples = []
        seen_mids = set()

        categories = [
            {"name": "电影", "itemStyle": {"color": "#d9534f"}},
            {"name": "导演", "itemStyle": {"color": "#9b59b6"}},
            {"name": "演员", "itemStyle": {"color": "#f0ad4e"}},
            {"name": "类型", "itemStyle": {"color": "#5cb85c"}},
            {"name": "地区", "itemStyle": {"color": "#5bc0de"}},
        ]

        def add_node(nid, name, category, size=25):
            if nid not in nodes_dict:
                node = {
                    "id": nid, "name": name, "category": category,
                    "symbolSize": size, "draggable": True,
                }
                # 电影节点可点击跳转到详情页
                if category == 0 and nid.startswith('m_'):
                    try:
                        mid = nid.split('_', 1)[1]
                        node["url"] = f"/movie/{mid}/"
                    except (IndexError, ValueError):
                        pass
                nodes_dict[nid] = node
            return nid

        def add_link(src, tgt, label, **kwargs):
            link = {"source": src, "target": tgt, "value": label}
            link.update(kwargs)
            links_set.add(json.dumps(link, sort_keys=True))

        # ── 模式 A: 全量概览 ──
        if mode == 'overview':
            rows = _g.run("""
                MATCH (m:Movie)<-[:DIRECTED_BY]-(d:Person)
                WITH m, d LIMIT 150
                OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
                RETURN m.mid AS mid, m.title AS mtitle, d.name AS dname, g.name AS gname
                LIMIT 300
            """).data()
            for r in rows:
                mid = r.get('mid')
                if mid is None: continue
                m_id = f"m_{mid}"
                add_node(m_id, r.get('mtitle', f'Movie#{mid}'), 0, 30)
                if r.get('dname'):
                    d_id = f"d_{r['dname']}"
                    add_node(d_id, r['dname'], 1, 20)
                    add_link(d_id, m_id, "导演")
                if r.get('gname'):
                    g_id = f"g_{r['gname']}"
                    add_node(g_id, r['gname'], 3, 15)
                    add_link(m_id, g_id, "类型")

        # ── 模式 B: 关键词查询 ──
        elif mode == 'query':
            # 路径1: 标题/简介匹配
            for r in _g.run("""
                MATCH (m:Movie)<-[:DIRECTED_BY]-(d:Person)
                WHERE m.title CONTAINS $q OR m.summary CONTAINS $q
                OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
                RETURN m.mid AS mid, m.title AS title, m.score AS score,
                       d.name AS director, collect(DISTINCT g.name) AS genres
                ORDER BY m.score DESC LIMIT 10
            """, q=query[:15]).data():
                mid = r.get('mid')
                if mid in seen_mids: continue
                seen_mids.add(mid)
                m_id = f"m_{mid}"
                add_node(m_id, r.get('title', ''), 0, min(50, 20 + int(r.get('score', 0) or 0)))
                if r.get('director'):
                    d_id = f"d_{r['director']}"
                    add_node(d_id, r['director'], 1, 22)
                    add_link(d_id, m_id, "导演")
                    triples.append(f"《{r['title']}》(ID:{mid})--[导演]-->{r['director']}")
                for gname in (r.get('genres') or []):
                    if gname:
                        g_id = f"g_{gname}"
                        add_node(g_id, gname, 3, 15)
                        add_link(m_id, g_id, "类型")

            # 路径2: 类型匹配
            for r in _g.run("""
                MATCH (m:Movie)-[:BELONGS_TO]->(g:Genre)
                WHERE g.name CONTAINS $q
                OPTIONAL MATCH (m)<-[:DIRECTED_BY]-(d:Person)
                RETURN m.mid AS mid, m.title AS title, g.name AS genre, d.name AS director
                ORDER BY m.score DESC LIMIT 8
            """, q=query[:10]).data():
                mid = r.get('mid')
                if mid in seen_mids: continue
                seen_mids.add(mid)
                m_id = f"m_{mid}"
                add_node(m_id, r.get('title', ''), 0, 25)
                g_id = f"g_{r['genre']}"
                add_node(g_id, r['genre'], 3, 15)
                add_link(m_id, g_id, "类型")
                triples.append(f"《{r['title']}》(ID:{mid})--[类型]-->{r['genre']}")

            # 路径3: 演员匹配
            for r in _g.run("""
                MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
                WHERE a.name CONTAINS $q
                OPTIONAL MATCH (m)<-[:DIRECTED_BY]-(d:Person)
                RETURN a.name AS actor, m.title AS title, m.mid AS mid, d.name AS director
                ORDER BY m.score DESC LIMIT 8
            """, q=query[:10]).data():
                mid = r.get('mid')
                if mid in seen_mids: continue
                seen_mids.add(mid)
                m_id = f"m_{mid}"
                add_node(m_id, r.get('title', ''), 0, 25)
                a_id = f"a_{r['actor']}"
                add_node(a_id, r['actor'], 2, 20)
                add_link(a_id, m_id, "主演")
                triples.append(f"《{r['title']}》(ID:{mid})--[主演]-->{r['actor']}")

            # 路径4: 导演匹配
            for r in _g.run("""
                MATCH (d:Person)-[:DIRECTED_BY]->(m:Movie)
                WHERE d.name CONTAINS $q
                OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
                RETURN d.name AS director, m.title AS title, m.mid AS mid,
                       collect(DISTINCT g.name) AS genres
                ORDER BY m.score DESC LIMIT 10
            """, q=query[:10]).data():
                mid = r.get('mid')
                if mid in seen_mids: continue
                seen_mids.add(mid)
                m_id = f"m_{mid}"
                add_node(m_id, r.get('title', ''), 0, 25)
                d_id = f"d_{r['director']}"
                add_node(d_id, r['director'], 1, 25)
                add_link(d_id, m_id, "导演")
                triples.append(f"《{r['title']}》(ID:{mid})--[导演]-->{r['director']}")

        # ── 模式 C: 电影详情（重点：电影间的关联拓扑）──
        elif mode == 'movie':
            try:
                target_mid = int(movie_id)
            except (ValueError, TypeError):
                return JsonResponse({"nodes": [], "links": [], "categories": categories, "triples": [], "error": "无效的 movie_id"})

            history_mids = []
            if request.user.is_authenticated:
                history_mids = list(
                    UserRating.objects.filter(user=request.user, score__gte=7.5)
                    .order_by('?').values_list('movie_id', flat=True)[:15]
                )

            # ── 查询 1: 中心电影的属性节点 ──
            rows = _g.run("""
                MATCH (target:Movie {mid: $mid})
                OPTIONAL MATCH (target)<-[:DIRECTED_BY]-(d:Person)
                WITH target, collect({node: d, type: 'director'}) AS directors
                OPTIONAL MATCH (target)<-[:ACTED_IN]-(a:Person)
                WITH target, directors, collect({node: a, type: 'actor'}) AS actors
                OPTIONAL MATCH (target)-[:BELONGS_TO]->(g:Genre)
                WITH target, directors, actors, collect({node: g, type: 'genre'}) AS genres
                OPTIONAL MATCH (target)-[:RELEASED_IN]->(r:Region)
                WITH target, directors, actors, genres, collect({node: r, type: 'region'}) AS regions
                WITH target, directors[..3] + actors[..3] + genres[..3] + regions[..1] AS attrs
                UNWIND attrs AS item
                WITH target, item.node AS attr, item.type AS attr_type
                OPTIONAL MATCH (attr)--(h:Movie)
                WHERE h.mid IN $hist_mids AND h.mid <> $mid
                RETURN target, attr, attr_type, h AS history
            """, mid=target_mid, hist_mids=history_mids).data()

            # ── 查询 2: 通过共享导演/演员/类型找到关联电影 ──
            related_rows = _g.run("""
                MATCH (target:Movie {mid: $mid})
                // 同导演的其他电影
                OPTIONAL MATCH (target)<-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]->(m1:Movie)
                WHERE m1.mid <> $mid
                WITH target, collect(DISTINCT {movie: m1, via: d.name, rel: '同导演'})[..3] AS by_dir
                // 同主演的其他电影
                OPTIONAL MATCH (target)<-[:ACTED_IN]-(a:Person)-[:ACTED_IN]->(m2:Movie)
                WHERE m2.mid <> $mid
                WITH target, by_dir, collect(DISTINCT {movie: m2, via: a.name, rel: '同主演'})[..3] AS by_act
                // 同类型的其他电影（取高分）
                OPTIONAL MATCH (target)-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(m3:Movie)
                WHERE m3.mid <> $mid
                WITH target, by_dir, by_act, collect(DISTINCT {movie: m3, via: g.name, rel: '同类型'})[..3] AS by_genre
                RETURN by_dir + by_act + by_genre AS related
            """, mid=target_mid).data()

            WEIGHT_MAP = {'director': 5, 'actor': 2, 'genre': 3, 'region': 1}
            director_names = set()
            candidate_rows = []

            for row in rows:
                attr = row['attr']
                if not attr: continue
                attr_type = row.get('attr_type', '')
                attr_name = attr.get('name', 'Unknown')
                if attr_type == 'director':
                    director_names.add(attr_name)
                elif attr_type == 'actor' and attr_name in director_names:
                    continue
                candidate_rows.append((WEIGHT_MAP.get(attr_type, 0), row))

            candidate_rows.sort(key=lambda x: x[0], reverse=True)
            candidate_rows = candidate_rows[:12]

            t_id = f"m_{target_mid}"
            title = ""
            added_history = set()
            related_movie_ids = set()  # 追踪已添加的关联电影

            # 添加中心电影属性节点
            for weight, row in candidate_rows:
                target = row['target']
                attr = row['attr']
                history = row['history']
                attr_type = row.get('attr_type', '')
                attr_name = attr.get('name', 'Unknown')

                cat_map = {'director': (1, "导演"), 'actor': (2, "主演"), 'genre': (3, "类型"), 'region': (4, "地区")}
                if attr_type not in cat_map: continue
                acat, link_name = cat_map[attr_type]

                title = title or target.get('title', 'Unknown')
                add_node(t_id, title, 0, 60)
                a_id = f"e_{attr.identity}"
                add_node(a_id, attr_name, acat, {1: 30, 2: 22, 3: 18, 4: 14}.get(acat, 20))
                add_link(a_id, t_id, link_name)
                triples.append(f"《{title}》--[{link_name}]-->{attr_name}")

                if history and len(added_history) < 2:
                    h_mid = history.get('mid')
                    hid = f"m_{h_mid}"
                    if hid not in added_history:
                        add_node(hid, history.get('title', ''), 0, 35)
                        add_link(hid, a_id, "关联")
                        added_history.add(h_mid)

            # ── 添加关联电影间的直接连线（核心：电影→电影拓扑）──
            if related_rows and related_rows[0].get('related'):
                for item in related_rows[0]['related']:
                    r_movie = item.get('movie')
                    if not r_movie: continue
                    r_mid = r_movie.get('mid')
                    if not r_mid or r_mid == target_mid: continue
                    r_id = f"m_{r_mid}"
                    r_title = r_movie.get('title', f'Movie#{r_mid}')
                    via = item.get('via', '')
                    rel = item.get('rel', '关联')

                    # 添加关联电影节点（比属性节点更大）
                    if r_id not in related_movie_ids:
                        score = r_movie.get('score', 0) or 0
                        node_size = max(30, min(50, 20 + int(score)))
                        add_node(r_id, r_title, 0, node_size)
                        related_movie_ids.add(r_id)

                    # 直接电影→电影连线（高权重）
                    add_link(t_id, r_id, rel)
                    triples.append(f"《{title}》--[{rel}:{via}]-->《{r_title}》")

        # ── 构建响应 ──
        links_list = [json.loads(lj) for lj in links_set]
        for link in links_list:
            val = link.get('value', '')
            if val == '导演':
                link['lineStyle'] = {"width": 2.5, "color": "#9b59b6", "opacity": 0.9}
                link['emphasis'] = {"lineStyle": {"width": 5}}
            elif '同导演' in val:
                link['lineStyle'] = {"width": 2.5, "color": "#9b59b6", "opacity": 0.85, "type": "dashed"}
            elif '同主演' in val:
                link['lineStyle'] = {"width": 2, "color": "#f0ad4e", "opacity": 0.8, "type": "dashed"}
            elif '同类型' in val:
                link['lineStyle'] = {"width": 2, "color": "#5cb85c", "opacity": 0.8, "type": "dashed"}

        return JsonResponse({
            "nodes": list(nodes_dict.values()),
            "links": links_list,
            "categories": categories,
            "triples": triples[:20],
            "mode": mode,
            "query": query,
            "movie_id": movie_id,
            "stats": {
                "node_count": len(nodes_dict),
                "link_count": len(links_list),
                "triple_count": len(triples),
            }
        })

    except Exception as e:
        logger.exception(f"[KG] 图谱查询异常: {e}")
        return JsonResponse({
            "nodes": [], "links": [], "categories": [],
            "triples": [], "error": str(e)
        })