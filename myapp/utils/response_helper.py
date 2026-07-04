"""
统一 JSON Response 规范模块
─────────────────────────────────────────────────────────
【论文可写点】：统一的 API 响应格式规范，前后端交互协议标准化。

【修改原因】：原接口返回格式不统一，有的用 JsonResponse + status，
            有的用 success/message 混合字段，前端处理困难。
【性能收益】：前端统一解析逻辑，减少分支判断。
"""
from django.http import JsonResponse


def api_success(data=None, message="操作成功"):
    """
    统一成功响应
    
    返回格式：
    {"success": true, "message": "...", "data": ...}
    """
    resp = {"success": True, "message": message}
    if data is not None:
        resp["data"] = data
    return JsonResponse(resp, json_dumps_params={"ensure_ascii": False})


def api_error(error="操作失败", status=200, data=None):
    """
    统一错误响应
    
    返回格式：
    {"success": false, "error": "...", "data": ...}
    """
    resp = {"success": False, "error": error}
    if data is not None:
        resp["data"] = data
    return JsonResponse(resp, status=status, json_dumps_params={"ensure_ascii": False})


def api_chat_response(content: str):
    """
    聊天接口专用响应（保持向后兼容）
    
    【修改原因】：ajax_chat 前端已约定 {"response": "..."} 格式，
                不能改变，但增加 success 字段便于统一处理。
    """
    return JsonResponse({
        "success": True,
        "response": content
    }, json_dumps_params={"ensure_ascii": False})


def api_explain_response(status: str, content: str, **extra):
    """
    推荐解释接口专用响应（保持向后兼容）
    
    【修改原因】：ajax_explain_rec 前端已约定 {status, content} 格式，
                保持兼容但增加 success 字段。
    """
    resp = {"status": status, "content": content, "success": status == "success"}
    resp.update(extra)
    return JsonResponse(resp, json_dumps_params={"ensure_ascii": False})