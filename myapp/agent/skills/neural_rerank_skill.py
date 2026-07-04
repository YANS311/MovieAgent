"""
NeuralRerankSkill — 神经网络精排 Skill
=================================================
封装 MAAN / SKB-FMLP 深度模型精排。
对应原有 AgentTool: MAANRerankTool

安全设计：
  - 模型文件不存在时返回 fallback 结果，不报错
  - 权重加载失败时自动降级为规则排序
=================================================
"""

import logging
from .base import BaseSkill

logger = logging.getLogger('movie_agent')


class NeuralRerankSkill(BaseSkill):
    """MAAN 多模态注意力网络精排。"""

    name = "maan_rerank"
    description = "使用 MAAN 深度多模态模型对候选电影精排（模型不可用时自动降级）"

    input_schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "description": "候选电影列表（含 movie_id 和 score）",
            },
            "user": {"description": "用户对象（可选）"},
            "top_k": {"type": "integer", "description": "精排后保留数量", "default": 10},
        },
        "required": ["candidates"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array", "description": "精排后候选列表"},
            "model_used": {"type": "string", "description": "实际使用的模型"},
            "count": {"type": "integer"},
        },
    }

    _model_available = None  # 类级别缓存，避免重复检测

    def can_handle(self, context: dict) -> bool:
        candidates = context.get('candidates', [])
        return len(candidates) > 0

    def run(self, context: dict) -> dict:
        import time
        t0 = time.time()

        candidates = context.get('candidates', [])
        user = context.get('user')
        top_k = context.get('top_k', 10)

        # 检测模型是否可用
        if self._model_available is None:
            self._model_available = self._check_model_files()

        if self._model_available:
            try:
                result = self._run_neural(candidates, user, top_k)
                elapsed = time.time() - t0
                result['meta']['elapsed'] = f"{elapsed:.3f}s"
                return result
            except Exception as e:
                logger.warning(f"[NeuralRerankSkill] 模型推理失败，降级: {e}")

        # 降级：按原始 score 排序
        return self.fallback(context, Exception("模型不可用"))

    def _run_neural(self, candidates, user, top_k) -> dict:
        """调用原有 MAANRerankTool。"""
        from myapp.agent.movie_agent import MAANRerankTool
        tool = MAANRerankTool()
        result = tool.execute(candidates=candidates, user=user, top_k=top_k)
        return {
            'skill': self.name,
            'success': True,
            'data': result.get('output', []),
            'meta': {
                'model_used': result.get('model', 'MAAN'),
                'count': result.get('count', 0),
            },
        }

    def fallback(self, context: dict, error: Exception) -> dict:
        """降级：按原始 score 排序，截取 top_k。"""
        candidates = context.get('candidates', [])
        top_k = context.get('top_k', 10)

        # 按 score 降序排序
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('score', 0),
            reverse=True,
        )[:top_k]

        return {
            'skill': self.name,
            'success': True,
            'data': sorted_candidates,
            'meta': {
                'fallback': True,
                'model_used': 'score_sort',
                'count': len(sorted_candidates),
                'error': str(error),
            },
        }

    @staticmethod
    def _check_model_files() -> bool:
        """检查模型权重文件是否存在。"""
        import os
        from django.conf import settings

        base = getattr(settings, 'BASE_DIR', os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

        # 检查常见模型路径
        model_paths = [
            os.path.join(base, 'deepfm_best.pth'),
            os.path.join(base, 'deepfm_sota.pth'),
            os.path.join(base, 'local_models', 'maan_best.pth'),
            os.path.join(base, 'ml_artifacts', 'model.pth'),
        ]

        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"[NeuralRerankSkill] 找到模型权重: {path}")
                return True

        logger.info("[NeuralRerankSkill] 未找到模型权重，将使用降级排序")
        return False
