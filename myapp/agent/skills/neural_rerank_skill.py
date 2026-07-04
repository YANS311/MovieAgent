"""
NeuralRerankSkill — 神经网络精排 Skill（v2）
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
    version = "2.0.0"
    priority = 85
    latency_level = "high"
    cost_level = "high"
    tags = ["ranking", "neural", "multimodal"]
    examples = [
        {"input": {"candidates": "候选列表"}, "output": "精排后列表"},
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "candidates": {"type": "array"},
            "user": {"description": "用户对象"},
            "top_k": {"type": "integer", "default": 10},
        },
        "required": ["candidates"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "model_used": {"type": "string"},
            "count": {"type": "integer"},
        },
    }

    _model_available = None

    def can_handle(self, context) -> bool:
        if hasattr(context, 'candidate_movies'):
            return len(context.candidate_movies) > 0
        candidates = context.get('candidates', [])
        return len(candidates) > 0

    def run(self, context) -> dict:
        import time
        t0 = time.time()

        if hasattr(context, 'candidate_movies'):
            candidates = context.candidate_movies
            user = context.user
            top_k = context.metadata.get('top_k', 10) if hasattr(context, 'metadata') else 10
        else:
            candidates = context.get('candidates', [])
            user = context.get('user')
            top_k = context.get('top_k', 10)

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

        return self.fallback(context, Exception("模型不可用"))

    def _run_neural(self, candidates, user, top_k) -> dict:
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

    def fallback(self, context, error: Exception) -> dict:
        if hasattr(context, 'candidate_movies'):
            candidates = context.candidate_movies
            top_k = context.metadata.get('top_k', 10) if hasattr(context, 'metadata') else 10
        else:
            candidates = context.get('candidates', [])
            top_k = context.get('top_k', 10)

        sorted_candidates = sorted(
            candidates, key=lambda x: x.get('score', 0), reverse=True,
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
        import os
        from django.conf import settings
        base = getattr(settings, 'BASE_DIR', os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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
