# model_zoo.py

import torch
from deepctr_torch.models import DeepFM, DCNMix

from finalmlp import FinalMLP


def get_model(model_name, linear_feature_columns, dnn_feature_columns, device='cpu', **kwargs):
    """
    🏭 模型工厂：一键切换 DeepFM / DCN-V2
    :param model_name: 'deepfm' 或 'dcnv2'
    :param kwargs: 接收动态参数，如 l2_reg, dropout 等
    """
    model_name = model_name.lower()

    # === 1. 基础通用配置 (Base Config) ===
    # 这些是所有模型都通用的参数
    common_params = {
        "linear_feature_columns": linear_feature_columns,
        "dnn_feature_columns": dnn_feature_columns,
        "task": 'binary',
        "device": device,
        "l2_reg_embedding": kwargs.get('l2_reg_embedding', 1e-5),  # 默认 1e-5
        "l2_reg_dnn": kwargs.get('l2_reg_dnn', 1e-5),  # 默认 1e-5
        "dnn_dropout": kwargs.get('dnn_dropout', 0.2),  # 默认 0.2
        "seed": 2024,
    }

    # === 2. 模型分支 ===

    # 🏛️ 方案 A: 经典老将 DeepFM (2017)
    if model_name == "deepfm":
        print(f"🚀 Initializing Baseline: DeepFM")
        return DeepFM(
            dnn_hidden_units=kwargs.get('dnn_hidden_units', (256, 128)),  # 默认轻量化
            **common_params
        )
    elif model_name == "WDL":
        from deepctr_torch.models import WDL
        print(f"🚀 Initializing Baseline: WDL")
        return WDL(
            dnn_hidden_units=kwargs.get('dnn_hidden_units', (256, 128)),  # 默认轻量化
            **common_params
        )
    elif model_name == "xDeepFM":
        from deepctr_torch.models import xDeepFM
        print(f"🚀 Initializing Baseline: xDeepFM")
        return xDeepFM(
            dnn_hidden_units=kwargs.get('dnn_hidden_units', (256, 128)),  # 默认轻量化
            cin_layer_size=kwargs.get('cin_layer_size', (128, 128)),  # 默认轻量化
            **common_params
        )
    # 👑 方案 B: 工业新王 DCN-V2 (DCNMix) (2021)
    # 适合 ML-32M 的推荐配置
    elif model_name == "DCN-V2":
        print(f"🔥 Initializing Backbone: DCN-V2 (DCNMix)")
        return DCNMix(
            # --- DCN-V2 特有参数 ---
            cross_num=3,  # 交叉层数：3层通常足够捕捉高阶特征
            num_experts=4,# 混合专家数 (MoE)：4个专家是性价比最高的选择
            low_rank=64,  # 低秩矩阵维度：64 既能压缩参数又能保证表达力

            # --- DNN 部分 ---
            # DCN 的 CrossNet 已经很强了，DNN 可以保持适中
            dnn_hidden_units=kwargs.get('dnn_hidden_units', (1024, 512, 256)),
            **common_params
        )

    elif model_name == 'FinalMLP':
        print("👑 Initializing FinalMLP (2023)")
        return FinalMLP(
            linear_feature_columns, dnn_feature_columns,
            mlp1_hidden_units=(1024, 512, 256),  # Stream 1
            mlp2_hidden_units=(1024, 512, 256),  # Stream 2
            dnn_dropout=0.4,
            **kwargs
        )

    else:
        raise ValueError(f"❌ Unknown model name: {model_name}")