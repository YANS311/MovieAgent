import torch
import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN


class FeatureSelection(nn.Module):
    """
    Feature Selection (Robust Version)
    兼容混合维度特征 (Sparse Embedding + Dense Input)
    """

    def __init__(self, num_fields, dropout=0.0):
        super(FeatureSelection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate_gen = None

    def build_gate(self, total_input_dim, num_fields, device):
        if self.gate_gen is None:
            self.gate_gen = nn.Sequential(
                nn.Linear(total_input_dim, num_fields),
                nn.BatchNorm1d(num_fields),
                nn.Sigmoid()
            ).to(device)

    def forward(self, feature_list, flat_input):
        batch_size = flat_input.size(0)
        num_fields = len(feature_list)

        if self.gate_gen is None:
            self.build_gate(flat_input.size(1), num_fields, flat_input.device)

        gates = self.gate_gen(flat_input)
        gates = self.dropout(gates)  # 在权重层应用 Dropout

        weighted_list = []
        for i, feat in enumerate(feature_list):
            g = gates[:, i].unsqueeze(1)
            if feat.dim() == 3:
                g = g.unsqueeze(2)
            weighted_list.append(feat * g)
        return weighted_list


class MultiHeadFusion(nn.Module):
    """
    Multi-Head Bilinear Fusion (AAAI 2023)
    """

    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.w_x = nn.Linear(input_dim, input_dim)
        self.w_y = nn.Linear(input_dim, input_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads, self.head_dim, self.head_dim))
        nn.init.xavier_normal_(self.w_xy)
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x, y):
        p_x = self.w_x(x)
        p_y = self.w_y(y)
        batch_size = x.shape[0]
        x_heads = p_x.view(batch_size, self.num_heads, self.head_dim)
        y_heads = p_y.view(batch_size, self.num_heads, self.head_dim)

        # Bilinear interaction: x^T * W * y
        interact = torch.einsum('bhk,hkk,bhk->bh', x_heads, self.w_xy, y_heads)
        # 将 interact 映射回 input_dim 或者直接 view (取决于具体实现，这里保持与你之前逻辑一致的简单平铺)
        interact_flat = (x_heads * y_heads).view(batch_size, -1)

        out = p_x + p_y + interact_flat
        return self.bn(out)


class FinalMLP(BaseModel):
    """
    FinalMLP (Asymmetric Version)
    支持 mlp1_dropout 和 mlp2_dropout 独立配置
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns1, dnn_feature_columns2,
                 mlp1_hidden_units=(256, 128), mlp2_hidden_units=(256, 128),
                 mlp1_dropout=0.2, mlp2_dropout=0.5, num_heads=4,
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-3, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_activation='relu',
                 dnn_use_bn=True, task='binary', device='cpu', gpus=None):
        # 将两个流的特征汇总给 BaseModel 用于初始化 Embedding 层
        all_dnn_cols = dnn_feature_columns1 + dnn_feature_columns2
        # 去重保留顺序
        unique_dnn = []
        [unique_dnn.append(x) for x in all_dnn_cols if x not in unique_dnn]

        super(FinalMLP, self).__init__(linear_feature_columns, unique_dnn,
                                       l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
                                       init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.dnn_feature_columns1 = dnn_feature_columns1
        self.dnn_feature_columns2 = dnn_feature_columns2

        # 计算各自的输入维度
        input_dim1 = self.compute_input_dim(dnn_feature_columns1)
        input_dim2 = self.compute_input_dim(dnn_feature_columns2)

        # Stream 1: 协同/ID 流
        self.mlp1 = DNN(input_dim1, mlp1_hidden_units, activation=dnn_activation,
                        l2_reg=l2_reg_dnn, dropout_rate=mlp1_dropout, use_bn=dnn_use_bn)

        # Stream 2: 语义/增强 流
        self.mlp2 = DNN(input_dim2, mlp2_hidden_units, activation=dnn_activation,
                        l2_reg=l2_reg_dnn, dropout_rate=mlp2_dropout, use_bn=dnn_use_bn)

        self.fusion = MultiHeadFusion(mlp1_hidden_units[-1], num_heads=num_heads)
        self.final_linear = nn.Linear(mlp1_hidden_units[-1], 1)
        self.to(device)

    def forward(self, X):
        # 分别提取两组特征的 Embedding
        emb1, den1 = self.input_from_feature_columns(X, self.dnn_feature_columns1, self.embedding_dict)
        y1 = self.mlp1(combined_dnn_input(emb1, den1))

        emb2, den2 = self.input_from_feature_columns(X, self.dnn_feature_columns2, self.embedding_dict)
        y2 = self.mlp2(combined_dnn_input(emb2, den2))

        fusion_out = self.fusion(y1, y2)
        logit = self.final_linear(fusion_out) + self.linear_model(X)
        return self.out(logit)