"""
MMAN: Multi-Modal Attention Network
================================================
多模态注意力推荐网络 — 替代 SKB-FMLP 的论文核心模型

架构创新点（相对 SKB-FMLP 的升级）：
  1. 模态分解 (Modality Decomposition)
     将 RAG 融合向量拆分为 Text / Visual 两个独立模态分支
  2. 跨模态注意力 (Cross-Modal Attention)
     Text ↔ Visual ↔ KG 三路交叉注意力融合
  3. 行为序列注意力 (Behavior-Aware DIN Attention)
     用户历史序列与目标电影的注意力加权
  4. 门控多模态融合 (Gated Multi-Modal Fusion)
     动态门控融合不同模态的贡献

超参配置 (论文实验配置)：
  - dropout = 0.1
  - PCA dim = 128 (UNIFIED_EMBED_DIM)
  - Text dim = 64, Visual dim = 64
================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepctr_torch.inputs import combined_dnn_input, SparseFeat, VarLenSparseFeat
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    Query 来自一个模态，Key/Value 来自另一个模态
    实现: Q·K^T / sqrt(d) → softmax → ·V
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        """
        Args:
            query:    [batch, dim]
            key_value: [batch, dim]
        Returns:
            [batch, dim]  跨模态融合后的表示
        """
        # 扩展维度以适配多头注意力: [batch, 1, dim]
        q = self.W_q(query).unsqueeze(1)
        k = self.W_k(key_value).unsqueeze(1)
        v = self.W_v(key_value).unsqueeze(1)

        # 多头拆分
        B = q.size(0)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)

        # 合并多头: [B, num_heads, 1, head_dim] → [B, 1, embed_dim] → [B, embed_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        out = self.W_o(out.squeeze(1))  # [B, embed_dim]

        # 残差连接 + LayerNorm
        return self.layer_norm(query + out)


class ModalityEncoder(nn.Module):
    """
    单模态编码器：MLP + Dropout + LayerNorm
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.layer_norm(self.net(x))


class MMAN(BaseModel):
    """
    Multi-Modal Attention Network (MMAN)
    
    相对 SKB-FMLP 的核心升级：
    1. 三路模态编码器（Text / Visual / KG）替代单一 MLP
    2. 跨模态注意力融合替代简单向量门控
    3. 保留 DIN 行为序列注意力
    4. 统一 dropout=0.1
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 history_feature_list,
                 text_dim=64, visual_dim=64,
                 hidden_dim=256, num_heads=4,
                 dropout=0.1, use_demographic=False, **kwargs):
        super(MMAN, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)

        self.history_feat_names = history_feature_list
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.embed_dim = self.embedding_dict[history_feature_list[0]].embedding_dim
        self.use_demographic = use_demographic

        # ==========================================
        # 模块 1: 行为序列注意力 (DIN-style)
        # ==========================================
        self.att_dnn = DNN(
            inputs_dim=4 * self.embed_dim,
            hidden_units=(256, 128),
            activation='relu',
            dropout_rate=dropout,
            device=self.device
        )
        self.att_linear = nn.Linear(128, 1)

        # ==========================================
        # 模块 2: 三路模态编码器
        # ==========================================
        # Text 编码器：处理 RAG 中的文本 PCA 特征 (前 text_dim 维)
        self.text_encoder = ModalityEncoder(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=self.embed_dim,
            dropout=dropout
        )

        # Visual 编码器：处理 RAG 中的视觉特征 (后 visual_dim 维)
        self.visual_encoder = ModalityEncoder(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
            output_dim=self.embed_dim,
            dropout=dropout
        )

        # KG 编码器：处理 Genre + Actor + Director 的聚合嵌入
        # 输入: genres_emb(128) + actors_emb(128) + directors_emb(128) = 384
        kg_input_dim = self.embed_dim * 3
        self.kg_encoder = ModalityEncoder(
            input_dim=kg_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.embed_dim,
            dropout=dropout
        )

        # ==========================================
        # 模块 3: 跨模态注意力 (Cross-Modal Attention)
        # ==========================================
        # Text → Visual 注意力
        self.cross_attn_tv = CrossModalAttention(self.embed_dim, num_heads, dropout)
        # Visual → KG 注意力
        self.cross_attn_vk = CrossModalAttention(self.embed_dim, num_heads, dropout)
        # KG → Text 注意力
        self.cross_attn_kt = CrossModalAttention(self.embed_dim, num_heads, dropout)

        # ==========================================
        # 模块 4: 门控多模态融合
        # ==========================================
        # 三路模态融合后的总维度
        fusion_dim = self.embed_dim * 3

        # 门控网络：动态学习各模态的贡献权重
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1)
        )

        # ==========================================
        # 模块 5: 人口统计学特征编码器（可选）
        # ==========================================
        if self.use_demographic:
            # occupation: 21 类，嵌入维度 8；sex: 3 类，嵌入维度 4；age_norm: 1 维连续
            self.demo_dim = 8 + 4 + 1  # occupation_emb + sex_emb + age_norm = 13
            self.demo_encoder = ModalityEncoder(
                input_dim=self.demo_dim,
                hidden_dim=64,
                output_dim=self.embed_dim,
                dropout=dropout
            )
            final_input_dim = self.embed_dim * 3  # 多模态 + 行为序列 + 人口统计
        else:
            final_input_dim = self.embed_dim * 2  # 多模态 + 行为序列

        # ==========================================
        # 模块 6: 最终预测层
        # ==========================================
        self.final_dnn = DNN(
            inputs_dim=final_input_dim,
            hidden_units=(256, 128, 64),
            activation='relu',
            dropout_rate=dropout,
            device=self.device
        )
        self.predict_layer = nn.Linear(64, 1, bias=False)

        # Linear 部分 (DeepCTR 通用)
        self.to(self.device)

    def forward(self, X):
        """
        前向传播
        
        Args:
            X: [batch, feature_dim] 拼接后的所有特征
        
        Returns:
            [batch, 1] sigmoid 预测分数
        """
        # ── Step 1: 提取嵌入和稠密特征 ──
        sparse_emb, dense_val = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        # ── Step 2: DIN 行为序列注意力 ──
        q_name = self.history_feat_names[0]
        query_emb = self.embedding_dict[q_name](
            X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long()
        )
        keys_emb = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long()
        )

        T = keys_emb.size(1)
        query_rep = query_emb.expand(-1, T, -1)
        att_input = torch.cat([query_rep, keys_emb, query_rep - keys_emb, query_rep * keys_emb], dim=-1)
        att_score = torch.softmax(
            self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1
        )
        hist_attn = torch.bmm(att_score, keys_emb).squeeze(1)  # [batch, embed_dim]

        # ── Step 3: 模态分解与编码 ──
        # 🔥 修复：deepctr_torch 的 dense_val 是 list of tensors
        # combined_dnn_input 需要 list，但切片需要 tensor，所以分别处理
        dnn_input = combined_dnn_input(sparse_emb, dense_val)

        # 将 dense_val 拼接为单个 tensor 用于模态切片
        if isinstance(dense_val, list):
            dense_tensor = torch.cat(dense_val, dim=-1)
        else:
            dense_tensor = dense_val

        # 提取 RAG 文本特征 (前 text_dim 维)
        rag_text_feat = dense_tensor[:, :self.text_dim]
        # 提取 RAG 视觉特征 (后 visual_dim 维)
        rag_visual_feat = dense_tensor[:, self.text_dim:self.text_dim + self.visual_dim]

        # 模态编码
        text_repr = self.text_encoder(rag_text_feat)      # [batch, embed_dim]
        visual_repr = self.visual_encoder(rag_visual_feat)  # [batch, embed_dim]

        # KG 编码：拼接 Genre + Actor + Director 嵌入
        # 通过 embedding_dict 直接查找，避免索引对不上的问题
        kg_feats = []
        for feat_name in ['genres', 'actors', 'directors']:
            if feat_name in self.embedding_dict:
                idx = self.feature_index[feat_name]
                feat_input = X[:, idx[0]:idx[1]].long()
                emb = self.embedding_dict[feat_name](feat_input)
                kg_feats.append(emb)

        # 🔥 核心修复：VarLenSparseFeat 的嵌入是 3D [batch, maxlen, dim]
        # 需要先对 maxlen 维度做 mean pooling，转为 2D 再拼接
        kg_feats_pooled = []
        for feat_tensor in kg_feats:
            if feat_tensor.dim() == 3:
                # [batch, maxlen, dim] → [batch, dim] (对非零位置取均值)
                kg_feats_pooled.append(feat_tensor.mean(dim=1))
            else:
                kg_feats_pooled.append(feat_tensor)

        if len(kg_feats_pooled) >= 3:
            kg_concat = torch.cat(kg_feats_pooled[:3], dim=-1)  # [batch, embed_dim*3]
        elif len(kg_feats_pooled) > 0:
            kg_concat = torch.cat(kg_feats_pooled + [
                torch.zeros(kg_feats_pooled[0].size(0), self.embed_dim, device=self.device)
                for _ in range(3 - len(kg_feats_pooled))
            ], dim=-1)
        else:
            kg_concat = torch.zeros(dnn_input.size(0), self.embed_dim * 3, device=self.device)

        kg_repr = self.kg_encoder(kg_concat)  # [batch, embed_dim]，此时确保为 2D

        # ── Step 4: 跨模态注意力融合 ──
        # Text ← Visual (文本吸收视觉信息)
        text_enhanced = self.cross_attn_tv(text_repr, visual_repr)
        # Visual ← KG (视觉吸收图谱信息)
        visual_enhanced = self.cross_attn_vk(visual_repr, kg_repr)
        # KG ← Text (图谱吸收文本信息)
        kg_enhanced = self.cross_attn_kt(kg_repr, text_repr)

        # 🔥 防守性修复：强制确保所有张量为 2D [batch, embed_dim]
        if text_enhanced.dim() == 3:
            text_enhanced = text_enhanced.squeeze(1)
        if visual_enhanced.dim() == 3:
            visual_enhanced = visual_enhanced.squeeze(1)
        if kg_enhanced.dim() == 3:
            kg_enhanced = kg_enhanced.squeeze(1)

        # ── Step 5: 门控多模态融合 ──
        modal_concat = torch.cat([text_enhanced, visual_enhanced, kg_enhanced], dim=-1)
        gate_weights = self.gate_net(modal_concat)  # [batch, 3]

        # 加权融合
        text_w = gate_weights[:, 0:1] * text_enhanced
        visual_w = gate_weights[:, 1:2] * visual_enhanced
        kg_w = gate_weights[:, 2:3] * kg_enhanced
        multimodal_repr = text_w + visual_w + kg_w  # [batch, embed_dim]

        # ── Step 6: 人口统计学特征编码（可选）──
        if self.use_demographic:
            demo_feats = []
            for feat_name in ['occupation', 'sex']:
                if feat_name in self.embedding_dict:
                    idx = self.feature_index[feat_name]
                    feat_input = X[:, idx[0]:idx[1]].long()
                    emb = self.embedding_dict[feat_name](feat_input)
                    # 确保是 2D [batch, dim]
                    if emb.dim() == 3:
                        emb = emb.squeeze(1)
                    demo_feats.append(emb)
            age_start = self.text_dim + self.visual_dim
            if dense_tensor.size(1) > age_start:
                age_feat = dense_tensor[:, age_start:age_start + 1]
                demo_feats.append(age_feat)
            if len(demo_feats) > 0:
                demo_concat = torch.cat(demo_feats, dim=-1)
                demo_repr = self.demo_encoder(demo_concat)
            else:
                demo_repr = torch.zeros(multimodal_repr.size(0), self.embed_dim, device=self.device)
            final_input = torch.cat([multimodal_repr, hist_attn, demo_repr], dim=-1)
        else:
            final_input = torch.cat([multimodal_repr, hist_attn], dim=-1)

        # ── Step 7: 最终预测 ──
        final_out = self.final_dnn(final_input)
        logit = self.predict_layer(final_out) + self.linear_model(X)

        return torch.sigmoid(logit)