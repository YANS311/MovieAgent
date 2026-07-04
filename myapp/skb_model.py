import torch
import torch.nn as nn
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN


class SKB_FMLP_Online(BaseModel):
    """
    SKB-FMLP (KAG 双轨增强版) 在线推理模型
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list,
                 mlp1_hidden_units=(512, 256), mlp2_hidden_units=(512, 256),
                 att_hidden_units=(512, 256), dnn_dropout=0.1, **kwargs):
        super(SKB_FMLP_Online, self).__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.history_feat_names = history_feature_list

        first_feat = history_feature_list[0]
        self.embed_dim = self.embedding_dict[first_feat].embedding_dim

        # Attention 模块 (捕捉长短程序列兴趣)
        self.att_dnn = DNN(inputs_dim=4 * self.embed_dim, hidden_units=att_hidden_units, activation='relu',
                           device=self.device)
        self.att_linear = nn.Linear(att_hidden_units[-1], 1)

        # 双流 MLP 主干
        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.mlp_sk = DNN(input_dim, mlp1_hidden_units, dropout_rate=dnn_dropout, device=self.device)
        self.mlp_behavior = DNN(input_dim + self.embed_dim, mlp2_hidden_units, dropout_rate=dnn_dropout,
                                device=self.device)

        # 核心创新：向量门控 (Vector Gating) - 用于 RAG 降噪
        self.vector_gate = nn.Sequential(
            nn.Linear(mlp1_hidden_units[-1], mlp1_hidden_units[-1]),
            nn.Sigmoid()
        )
        self.dnn_predict = nn.Linear(mlp1_hidden_units[-1], 1, bias=False)
        self.to(self.device)

    def forward(self, X):
        sparse_emb, dense_val = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        q_name = self.history_feat_names[0]
        query = self.embedding_dict[q_name](X[:, self.feature_index[q_name][0]:self.feature_index[q_name][1]].long())
        keys = self.embedding_dict[q_name](
            X[:, self.feature_index['hist_' + q_name][0]:self.feature_index['hist_' + q_name][1]].long())

        T = keys.size(1)
        query_rep = query.expand(-1, T, -1)
        att_input = torch.cat([query_rep, keys, query_rep - keys, query_rep * keys], dim=-1)
        att_score = torch.softmax(self.att_linear(self.att_dnn(att_input)).transpose(1, 2), dim=-1)
        hist_attn = torch.bmm(att_score, keys).squeeze(1)

        dnn_input = combined_dnn_input(sparse_emb, dense_val)
        sk_out = self.mlp_sk(dnn_input)
        beh_out = self.mlp_behavior(torch.cat([dnn_input, hist_attn], dim=-1))

        gate = self.vector_gate(sk_out)
        fusion_out = gate * beh_out + (1 - gate) * sk_out

        logit = self.dnn_predict(fusion_out) + self.linear_model(X)
        return torch.sigmoid(logit)