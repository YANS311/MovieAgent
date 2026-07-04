"""
Knowledge Graph Embedding Lookup Module
支持从三元组文件加载预训练的KG嵌入向量
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import pickle


class KGEmbedding(nn.Module):
    """
    Knowledge Graph Embedding Lookup Module

    使用方式：
        kg_embed = KGEmbedding(embed_dim=128, device='cuda')
        kg_embed.load_from_triplets('kg_triplets.csv')
        vectors = kg_embed.get_movie_embeddings([1, 2, 3])
    """

    def __init__(self, embed_dim: int = 128, device: str = 'cpu'):
        """
        Args:
            embed_dim: 嵌入向量维度 (默认: 128)
            device: 计算设备 ('cuda' or 'cpu')
        """
        super(KGEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

        # 存储嵌入向量的字典: {entity_id: embedding_vector}
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        self.movie_to_entities = {}  # movie_id -> [related_entities]
        self.is_initialized = False

    def load_from_triplets(self, triplet_file: str, entity_embed_file: Optional[str] = None,
                          relation_embed_file: Optional[str] = None):
        """
        从三元组文件加载KG嵌入

        Args:
            triplet_file: CSV文件，格式为 (head, relation, tail)
            entity_embed_file: 预训练实体嵌入文件 (.npy or .pkl)
            relation_embed_file: 预训练关系嵌入文件 (.npy or .pkl)
        """
        # 加载三元组
        print(f"📖 Loading KG triplets from {triplet_file}...")
        if not os.path.exists(triplet_file):
            print(f"⚠️ File not found: {triplet_file}. Using random initialization.")
            self._init_random_embeddings()
            return

        triplets = pd.read_csv(triplet_file, names=['head', 'relation', 'tail'])

        # 收集所有实体和关系
        all_entities = set(triplets['head'].unique()) | set(triplets['tail'].unique())
        all_relations = set(triplets['relation'].unique())

        print(f"✅ Loaded {len(triplets)} triplets with {len(all_entities)} entities, {len(all_relations)} relations")

        # 加载或初始化嵌入
        if entity_embed_file and os.path.exists(entity_embed_file):
            self._load_embeddings_from_file(entity_embed_file, all_entities, 'entity')
        else:
            self._init_embeddings_from_triplets(triplets, all_entities, all_relations)

        # 构建关系映射
        for _, row in triplets.iterrows():
            head_id = row['head']
            if head_id not in self.movie_to_entities:
                self.movie_to_entities[head_id] = []
            self.movie_to_entities[head_id].append({
                'relation': row['relation'],
                'tail': row['tail']
            })

        self.is_initialized = True

    def _init_embeddings_from_triplets(self, triplets: pd.DataFrame,
                                      entities, relations):
        """使用TransE方法初始化嵌入向量"""
        print("🔄 Initializing embeddings using TransE method...")

        # 为每个实体生成随机向量
        for entity in entities:
            # 如果entity是movie_id，使用特殊初始化
            if isinstance(entity, int) or (isinstance(entity, str) and entity.isdigit()):
                # 为movie使用特殊的初始化策略
                vec = np.random.normal(0, 0.05, self.embed_dim).astype(np.float32)
            else:
                vec = np.random.normal(0, 0.1, self.embed_dim).astype(np.float32)
            self.entity_embeddings[entity] = torch.FloatTensor(vec).to(self.device)

        # 为关系生成随机向量
        for relation in relations:
            vec = np.random.normal(0, 0.05, self.embed_dim).astype(np.float32)
            self.relation_embeddings[relation] = torch.FloatTensor(vec).to(self.device)

        # TransE迭代优化 (可选，用于更好的初始化)
        self._transe_optimization(triplets, epochs=5)

    def _transe_optimization(self, triplets: pd.DataFrame, epochs: int = 5,
                            learning_rate: float = 0.001):
        """使用TransE损失函数优化嵌入向量"""
        optimizer = torch.optim.Adam(
            list(self.entity_embeddings.values()) +
            list(self.relation_embeddings.values()),
            lr=learning_rate
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for _, row in triplets.iterrows():
                h_id, rel_id, t_id = row['head'], row['relation'], row['tail']

                if h_id not in self.entity_embeddings or t_id not in self.entity_embeddings:
                    continue

                h = self.entity_embeddings[h_id]
                r = self.relation_embeddings[rel_id]
                t = self.entity_embeddings[t_id]

                # TransE损失: ||h + r - t||²
                loss = torch.norm(h + r - t, p=2) ** 2
                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                print(f"  TransE Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(triplets):.6f}")

    def _load_embeddings_from_file(self, embed_file: str, entities, embed_type: str = 'entity'):
        """从文件加载预训练嵌入"""
        print(f"📂 Loading {embed_type} embeddings from {embed_file}...")

        if embed_file.endswith('.npy'):
            embeddings = np.load(embed_file)
        elif embed_file.endswith('.pkl'):
            with open(embed_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {embed_file}")

        if isinstance(embeddings, dict):
            for entity in entities:
                if entity in embeddings:
                    vec = embeddings[entity]
                    if not isinstance(vec, torch.Tensor):
                        vec = torch.FloatTensor(vec)
                    self.entity_embeddings[entity] = vec.to(self.device)

    def _init_random_embeddings(self, num_entities: int = 10000):
        """使用随机向量初始化嵌入"""
        print(f"🎲 Initializing {num_entities} random embeddings...")
        for i in range(1, num_entities + 1):
            vec = np.random.normal(0, 0.1, self.embed_dim).astype(np.float32)
            self.entity_embeddings[i] = torch.FloatTensor(vec).to(self.device)

    def get_movie_embeddings(self, movie_ids: list) -> np.ndarray:
        """
        获取电影的KG嵌入向量

        Args:
            movie_ids: 电影ID列表 [1, 2, 3, ...]

        Returns:
            形状为 (len(movie_ids), embed_dim) 的numpy数组
        """
        if not self.is_initialized:
            raise RuntimeError("KGEmbedding not initialized. Call load_from_triplets() first.")

        embeddings = []
        for mid in movie_ids:
            if mid in self.entity_embeddings:
                emb = self.entity_embeddings[mid]
            else:
                # 使用零向量作为默认值
                emb = torch.zeros(self.embed_dim, device=self.device)

            embeddings.append(emb.cpu().numpy())

        return np.array(embeddings, dtype=np.float32)

    def get_movie_with_context(self, movie_ids: list, aggregation: str = 'mean') -> np.ndarray:
        """
        获取电影及其相关实体的聚合嵌入

        Args:
            movie_ids: 电影ID列表
            aggregation: 聚合方式 ('mean', 'max', 'sum')

        Returns:
            聚合后的嵌入向量
        """
        movie_embeddings = self.get_movie_embeddings(movie_ids)

        # 聚合相关实体的嵌入
        enriched_embeddings = []
        for mid in movie_ids:
            emb = movie_embeddings[len(enriched_embeddings)]

            if mid in self.movie_to_entities and len(self.movie_to_entities[mid]) > 0:
                context_embeddings = []
                for relation_info in self.movie_to_entities[mid]:
                    tail_id = relation_info['tail']
                    if tail_id in self.entity_embeddings:
                        context_embeddings.append(
                            self.entity_embeddings[tail_id].cpu().numpy()
                        )

                if context_embeddings:
                    context_emb = np.array(context_embeddings)
                    if aggregation == 'mean':
                        context_agg = context_emb.mean(axis=0)
                    elif aggregation == 'max':
                        context_agg = context_emb.max(axis=0)
                    else:  # sum
                        context_agg = context_emb.sum(axis=0)

                    # 融合电影嵌入和上下文嵌入
                    emb = (emb + context_agg) / 2

            enriched_embeddings.append(emb)

        return np.array(enriched_embeddings, dtype=np.float32)

    def save(self, save_path: str):
        """保存KG嵌入到文件"""
        save_data = {
            'entity_embeddings': {k: v.cpu().numpy() for k, v in self.entity_embeddings.items()},
            'relation_embeddings': {k: v.cpu().numpy() for k, v in self.relation_embeddings.items()},
            'movie_to_entities': self.movie_to_entities,
            'embed_dim': self.embed_dim
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✅ KG embeddings saved to {save_path}")

    def load(self, load_path: str):
        """从文件加载KG嵌入"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)

        self.embed_dim = save_data['embed_dim']
        self.entity_embeddings = {
            k: torch.FloatTensor(v).to(self.device)
            for k, v in save_data['entity_embeddings'].items()
        }
        self.relation_embeddings = {
            k: torch.FloatTensor(v).to(self.device)
            for k, v in save_data['relation_embeddings'].items()
        }
        self.movie_to_entities = save_data['movie_to_entities']
        self.is_initialized = True
        print(f"✅ KG embeddings loaded from {load_path}")


# 使用示例
if __name__ == "__main__":
    # 创建KG嵌入模块
    kg_embed = KGEmbedding(embed_dim=128, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 选项1: 从三元组文件加载
    # kg_embed.load_from_triplets('kg_triplets.csv')

    # 选项2: 使用随机初始化 (用于演示)
    kg_embed._init_random_embeddings(num_entities=3883)  # ml-1m数据集有3883部电影
    kg_embed.is_initialized = True

    # 获取电影嵌入
    movie_ids = [1, 2, 3, 10, 100]
    embeddings = kg_embed.get_movie_embeddings(movie_ids)
    print(f"Movie embeddings shape: {embeddings.shape}")
    print(f"First embedding: {embeddings[0][:5]}...")  # 打印前5个值

    # 保存和加载
    kg_embed.save('kg_embeddings_cache.pkl')
    # kg_embed.load('kg_embeddings_cache.pkl')

