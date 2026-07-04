import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from node2vec import Node2Vec
import gc


def main():
    print("--- 🚀 Step 2: Generating RAG & KG Embeddings ---")

    # 1. 读取清洗好的数据
    df = pd.read_csv('ml32m_enhanced_meta.csv')
    # 读取原始 movies.csv 补全 genre 信息
    ml_movies = pd.read_csv('./ml-32m/movies.csv')
    df = pd.merge(df, ml_movies[['movieId', 'genres']], on='movieId', how='left')

    # 建立 ID 映射 (movieId -> Matrix Index)
    movie_ids = df['movieId'].unique()
    id2idx = {mid: i for i, mid in enumerate(movie_ids)}
    num_movies = len(movie_ids)

    print(f"   -> Processing {num_movies} movies...")

    # ==========================
    # Part A: RAG (Text)
    # ==========================
    print("\n[Part A] Generating RAG Embeddings (BERT)...")

    # 构造文本：标题 + 导演 + 简介
    # 这样语义更丰富
    df['text_input'] = "Title: " + df['title'].fillna('') + \
                       ". Director: " + df['director'].fillna('') + \
                       ". Plot: " + df['overview'].fillna('')

    texts = df['text_input'].tolist()

    # 加载模型 (自动下载)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)

    # PCA 降维 (384 -> 32)
    print("   -> PCA Reduction...")
    pca = PCA(n_components=32)
    rag_emb = pca.fit_transform(embeddings)

    np.save('ml32m_rag_emb.npy', rag_emb)
    print("   ✅ RAG Embeddings saved.")

    del model, embeddings, texts
    gc.collect()

    # ==========================
    # Part B: KG (Graph)
    # ==========================
    print("\n[Part B] Generating KG Embeddings (Node2Vec)...")

    G = nx.Graph()

    print("   -> Building Knowledge Graph...")
    for idx, row in df.iterrows():
        mid = row['movieId']
        m_node = f"m_{mid}"

        # 1. 电影 - 导演
        if row['director'] and row['director'] != 'Unknown':
            d_node = f"d_{row['director']}"
            G.add_edge(m_node, d_node)

        # 2. 电影 - 演员 (前3名)
        if isinstance(row['cast'], str):
            actors = row['cast'].split('|')
            for a in actors:
                if a:
                    a_node = f"a_{a}"
                    G.add_edge(m_node, a_node)

        # 3. 电影 - 类型 (Genre)
        if isinstance(row['genres'], str):
            genres = row['genres'].split('|')
            for g in genres:
                if g != '(no genres listed)':
                    g_node = f"g_{g}"
                    G.add_edge(m_node, g_node)

    print(f"   -> Graph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 训练 Node2Vec
    # 这是一个耗时操作，如果太慢可以减小 num_walks
    print("   -> Training Node2Vec (Hold tight)...")
    n2v = Node2Vec(G, dimensions=32, walk_length=8, num_walks=8, workers=1, quiet=False)
    model_n2v = n2v.fit(window=5, min_count=1)

    # 提取向量
    kg_emb = np.zeros((num_movies, 32))
    for mid in movie_ids:
        idx = id2idx[mid]
        node_name = f"m_{mid}"
        if node_name in model_n2v.wv:
            kg_emb[idx] = model_n2v.wv[node_name]
        else:
            kg_emb[idx] = np.random.normal(0, 0.01, 32)

    np.save('ml32m_kg_emb.npy', kg_emb)
    np.save('ml32m_id_map.npy', id2idx)
    print("   ✅ KG Embeddings & ID Map saved.")


if __name__ == "__main__":
    main()