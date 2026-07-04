# 文件: myapp/utils/vector_rag.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "faiss_movie_index"

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"向量库加载失败: {e}")
    vector_db = None


def query_vector_rag(query_text, k=3):
    if not vector_db: return ""
    try:
        docs = vector_db.similarity_search(query_text, k=k)
        context = ""
        for i, doc in enumerate(docs):
            # 1. 尝试多种可能的 Key (兼容旧索引)
            mid = doc.metadata.get('id') or doc.metadata.get('movie_id') or doc.metadata.get('mid')
            title = doc.metadata.get('title', '未知电影')

            # 2. 强校验：如果没有 ID，直接跳过这条脏数据！
            if not mid:
                continue

            context += f"{i + 1}. 《{title}》(ID:{mid}): {doc.page_content[:100]}...\n"

        return context
    except Exception as e:
        print(f"VectorRAG Error: {e}")
        return ""