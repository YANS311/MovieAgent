# 文件: myapp/management/commands/build_rag_index.py (Hugging Face 版)

import os
import time
import math
import torch  # 用于检测 GPU
from django.core.management.base import BaseCommand
from myapp.models import Movie
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- ↓↓↓ 1. 关键升级：导入 HuggingFaceEmbeddings ↓↓↓ ---
from langchain_huggingface import HuggingFaceEmbeddings


# --- ↑↑↑ ---

class Command(BaseCommand):
    help = '构建 RAG 向量索引库 (基于本地 Hugging Face BGE 模型)'
    FAISS_INDEX_PATH = "faiss_movie_index"

    def add_arguments(self, parser):
        parser.add_argument(
            '--incremental', action='store_true',
            help='增量模式：仅对新增电影做嵌入编码，合并至已有索引'
        )

    def handle(self, *args, **options):
        if options.get('incremental'):
            self._handle_incremental()
            return
        self._handle_full()

    def _handle_full(self):
        self.stdout.write("--- RAG 索引全量构建开始 (Local Hugging Face) ---")

        # 1. 检测设备 (GPU/CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stdout.write(f"正在使用设备: {device.upper()}")

        # 2. 加载电影数据 (保持不变)
        all_movies = Movie.objects.all()
        if not all_movies.exists():
            self.stderr.write(self.style.ERROR("数据库中没有电影。"))
            return

        langchain_documents = []
        movies_with_summary = 0

        # (数据转换逻辑保持不变)
        for movie in all_movies:
            genres_str = ", ".join([g.name for g in movie.genres.all()])
            actors_str = ", ".join([a.name for a in movie.actors.all()][:10])

            # 🔥 [新增] 提取导演名字
            # 注意：前提是你已经跑过 import_new_movies.py 把导演存进数据库了
            directors_str = "暂无"
            if hasattr(movie, 'directors'):
                d_list = [d.name for d in movie.directors.all()]
                if d_list:
                    directors_str = ", ".join(d_list)


            summary_text = movie.summary or ""
            if len(summary_text) > 20:
                movies_with_summary += 1

            # 构造 "评分" 文本
            score_text = f"{movie.score or 0.0}"

            combined_text = (
                f"标题: {movie.title}\n"
                f"评分: {score_text}\n"
                f"导演: {directors_str}\n"  
                f"类型: {genres_str}\n"
                f"演员: {actors_str}\n"
                f"简介: {summary_text or '暂无简介'}"
            )

            metadata = {
                "movie_id": movie.id,
                "title": movie.title,
                "poster": movie.poster or "",
                "score": float(movie.score or 0.0)
            }
            langchain_documents.append(Document(page_content=combined_text, metadata=metadata))

        self.stdout.write(f"成功转换 {len(langchain_documents)} 部电影。 (含简介: {movies_with_summary})")

        # 3. 分割文档
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = splitter.split_documents(langchain_documents)
        self.stdout.write(f"文档被分割为 {len(split_docs)} 个片段。")

        # --- ↓↓↓ 4. 关键升级：加载 BGE 中文模型 ↓↓↓ ---
        # BAAI/bge-small-zh-v1.5 是目前最强的轻量级中文 Embedding 模型之一
        model_name = "BAAI/bge-small-zh-v1.5"
        self.stdout.write(f"正在加载 Hugging Face 模型: {model_name} ...")

        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},  # 使用 GPU
                encode_kwargs={'normalize_embeddings': True}  # 归一化，提升余弦相似度效果
            )
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"模型加载失败: {e}"))
            self.stderr.write("请检查网络，Hugging Face 首次运行需要下载模型。")
            return
        # --- ↑↑↑ ---

        # 5. 创建索引 (本地运行，无需分批休眠)
        self.stdout.write("正在创建 FAISS 索引 (本地计算)...")
        start_time = time.time()

        # 因为是本地，我们可以一次性处理更多，或者直接全部处理
        # 如果显存不够，FAISS 会自动处理，或者我们可以分批但不需要 sleep
        vector_db = FAISS.from_documents(split_docs, embedding_model)

        end_time = time.time()
        self.stdout.write(f"索引创建成功！耗时: {end_time - start_time:.2f} 秒。")

        vector_db.save_local(self.FAISS_INDEX_PATH)
        self.stdout.write(self.style.SUCCESS(f"--- RAG 索引已构建并保存到 '{self.FAISS_INDEX_PATH}' ---"))

    def _handle_incremental(self):
        """增量模式：仅对新增电影做嵌入编码，合并至已有索引"""
        import pickle

        self.stdout.write("--- RAG 索引增量构建开始 ---")

        # 1. 检查已有索引是否存在
        index_path = self.FAISS_INDEX_PATH
        index_file = os.path.join(index_path, "index.faiss")
        if not os.path.exists(index_file):
            self.stderr.write(self.style.ERROR(
                f"已有索引不存在于 '{index_path}'，请先执行全量构建。"
            ))
            return

        # 2. 加载已有索引，获取已索引的电影 ID
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "BAAI/bge-small-zh-v1.5"
        self.stdout.write(f"加载 Embedding 模型: {model_name}")

        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"模型加载失败: {e}"))
            return

        self.stdout.write("加载已有 FAISS 索引...")
        existing_db = FAISS.load_local(
            index_path, embedding_model,
            allow_dangerous_deserialization=True
        )

        # 从已有索引的 docstore 中提取已索引的电影 ID
        indexed_ids = set()
        for doc_id, doc in existing_db.docstore._dict.items():
            mid = doc.metadata.get('movie_id') or doc.metadata.get('id')
            if mid:
                indexed_ids.add(int(mid))

        self.stdout.write(f"已有索引包含 {len(indexed_ids)} 部电影")

        # 3. 找出新增电影
        all_movies = Movie.objects.exclude(id__in=indexed_ids)
        new_count = all_movies.count()
        if new_count == 0:
            self.stdout.write(self.style.SUCCESS("没有新增电影，索引无需更新。"))
            return

        self.stdout.write(f"发现 {new_count} 部新增电影，开始编码...")

        # 4. 构建新增电影的文档
        new_documents = []
        for movie in all_movies:
            genres_str = ", ".join([g.name for g in movie.genres.all()])
            actors_str = ", ".join([a.name for a in movie.actors.all()][:10])
            directors_str = "暂无"
            if hasattr(movie, 'directors'):
                d_list = [d.name for d in movie.directors.all()]
                if d_list:
                    directors_str = ", ".join(d_list)

            summary_text = movie.summary or ""
            score_text = f"{movie.score or 0.0}"

            combined_text = (
                f"标题: {movie.title}\n"
                f"评分: {score_text}\n"
                f"导演: {directors_str}\n"
                f"类型: {genres_str}\n"
                f"演员: {actors_str}\n"
                f"简介: {summary_text or '暂无简介'}"
            )

            metadata = {
                "movie_id": movie.id,
                "title": movie.title,
                "poster": movie.poster or "",
                "score": float(movie.score or 0.0)
            }
            new_documents.append(Document(page_content=combined_text, metadata=metadata))

        # 5. 分割文档
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = splitter.split_documents(new_documents)
        self.stdout.write(f"新增文档分割为 {len(split_docs)} 个片段")

        # 6. 创建新索引并合并
        start_time = time.time()
        new_db = FAISS.from_documents(split_docs, embedding_model)
        existing_db.merge_from(new_db)
        existing_db.save_local(index_path)

        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"--- 增量构建完成！新增 {new_count} 部电影，耗时 {elapsed:.2f} 秒 ---"
        ))