import os

import faiss
import numpy as np
import pickle
from django.core.management.base import BaseCommand
from transformers import CLIPProcessor, CLIPModel
from myapp.models import Movie
from PIL import Image
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"




class Command(BaseCommand):
    help = '构建视觉风格索引 (FAISS)'

    def handle(self, *args, **options):
        local_model_path = "./local_models/clip-vit-base-patch32"
        # 1. 加载模型
        model = CLIPModel.from_pretrained(local_model_path)
        processor = CLIPProcessor.from_pretrained(local_model_path)

        # 2. 准备容器
        # CLIP base 的向量维度是 512
        index = faiss.IndexFlatIP(512)
        movie_ids = []  # 用于记录 index 对应的 movie_id

        movies = Movie.objects.filter(poster_file__isnull=False)

        self.stdout.write("开始提取视觉特征...")
        for movie in movies:
            try:
                image = Image.open(movie.poster_file.path)
                inputs = processor(images=image, return_tensors="pt", padding=True)

                # 提取图像特征 (Image Embeddings)
                image_features = model.get_image_features(**inputs)

                # 归一化 (这是计算余弦相似度的关键)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                # 转为 numpy
                vector = image_features.detach().numpy()

                # 存入 FAISS
                index.add(vector)
                movie_ids.append(movie.id)

            except Exception:
                pass

        # 3. 保存索引和ID映射
        faiss.write_index(index, "faiss_visual_index.bin")
        with open("visual_ids.pkl", "wb") as f:
            pickle.dump(movie_ids, f)

        self.stdout.write("视觉索引构建完成！")