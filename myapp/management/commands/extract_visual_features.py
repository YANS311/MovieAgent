import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from django.core.management.base import BaseCommand
from myapp.models import Movie  # 此时可以正常导入


class Command(BaseCommand):
    help = '利用 CLIP 模型提取电影海报的视觉特征 (Visual KG)'

    def handle(self, *args, **options):
        self.stdout.write("--- 🚀 启动视觉特征提取任务 ---")

        # 1. 检查计算设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stdout.write(f"正在使用设备: {device}")

        # 2. 加载预训练 CLIP 模型 (openai/clip-vit-base-patch32)
        # 建议提前下载到本地，或使用镜像站：os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model_name = "openai/clip-vit-base-patch32"
        try:
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            self.stderr.write(f"模型加载失败: {e}")
            return

        # 3. 筛选需要处理的电影
        # 条件：有本地海报文件 + 视觉向量字段为空
        movies = Movie.objects.filter(
            poster_file__isnull=False,
            poster_embedding_json__isnull=True
        ).exclude(poster_file='')

        total = movies.count()
        if total == 0:
            self.stdout.write(self.style.SUCCESS("✨ 所有海报已处理完毕，无需提取。"))
            return

        self.stdout.write(f"发现 {total} 部电影等待处理...")

        # 4. 循环提取特征
        for movie in tqdm(movies, desc="提取进度"):
            try:
                # 获取图片的绝对磁盘路径
                img_path = movie.poster_file.path
                if not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")

                # 预处理与模型推理
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    vision_outputs = model.get_image_features(**inputs)

                # 转换向量为 List 以便存储为 JSON
                # CLIP 通常输出 512 维向量
                embedding = vision_outputs[0].cpu().numpy().tolist()

                # 保存到数据库
                movie.poster_embedding_json = embedding
                movie.has_mm_features = True
                movie.save()

            except Exception as e:
                self.stderr.write(f"❌ 电影 ID {movie.id} ({movie.title}) 处理失败: {e}")

        self.stdout.write(self.style.SUCCESS(f"✅ 处理完成！共提取 {total} 条视觉特征。"))