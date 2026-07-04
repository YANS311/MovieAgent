import torch
from PIL import Image
from django.core.management.base import BaseCommand
from transformers import CLIPProcessor, CLIPModel
from myapp.models import Movie


class Command(BaseCommand):
    help = '使用 CLIP 模型审查海报内容安全'

    def handle(self, *args, **options):
        self.stdout.write("正在加载 CLIP 模型...")
        # 选用这个模型是因为它在 HuggingFace 上下载量大，且刚好够用
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        )
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        )

        # 定义负面标签 (你可以根据需要添加)
        # 注意：CLIP 是英文模型，提示词用英文效果最好
        labels = [
            "a normal movie poster suitable for public display,even if it has red colors",  # 0. 正常：强调适合公开展示
            "a photo containing bloody gore, violence and blood",  # 1. 血腥：强调暴力和血液
            "explicit nudity, sexual content, or adult content",  # 2. 暴露：强调成人内容
            "a scary horror image with ghost, skull or monster",  # 3. 恐怖：强调具体的恐怖元素
        ]

        movies = Movie.objects.filter(poster_file__isnull=False)

        count = 0
        for movie in movies:
            try:
                image = Image.open(movie.poster_file.path)

                # 预处理并推理
                inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)

                # 获取概率 (logits_per_image: [1, 4])
                probs = outputs.logits_per_image.softmax(dim=1)[0]

                # 逻辑：如果"正常"的概率小于 20%，或者某个负面标签概率 > 85%，则判定为敏感
                normal_prob = probs[0].item()
                bloody_prob = probs[1].item()
                nude_prob = probs[2].item()
                scary_prob = probs[3].item()

                if normal_prob < 0.2:
                    # 判定为敏感
                    flag_type = ""
                    if bloody_prob > 0.85:
                        flag_type = "血腥"
                    elif nude_prob > 0.9:
                        flag_type = "暴露"
                    elif scary_prob > 0.8:
                        flag_type = "恐怖"

                    if flag_type:
                        movie.is_sensitive = True
                        movie.sensitive_type = flag_type
                        movie.save()
                        self.stdout.write(self.style.WARNING(f"⚠️ [{flag_type}] {movie.title}"))
                        count += 1
                    if not flag_type:
                        movie.is_sensitive = False
                        movie.sensitive_type = ""
                        movie.save() #安全

            except Exception as e:
                print(f"Error {movie.title}: {e}")

        self.stdout.write(f"审查完成！共发现 {count} 张敏感海报。")