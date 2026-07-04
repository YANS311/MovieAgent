# 文件: myapp/management/commands/extract_mm_features_florence.py
import os
import torch
from django.core.management.base import BaseCommand
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoModelForCausalLM
from myapp.models import Movie
from tqdm import tqdm

# 🔥 核心补丁 1：允许加载下载不完全（截断）的图片
#ImageFile.LOAD_TRUNCATED_IMAGES = True


class Command(BaseCommand):
    help = '最终修复版：Eager模式 + 手动Mask'

    def handle(self, *args, **options):
        self.stdout.write("🚀 启动 Florence-2 最终修复版...")

        movies = Movie.objects.filter(poster_file__isnull=False, has_mm_features=False)
        total = movies.count()
        if total == 0:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = 'microsoft/Florence-2-large'

        # ---------------------------------------------------------
        # 🔧 修复点 1：加载阶段 (必须强制使用 eager 模式)
        # ---------------------------------------------------------
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"  # 👈 必须加回来，否则报 _supports_sdpa 错
            ).to(device)

            # 为了双重保险，如果还没有这个属性，手动补一个
            if not hasattr(model, '_supports_sdpa'):
                model._supports_sdpa = False

            model.eval()
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.stdout.write(self.style.SUCCESS("✅ 模型加载成功 (Eager Mode)。"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ 加载失败: {e}"))
            return

        success_count = 0
        skip_count = 0

        for movie in tqdm(movies, total=total, desc="⚡ 正在处理"):
            try:
                # 0. 基础检查
                if not movie.poster_file or not os.path.exists(movie.poster_file.path):
                    movie.has_mm_features = True
                    movie.save()
                    continue

                # 1. 图片加载
                try:
                    raw_img = Image.open(movie.poster_file.path)
                    image = raw_img.convert("RGB")
                    # 适当缩小尺寸，防止显存溢出导致的异常
                    image.thumbnail((1024, 1024))
                    # 确保图片数据完全加载
                    if image.size[0] == 0 or image.size[1] == 0: raise Exception("空图片")
                    raw_img.close()
                except Exception:
                    continue

                # 2. Processor 处理
                prompt = '<DETAILED_CAPTION>'
                inputs = processor(text=[prompt], images=[image], return_tensors="pt")

                # ---------------------------------------------------------
                # 🔧 修复点 2：推理准备 (手动补全 attention_mask)
                # ---------------------------------------------------------
                # 解决 'NoneType' shape 报错的核心：
                # 如果 Eager 模式下没有 mask，模型计算 Attention 时就会崩。
                if "attention_mask" not in inputs or inputs["attention_mask"] is None:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

                # 数据搬运
                model_inputs = {}
                for k, v in inputs.items():
                    if v is None: continue
                    if k == "pixel_values":
                        model_inputs[k] = v.to(device, torch.float16)
                    elif isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(device)

                # 3. 推理
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        num_beams=1,
                    )

                output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # 4. 保存
                movie.poster_caption = output
                movie.has_mm_features = True
                movie.save()
                success_count += 1

                if success_count % 50 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                # 打印更详细的报错信息
                self.stdout.write(self.style.ERROR(f"🚨 {movie.title} 失败: {str(e)}"))
                skip_count += 1