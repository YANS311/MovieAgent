import torch
from django.core.management.base import BaseCommand
# 🔥 修改 1: 改用 Auto 组件，兼容性更强，解决 NoneType 报错
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from myapp.models import Movie
from tqdm import tqdm


class Command(BaseCommand):
    help = '数据对齐：将 Florence-2 生成的英文 Caption 映射为中文语义'

    def handle(self, *args, **options):
        self.stdout.write("🔄 启动语义对齐引擎 (English -> Chinese)...")

        # 🔥 修改 2: 更精准的筛选逻辑
        # 1. has_mm_features=True (说明是经过 Florence-2 处理的)
        # 2. exclude(...): 排除掉包含任何中文字符的记录
        # 这样就能精准选中那 4340 条纯英文描述，而不会误伤包含 "IMAX" 字样的中文描述
        movies = Movie.objects.filter(has_mm_features=True) \
            .exclude(poster_caption__regex=r'[\u4e00-\u9fa5]')

        total = movies.count()
        if total == 0:
            self.stdout.write(self.style.SUCCESS("✨ 没有检测到纯英文描述，所有数据已是对齐状态。"))
            return

        self.stdout.write(f"📊 精准定位到 {total} 条纯英文描述待翻译...")

        # 2. 加载翻译模型
        model_name = "Helsinki-NLP/opus-mt-en-zh"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.stdout.write("📥 加载翻译模型 (AutoMode)...")
            # 🔥 关键修复: 使用 AutoTokenizer 自动处理 spm 文件路径
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self.stdout.write(self.style.SUCCESS("✅ 模型加载成功"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ 模型加载严重失败: {e}"))
            self.stdout.write("建议尝试手动删除缓存")
            return

        # 3. 批量推理配置
        BATCH_SIZE =8

        batch_movies = []
        batch_texts = []

        # 4. 开始循环
        for movie in tqdm(movies.iterator(), total=total, desc="⚡ 语义对齐中"):

            # 简单清洗：Florence-2 有时会生成 'The image shows...' 我们只需要核心内容
            text = movie.poster_caption or ""
            text = text.replace('<DETAILED_CAPTION>', '').strip()

            if not text:
                continue

            batch_movies.append(movie)
            batch_texts.append(text)

            # 积攒够一个 Batch 就处理
            if len(batch_texts) >= BATCH_SIZE:
                self.process_batch(model, tokenizer, batch_texts, batch_movies, device)
                batch_movies = []
                batch_texts = []

        # 处理尾巴
        if batch_texts:
            self.process_batch(model, tokenizer, batch_texts, batch_movies, device)

        self.stdout.write(self.style.SUCCESS(f"🎉 任务完成！{total} 条数据已完成中文化对齐。"))

    def process_batch(self, model, tokenizer, texts, movie_objs, device):
        try:
            # 1. Tokenize
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # 2. Generate
            with torch.no_grad():
                translated = model.generate(**inputs)

            # 3. Decode
            decoded_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

            # 4. Update DB
            for movie, zh_text in zip(movie_objs, decoded_texts):
                # 直接覆盖原字段，统一语义空间
                movie.poster_caption = zh_text
                movie.save(update_fields=['poster_caption'])

        except Exception as e:
            print(f"⚠️ Batch 翻译跳过: {e}")