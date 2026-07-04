import os
import requests
import time
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Q

from myapp.models import Movie  # 确保导入正确的 Movie 模型


class Command(BaseCommand):
    help = '批量下载电影海报到本地存储'

    def handle(self, *args, **options):
        self.stdout.write("🚀 开始下载海报任务...")

        # 筛选条件：有 URL 但还没有本地文件的电影
        # 筛选条件：
        # 1. 有 URL (poster 不是 NULL 且 不是空字符串)
        # 2. 没有本地文件 (poster_file 是 NULL 或者 是空字符串)
        movies = Movie.objects.filter(
            poster__isnull=False
        ).exclude(
            poster=''
        ).filter(
            Q(poster_file='') | Q(poster_file__isnull=True)
        )

        total = movies.count()
        self.stdout.write(f"📊 共发现 {total} 部电影需要下载海报")

        success_count = 0
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            # 如果是豆瓣图片，加上 Referer
            # 'Referer': 'https://movie.douban.com/'
        }

        for i, movie in enumerate(movies):
            url = movie.poster
            if not url.startswith('https'):
                continue

            try:
                self.stdout.write(f"[{i + 1}/{total}] 下载: {movie.title} ... ", ending='')

                response = requests.get(url, headers=headers, stream=True, timeout=15)
                if response.status_code == 200:
                    content = response.content

                    # --- 核心校验：确保它是真正的图片且不完整 ---
                    if len(content) < 5000:  # 小于 5KB 的基本都是假图
                        self.stdout.write(self.style.WARNING('× (文件太小)'))
                        continue

                    try:
                        from PIL import Image
                        from io import BytesIO
                        img = Image.open(BytesIO(content))
                        img.verify()  # 校验图片完整性
                    except Exception:
                        self.stdout.write(self.style.WARNING('× (图片损坏)'))
                        continue

                    file_name = url.split('/')[-1].split('?')[0]
                    movie.poster_file.save(file_name, ContentFile(content), save=True)
                    self.stdout.write(self.style.SUCCESS('√'))
                else:
                    self.stdout.write(self.style.WARNING(f'× (HTTP {response.status_code})'))

                # 礼貌性延时，防止被封 IP (如果是大量下载建议开启)
                time.sleep(0.5)

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error: {e}'))

        self.stdout.write(self.style.SUCCESS(f"\n🎉 任务完成！成功下载 {success_count}/{total} 张海报"))