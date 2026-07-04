import os
import django

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')  # 确认你的 settings 路径
django.setup()

from django.db.models import Q
from myapp.models import Movie
import gc


def windows_safe_reset():
    # 1. 查找坏数据
    bad_movies = Movie.objects.filter(
        Q(poster_caption__isnull=True) |
        Q(poster_caption="") |
        Q(poster_caption__icontains="Invalid") |
        Q(poster_caption__icontains="corrupt") |
        Q(poster_caption__icontains="error")
    )

    total = bad_movies.count()
    print(f"🔍 在 Windows 环境下找到 {total} 条待重置记录...")

    success_count = 0
    for movie in bad_movies:
        try:
            # 2. 物理删除文件（带错误处理，应对 Windows 文件锁定）
            if movie.poster_file:
                try:
                    file_path = movie.poster_file.path
                    if os.path.exists(file_path):
                        # 尝试强制关闭可能的文件句柄（Windows 常见问题）
                        gc.collect()
                        os.remove(file_path)
                except PermissionError:
                    print(f"⚠️ 文件被占用，无法删除: {movie.title}")
                except Exception as e:
                    print(f"⚠️ 删除文件出错: {e}")

            # 3. 字段重置
            movie.poster_file = None
            movie.has_mm_features = False
            movie.poster_caption = ""
            movie.save()
            success_count += 1

            if success_count % 100 == 0:
                print(f"已清理 {success_count} 条...")

        except Exception as e:
            print(f"❌ 重置 {movie.title} 失败: {e}")

    print(f"✅ 成功重置 {success_count} 条。Windows 兼容性清理完成！")


if __name__ == "__main__":
    windows_safe_reset()