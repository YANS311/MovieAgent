import os
import sys
import time
from django.apps import AppConfig

class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'

    def ready(self):
        """
        [系统启动预热脚本]
        目标：消除所有首屏加载延迟 (Cold Start Latency)
        """
        # 防止 runserver 的 Reload 机制导致重复运行
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            return

        print("\n" + "="*50)
        print("🚀 [系统启动] 开始全链路预热 (System Warm-up)...")
        t_start = time.time()

        try:
            # 必须在 ready 内部引入，避免 AppRegistryNotReady 错误
            from . import views
            import jieba.posseg as pseg
            from django.db import connection
            from django_redis import get_redis_connection
            views.warmup_all_systems()



            # ------------------------------------------------
            # 5. (可选) 预热 RAG Embedding 模型
            # ------------------------------------------------
            # 如果您的 RAG 是懒加载的，建议也在 views 里搞个 load_rag_resources()
            #print("   🤖 [5/5] 正在加载 RAG 向量模型...")
            #views.load_rag_resources()

        except Exception as e:
            print(f"   ❌ [警告] 预热过程中出现错误: {e}")
            print("   (这不会影响服务器启动，但首屏可能会变慢)")
            # 打印详细错误堆栈，方便调试
            import traceback
            traceback.print_exc()

        t_end = time.time()
        print(f"✅ [系统启动] 预热完成！耗时: {t_end - t_start:.2f}s")
        print("="*50 + "\n")