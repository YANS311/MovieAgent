"""
企业级日志系统配置
================================================
替代 views.py 中散落的 print() 语句，
提供分级日志、日志轮转和结构化输出能力。

升级动机（方案三）：
  - 原实现：print() → 生产环境丢失，无法分级过滤
  - 新实现：logging 模块 → 文件持久化 + 级别过滤 + 日志轮转

使用方式：
    from myapp.utils.logger_config import get_logger
    logger = get_logger('movie_agent')
    logger.info("推荐完成", extra={'user_id': 123, 'intent': 'QUERY_MOVIE'})
================================================
"""

import logging
import logging.handlers
import os
from datetime import datetime


# 日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志格式
DETAILED_FORMAT = (
    '%(asctime)s | %(levelname)-8s | %(name)s | '
    '%(funcName)s:%(lineno)d | %(message)s'
)
SIMPLE_FORMAT = '%(asctime)s | %(levelname)-8s | %(message)s'
JSON_FORMAT = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","func":"%(funcName)s","line":%(lineno)d,"msg":"%(message)s"}'


def get_logger(name='movie_agent', level=logging.DEBUG):
    """
    获取一个配置好的 logger 实例。
    
    特性：
    1. Console Handler: INFO 级别，彩色输出
    2. File Handler: DEBUG 级别，按大小轮转（10MB × 5 个备份）
    3. Error File Handler: ERROR 级别，独立错误日志
    
    Args:
        name: logger 名称
        level: 最低日志级别
    
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    
    # 防止重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False  # 防止日志传播到 root logger
    
    # ── Handler 1: Console ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter(SIMPLE_FORMAT))
    logger.addHandler(console_handler)
    
    # ── Handler 2: 全量日志文件（DEBUG 级别，轮转）──
    log_file = os.path.join(LOG_DIR, f'{name}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
    logger.addHandler(file_handler)
    
    # ── Handler 3: 错误日志文件（ERROR 级别，独立）──
    error_file = os.path.join(LOG_DIR, f'{name}_error.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
    logger.addHandler(error_handler)
    
    return logger


class ColorFormatter(logging.Formatter):
    """
    彩色日志格式化器（终端输出用）
    """
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# 预配置的 logger 实例（懒加载）
_app_logger = None


def app_logger():
    """
    获取全局应用 logger（懒加载单例）。
    适用于 views.py 中的通用日志。
    """
    global _app_logger
    if _app_logger is None:
        _app_logger = get_logger('movie_agent')
    return _app_logger


# print_to_log 迁移辅助函数
# 用于快速将 print() 替换为 logger 调用
def log_migration_helper():
    """
    迁移指南：
    
    原代码                                    → 新代码
    ─────────────────────────────────────────────────────────
    print(f"✅ [CLIP] ...")                    → logger.info("[CLIP] ...")
    print(f"⚠️ [Warning] ...")                → logger.warning("[Warning] ...")
    print(f"❌ [Error] ...")                   → logger.error("[Error] ...")
    print(f"[DEBUG] ...")                      → logger.debug("...")
    except Exception as e:                     → except Exception as e:
        print(f"Error: {e}")                       logger.exception("...", exc_info=True)
        traceback.print_exc()                      # 不再需要手动打印 traceback
    """
    pass