#!/bin/bash
set -e

# 等待 MySQL 就绪
echo "等待 MySQL 就绪..."
for i in $(seq 1 30); do
    if python -c "import socket; s=socket.socket(); s.connect(('${DB_HOST:-localhost}', int('${DB_PORT:-3306}'))); s.close()" 2>/dev/null; then
        echo "MySQL 已就绪"
        break
    fi
    echo "MySQL 未就绪，等待中... ($i/30)"
    sleep 2
done

# 执行数据库迁移
echo "执行数据库迁移..."
python manage.py migrate --noinput || echo "迁移失败（可能数据库已存在），继续启动..."

# 收集静态文件
echo "收集静态文件..."
python manage.py collectstatic --noinput 2>/dev/null || true

# 启动 Uvicorn ASGI 服务器（异步多 Worker）
echo "启动 Uvicorn ASGI 服务器..."
exec uvicorn movie.asgi:application --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-8}
