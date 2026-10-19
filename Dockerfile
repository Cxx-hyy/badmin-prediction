FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/models/model.pkl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000 -r requirements.txt

COPY app ./app
COPY ml/registry ./ml/registry

# 模型文件不进入镜像：运行时通过 -v 挂载，或由部署平台注入
ARG APP_ENV=production
ENV APP_ENV=${APP_ENV}

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]