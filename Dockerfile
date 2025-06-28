FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libpq-dev \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python이 /app 하위의 app/ 패키지를 찾도록 설정
ENV PYTHONPATH=/app

# 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

# 애플리케이션 코드 복사
COPY app/    app/
COPY core/   core/
COPY scripts/ scripts/
RUN mkdir models

# 환경변수 파일 복사(.env)
COPY .env .

VOLUME ["/app/models"]
EXPOSE 8000

# Gunicorn + Uvicorn 워커로 프로덕션 서버 구동
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "app.main:app"]
