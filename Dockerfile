FROM python:3.10-slim

# 필수 빌드 도구 설치
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

# 작업 디렉토리 설정
WORKDIR /app

# pip 최신화 및 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 노출 및 FastAPI 실행
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
