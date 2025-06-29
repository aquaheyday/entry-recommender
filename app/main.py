import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from app.config import settings
from app.lifecycle import on_startup, on_shutdown
from app.routers import train, recommend

# 로깅 설정 (생략 가능)
logging.basicConfig(level=logging.INFO)
file_handler = RotatingFileHandler("/app/logs/app.log", maxBytes=10*1024*1024, backupCount=3)
logging.getLogger().addHandler(file_handler)
app = FastAPI(
    debug=True,
    title="Recommendation Service",
    version="1.0.0",
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 모델별 라우터 포함
app.include_router(train.router)
app.include_router(recommend.router)