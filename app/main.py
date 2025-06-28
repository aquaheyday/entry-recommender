from fastapi import FastAPI
from app.config import settings
from app.lifecycle import on_startup, on_shutdown
from app.routers import train, recommend

app = FastAPI(
    title="Recommendation Service",
    version="1.0.0",
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(train.router, prefix="/train", tags=["Training"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])