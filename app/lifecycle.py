import logging
from .config import settings

logger = logging.getLogger("uvicorn")

async def on_startup():
    logger.info("🚀 Application startup")

async def on_shutdown():
    logger.info("🛑 Application shutdown")
