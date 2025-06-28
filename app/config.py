from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CLICKHOUSE_HOST: str
    CLICKHOUSE_PORT: int = 9000
    MODEL_BASE_DIR: str = "models"

    # .env에 있는 추가 필드 선언
    CLICKHOUSE_DB: str
    CLICKHOUSE_TABLE: str
    CLICKHOUSE_DAYS: int

    class Config:
        env_file = ".env"

settings = Settings()
