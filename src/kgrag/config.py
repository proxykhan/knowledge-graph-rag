from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    llm_provider: str = "gemini"           # "gemini" or "claude"

    gemini_api_key: str = ""
    # 3.7/3.8 Flash: 503 "high demand" on the free tier. 3.5 Flash: only 20 requests/day free.
    # 3.5 Flash-Lite with 4-chunk batches matched 3.5 Flash on company-company relationships.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: str = "MEDIUM"  # MINIMAL | LOW | MEDIUM | HIGH
    gemini_min_interval_s: float = 6.0     # start at 10 requests/min; slows down on 429

    anthropic_api_key: str = ""
    anthropic_workspace_id: str = ""
    claude_model: str = "claude-opus-5-5"

    sec_user_agent: str = ""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "kgrag-dev-password"
    postgres_dsn: str = "postgresql://kgrag:kgrag-dev-password@localhost:5432/kgrag"

    embedding_model: str = "BAAI/bge-small-en-v1.5"


settings = Settings()
