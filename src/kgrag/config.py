from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    llm_provider: str = "gemini"           # answers, routing, group labels: "gemini", "ollama" or "claude"
    extraction_provider: str = "gemini"    # builds the graph; its cache is keyed by this model

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_think: str = "false"            # "false" | "true" (answers) | "all" (answers and routing); slower
    ollama_num_ctx: int = 8192             # context window; fits an 8 GB GPU alongside the model

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

    # Public deployments: questions per client IP (0 = unlimited, the local default)
    ask_limit_per_minute: int = 0
    ask_limit_per_day: int = 0


settings = Settings()
