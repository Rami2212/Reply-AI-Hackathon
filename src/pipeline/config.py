from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class OpenRouterConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 50
    base_url: str = "https://openrouter.ai/api/v1"


@dataclass
class LangfuseConfig:
    public_key: str
    secret_key: str
    host: str = "https://challenges.reply.com/langfuse"


def load_project_env() -> Path:
    """Load .env from repository root (works regardless of current working directory)."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    return env_path


def build_openrouter_config() -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
        model=(os.getenv("OPENROUTER_MODEL") or "gpt-4o-mini").strip(),
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE") or "0.7"),
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS") or "50"),
    )


def build_langfuse_config() -> LangfuseConfig:
    return LangfuseConfig(
        public_key=(os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip(),
        secret_key=(os.getenv("LANGFUSE_SECRET_KEY") or "").strip(),
        host=(os.getenv("LANGFUSE_HOST") or "https://challenges.reply.com/langfuse").strip(),
    )

