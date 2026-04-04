"""Centralised configuration for TTE, loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present and prefer the repo's current values for local runs.
load_dotenv(override=True)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TTEConfig:
    """Singleton-style configuration container."""

    def __init__(self):
        self.API_KEY: str = os.getenv("NVIDIA_API_KEY", os.getenv("KIMI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
        # Backward-compat aliases for older call sites.
        self.OPENAI_API_KEY: str = self.API_KEY
        self.NVIDIA_API_KEY: str = self.API_KEY
        self.KIMI_API_KEY: str = self.API_KEY
        self.API_BASE_URL: str = os.getenv("TTE_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.MODEL_GENERATOR: str = os.getenv("TTE_MODEL_GENERATOR", "moonshotai/kimi-k2.5")
        self.MODEL_AGENT: str = os.getenv("TTE_MODEL_AGENT", "moonshotai/kimi-k2.5")
        self.MAX_BRANCHES: int = min(int(os.getenv("TTE_MAX_BRANCHES", "4")), 10)
        self.MAX_STEPS: int = min(int(os.getenv("TTE_MAX_STEPS_PER_BRANCH", "8")), 15)
        # API throttling defaults tuned for stricter free-tier/provider limits.
        self.MAX_REQUESTS_PER_MINUTE: int = max(1, int(os.getenv("TTE_MAX_REQUESTS_PER_MINUTE", "40")))
        self.LLM_MAX_RETRIES: int = max(0, int(os.getenv("TTE_LLM_MAX_RETRIES", "2")))
        self.LLM_BACKOFF_BASE_SECONDS: float = max(0.1, float(os.getenv("TTE_LLM_BACKOFF_BASE_SECONDS", "1.5")))
        self.LLM_BACKOFF_MAX_SECONDS: float = max(0.5, float(os.getenv("TTE_LLM_BACKOFF_MAX_SECONDS", "20.0")))
        self.LLM_REQUEST_TIMEOUT_SECONDS: float = max(
            5.0, float(os.getenv("TTE_LLM_REQUEST_TIMEOUT_SECONDS", "20.0"))
        )
        self.ENABLE_LLM_JUDGE: bool = _env_flag("TTE_ENABLE_LLM_JUDGE", False)
        self.LOG_DIR: Path = Path(os.getenv("TTE_LOG_DIR", "./logs"))
        self.EPSILON: float = 1e-8

        # Create log directory if missing
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Return True if all critical config values are set."""
        if not self.API_KEY:
            return False
        return True


# Module-level singleton
config = TTEConfig()
