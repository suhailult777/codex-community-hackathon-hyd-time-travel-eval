"""Centralised configuration for TTE, loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present and prefer the repo's current values for local runs.
load_dotenv(override=True)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(name: str | None) -> str:
    normalized = (name or "nvidia").strip().lower()
    if normalized in {"openai", "nvidia"}:
        return normalized
    return "nvidia"


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved runtime settings for one provider."""

    name: str
    label: str
    api_key: str
    api_key_env_var: str
    base_url: str
    agent_model: str
    generator_model: str


class TTEConfig:
    """Singleton-style configuration container."""

    def __init__(self):
        self.PROVIDER: str = _normalize_provider(os.getenv("TTE_PROVIDER", "nvidia"))
        legacy_api_base_url = os.getenv("TTE_API_BASE_URL")
        legacy_model_generator = os.getenv("TTE_MODEL_GENERATOR")
        legacy_model_agent = os.getenv("TTE_MODEL_AGENT")

        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", os.getenv("KIMI_API_KEY", ""))
        self.KIMI_API_KEY: str = self.NVIDIA_API_KEY

        self.NVIDIA_API_BASE_URL: str = os.getenv(
            "TTE_NVIDIA_API_BASE_URL",
            legacy_api_base_url if self.PROVIDER == "nvidia" and legacy_api_base_url else "https://integrate.api.nvidia.com/v1",
        )
        self.OPENAI_API_BASE_URL: str = os.getenv(
            "TTE_OPENAI_API_BASE_URL",
            legacy_api_base_url if self.PROVIDER == "openai" and legacy_api_base_url else "https://api.openai.com/v1",
        )
        self.NVIDIA_MODEL_GENERATOR: str = os.getenv(
            "TTE_NVIDIA_MODEL_GENERATOR",
            legacy_model_generator if self.PROVIDER == "nvidia" and legacy_model_generator else "moonshotai/kimi-k2.5",
        )
        self.NVIDIA_MODEL_AGENT: str = os.getenv(
            "TTE_NVIDIA_MODEL_AGENT",
            legacy_model_agent if self.PROVIDER == "nvidia" and legacy_model_agent else "moonshotai/kimi-k2.5",
        )
        self.OPENAI_MODEL_GENERATOR: str = os.getenv(
            "TTE_OPENAI_MODEL_GENERATOR",
            legacy_model_generator if self.PROVIDER == "openai" and legacy_model_generator else "gpt-5.4-nano",
        )
        self.OPENAI_MODEL_AGENT: str = os.getenv(
            "TTE_OPENAI_MODEL_AGENT",
            legacy_model_agent if self.PROVIDER == "openai" and legacy_model_agent else "gpt-5.4-nano",
        )

        if self.PROVIDER == "openai":
            self.API_KEY = self.OPENAI_API_KEY
            self.API_BASE_URL = self.OPENAI_API_BASE_URL
            self.MODEL_GENERATOR = self.OPENAI_MODEL_GENERATOR
            self.MODEL_AGENT = self.OPENAI_MODEL_AGENT
        else:
            self.API_KEY = self.NVIDIA_API_KEY
            self.API_BASE_URL = self.NVIDIA_API_BASE_URL
            self.MODEL_GENERATOR = self.NVIDIA_MODEL_GENERATOR
            self.MODEL_AGENT = self.NVIDIA_MODEL_AGENT
        self.EXECUTION_MODE: str = os.getenv("TTE_EXECUTION_MODE", "auto").strip().lower()
        self.PROVIDER_PROFILE: str = os.getenv("TTE_PROVIDER_PROFILE", "auto").strip().lower()
        self.MAX_BRANCHES: int = min(int(os.getenv("TTE_MAX_BRANCHES", "4")), 10)
        self.MAX_STEPS: int = min(int(os.getenv("TTE_MAX_STEPS_PER_BRANCH", "8")), 15)
        # API throttling defaults tuned for stricter free-tier/provider limits.
        self.MAX_REQUESTS_PER_MINUTE: int = max(1, int(os.getenv("TTE_MAX_REQUESTS_PER_MINUTE", "40")))
        self.RATE_LIMIT_SAFETY_FACTOR: float = min(
            1.0,
            max(0.1, float(os.getenv("TTE_RATE_LIMIT_SAFETY_FACTOR", "0.8"))),
        )
        self.LLM_MAX_RETRIES: int = max(0, int(os.getenv("TTE_LLM_MAX_RETRIES", "2")))
        self.LLM_BACKOFF_BASE_SECONDS: float = max(0.1, float(os.getenv("TTE_LLM_BACKOFF_BASE_SECONDS", "1.5")))
        self.LLM_BACKOFF_MAX_SECONDS: float = max(0.5, float(os.getenv("TTE_LLM_BACKOFF_MAX_SECONDS", "20.0")))
        self.LLM_REQUEST_TIMEOUT_SECONDS: float = max(
            5.0, float(os.getenv("TTE_LLM_REQUEST_TIMEOUT_SECONDS", "20.0"))
        )
        self.ENABLE_LLM_JUDGE: bool = _env_flag("TTE_ENABLE_LLM_JUDGE", True)
        self.AGENT_BATCH_HISTORY: int = max(1, int(os.getenv("TTE_AGENT_BATCH_HISTORY", "2")))
        self.LOG_DIR: Path = Path(os.getenv("TTE_LOG_DIR", "./logs"))
        self.CACHE_PATH: Path = Path(os.getenv("TTE_CACHE_PATH", self.LOG_DIR / "llm_cache.sqlite3"))
        self.EPSILON: float = 1e-8

        # Create log directory if missing
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def provider_label(provider: str) -> str:
        return "OpenAI" if _normalize_provider(provider) == "openai" else "NVIDIA"

    @staticmethod
    def provider_options() -> list[str]:
        return ["nvidia", "openai"]

    def get_provider_settings(self, provider: str | None = None) -> ProviderSettings:
        """Resolve runtime settings for a specific provider."""
        selected = _normalize_provider(provider or self.PROVIDER)
        if selected == self.PROVIDER:
            return ProviderSettings(
                name=selected,
                label=self.provider_label(selected),
                api_key=self.API_KEY,
                api_key_env_var="OPENAI_API_KEY" if selected == "openai" else "NVIDIA_API_KEY",
                base_url=self.API_BASE_URL,
                agent_model=self.MODEL_AGENT,
                generator_model=self.MODEL_GENERATOR,
            )
        if selected == "openai":
            return ProviderSettings(
                name="openai",
                label="OpenAI",
                api_key=self.OPENAI_API_KEY,
                api_key_env_var="OPENAI_API_KEY",
                base_url=self.OPENAI_API_BASE_URL,
                agent_model=self.OPENAI_MODEL_AGENT,
                generator_model=self.OPENAI_MODEL_GENERATOR,
            )
        return ProviderSettings(
            name="nvidia",
            label="NVIDIA",
            api_key=self.NVIDIA_API_KEY,
            api_key_env_var="NVIDIA_API_KEY",
            base_url=self.NVIDIA_API_BASE_URL,
            agent_model=self.NVIDIA_MODEL_AGENT,
            generator_model=self.NVIDIA_MODEL_GENERATOR,
        )

    def validate(self, provider: str | None = None) -> bool:
        """Return True if all critical config values are set."""
        return bool(self.get_provider_settings(provider).api_key)


# Module-level singleton
config = TTEConfig()
