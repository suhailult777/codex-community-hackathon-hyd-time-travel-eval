"""Provider capability resolution for provider-agnostic execution planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    """Normalized capability profile for a model on an OpenAI-compatible provider."""

    provider_profile: str
    provider_label: str
    base_url: str
    model: str
    supports_json_object: bool
    supports_batched_agent_prompt: bool
    supports_batched_judge_prompt: bool
    preferred_effective_rpm_factor: float


_BATCH_HINTS = (
    "gpt",
    "kimi",
    "moonshot",
    "qwen",
    "llama",
    "mixtral",
    "deepseek",
    "claude",
    "gemini",
)


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _provider_label(base_url: str) -> str:
    lowered = _normalize_text(base_url)
    if "api.openai.com" in lowered or "openai" in lowered:
        return "OpenAI"
    if "nvidia" in lowered:
        return "NVIDIA"
    if "moonshot" in lowered:
        return "Moonshot"
    return "Generic-compatible"


def _profile_from_auto(base_url: str, model: str) -> str:
    haystack = f"{_normalize_text(base_url)} {_normalize_text(model)}"
    if any(token in haystack for token in _BATCH_HINTS):
        return "batched_json"
    if haystack:
        return "strict_json"
    return "generic"


def resolve_provider_capabilities(
    *,
    base_url: str,
    model: str,
    provider_profile: str = "auto",
) -> ProviderCapabilities:
    """Resolve a config-driven capability profile for a provider/model pair."""
    requested_profile = _normalize_text(provider_profile) or "auto"
    resolved_profile = _profile_from_auto(base_url, model) if requested_profile == "auto" else requested_profile

    if resolved_profile == "batched_json":
        return ProviderCapabilities(
            provider_profile="batched_json",
            provider_label=_provider_label(base_url),
            base_url=base_url,
            model=model,
            supports_json_object=True,
            supports_batched_agent_prompt=True,
            supports_batched_judge_prompt=True,
            preferred_effective_rpm_factor=0.80,
        )

    if resolved_profile == "strict_json":
        return ProviderCapabilities(
            provider_profile="strict_json",
            provider_label=_provider_label(base_url),
            base_url=base_url,
            model=model,
            supports_json_object=True,
            supports_batched_agent_prompt=False,
            supports_batched_judge_prompt=False,
            preferred_effective_rpm_factor=0.75,
        )

    return ProviderCapabilities(
        provider_profile="generic",
        provider_label=_provider_label(base_url),
        base_url=base_url,
        model=model,
        supports_json_object=True,
        supports_batched_agent_prompt=False,
        supports_batched_judge_prompt=False,
        preferred_effective_rpm_factor=0.65,
    )
