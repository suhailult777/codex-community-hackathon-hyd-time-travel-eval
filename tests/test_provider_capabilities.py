from core.provider_capabilities import resolve_provider_capabilities


def test_auto_profile_resolves_to_batched_json_for_known_models():
    capabilities = resolve_provider_capabilities(
        base_url="https://integrate.api.nvidia.com/v1",
        model="moonshotai/kimi-k2.5",
        provider_profile="auto",
    )
    assert capabilities.provider_profile == "batched_json"
    assert capabilities.supports_batched_agent_prompt is True
    assert capabilities.supports_batched_judge_prompt is True
    assert capabilities.provider_label == "NVIDIA"


def test_openai_models_resolve_to_openai_batched_profile():
    capabilities = resolve_provider_capabilities(
        base_url="https://api.openai.com/v1",
        model="gpt-5.4-nano",
        provider_profile="auto",
    )
    assert capabilities.provider_profile == "batched_json"
    assert capabilities.provider_label == "OpenAI"
    assert capabilities.supports_batched_agent_prompt is True
    assert capabilities.supports_batched_judge_prompt is True


def test_auto_profile_falls_back_to_strict_or_generic_for_unknown_models():
    capabilities = resolve_provider_capabilities(
        base_url="https://example.invalid/v1",
        model="custom-model",
        provider_profile="auto",
    )
    assert capabilities.provider_profile in {"strict_json", "generic"}
    assert capabilities.supports_json_object is True


def test_explicit_generic_profile_disables_batching():
    capabilities = resolve_provider_capabilities(
        base_url="https://integrate.api.nvidia.com/v1",
        model="moonshotai/kimi-k2.5",
        provider_profile="generic",
    )
    assert capabilities.provider_profile == "generic"
    assert capabilities.supports_batched_agent_prompt is False
    assert capabilities.supports_batched_judge_prompt is False
