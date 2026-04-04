from core.llm_cache import PersistentLLMCache, build_cache_key


def test_build_cache_key_changes_when_model_or_execution_mode_changes():
    payload = {"task": "Deploy app", "step": 1}
    key_a = build_cache_key(
        base_url="https://example.com/v1",
        model="model-a",
        provider_profile="batched_json",
        execution_mode="turbo",
        prompt_version="v1",
        purpose="agent.step",
        payload=payload,
    )
    key_b = build_cache_key(
        base_url="https://example.com/v1",
        model="model-b",
        provider_profile="batched_json",
        execution_mode="turbo",
        prompt_version="v1",
        purpose="agent.step",
        payload=payload,
    )
    key_c = build_cache_key(
        base_url="https://example.com/v1",
        model="model-a",
        provider_profile="batched_json",
        execution_mode="standard",
        prompt_version="v1",
        purpose="agent.step",
        payload=payload,
    )
    assert key_a != key_b
    assert key_a != key_c


def test_persistent_cache_round_trips(tmp_path):
    cache = PersistentLLMCache(tmp_path / "llm_cache.sqlite3")
    cache.set("sample-key", {"value": 123, "message": "ok"})
    assert cache.get("sample-key") == {"value": 123, "message": "ok"}
    assert cache.get("missing-key") is None
