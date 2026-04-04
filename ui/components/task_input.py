"""Task input component with budget-aware defaults and runtime estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from core.config import config
from core.runtime_budget import estimate_runtime, format_duration, recommend_live_profile

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_data(show_spinner=False)
def load_presets() -> List[Dict[str, Any]]:
    path = DATA_DIR / "test_cases.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def render_task_input(
    *,
    rpm_budget: int,
    default_provider: str = "nvidia",
    llm_judge_enabled: bool = False,
    execution_mode: str = "auto",
    provider_profile: str = "auto",
    rate_limit_safety_factor: float = 0.8,
) -> Tuple[str, Dict[str, Any] | None, int, int, bool, bool, str]:
    """Render the sidebar controls and return the current run configuration."""
    presets = load_presets()
    preset_names = ["Custom"] + [preset["task"][:60] for preset in presets]
    default_branches, default_steps = recommend_live_profile(rpm_budget)

    st.sidebar.header("Configuration")
    st.sidebar.caption(f"Configured provider budget: up to {rpm_budget} requests per minute")
    st.sidebar.caption(
        f"Recommended live profile: {default_branches} branches x {default_steps} steps for a smoother run"
    )

    provider_options = config.provider_options()
    default_provider_name = default_provider if default_provider in provider_options else provider_options[0]
    selected_provider = st.sidebar.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(default_provider_name),
        format_func=config.provider_label,
    )
    provider_settings = config.get_provider_settings(selected_provider)
    st.sidebar.caption(
        f"{provider_settings.label}: {provider_settings.agent_model} via {provider_settings.base_url}"
    )

    demo_mode = st.sidebar.checkbox("Demo mode (no API calls)", value=False)
    choice = st.sidebar.selectbox("Preset task", preset_names)

    if choice == "Custom":
        task = st.sidebar.text_area(
            "Base task",
            value="Deploy version 2 of the frontend application to production",
            height=80,
        )
        env_state = None
    else:
        idx = preset_names.index(choice) - 1
        preset = presets[idx]
        task = preset["task"]
        env_state = preset.get("default_state")
        st.sidebar.info(f"Using preset: {preset['id']}")

    n_branches = st.sidebar.slider("Number of branches", 2, 7, default_branches)
    max_steps = st.sidebar.slider("Max steps per branch", 3, 12, default_steps)

    use_llm = not demo_mode
    runtime = estimate_runtime(
        n_branches=n_branches,
        max_steps=max_steps,
        use_llm=use_llm,
        use_llm_agent=use_llm,
        use_llm_judge=use_llm and llm_judge_enabled,
        rpm=rpm_budget,
        execution_mode=execution_mode,
        provider_profile=provider_profile,
        base_url=provider_settings.base_url,
        agent_model=provider_settings.agent_model,
        generator_model=provider_settings.generator_model,
        rate_limit_safety_factor=rate_limit_safety_factor,
    )

    if use_llm:
        st.sidebar.caption(f"Execution mode: {runtime.execution_mode} | profile: {runtime.provider_profile}")
        st.sidebar.caption(
            f"Estimated logical calls: {runtime.logical_calls} | scheduled provider calls: {runtime.scheduled_provider_calls}"
        )
        st.sidebar.caption(
            f"Effective RPM: {runtime.effective_rpm} | minimum time: about {format_duration(runtime.estimated_min_seconds)}"
        )
        if llm_judge_enabled:
            st.sidebar.caption("LLM judge is enabled and included in the provider call estimate.")
        if runtime.scheduled_provider_calls > runtime.effective_rpm:
            st.sidebar.warning("This setup is likely to take more than a minute at the current effective RPM.")
    else:
        st.sidebar.success("Demo mode uses zero provider calls.")

    run_clicked = st.sidebar.button("Run Evaluation", type="primary", use_container_width=True)
    return task, env_state, n_branches, max_steps, use_llm, run_clicked, selected_provider
