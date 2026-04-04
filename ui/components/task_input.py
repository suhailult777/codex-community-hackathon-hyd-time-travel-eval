"""Task input component with budget-aware defaults and runtime estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _estimate_total_api_calls(
    *,
    n_branches: int,
    max_steps: int,
    use_llm: bool,
    use_llm_agent: bool,
) -> int:
    calls = 0
    if use_llm:
        calls += 1
    if use_llm_agent:
        calls += n_branches * max_steps
    return calls


def _estimate_min_runtime_seconds(total_api_calls: int, rpm: int) -> int:
    if total_api_calls <= 0:
        return 0
    return max(1, int(round((total_api_calls * 60.0) / max(1, rpm))))


def _recommend_live_profile(rpm: int) -> tuple[int, int]:
    if rpm >= 40:
        return 3, 6
    if rpm >= 25:
        return 3, 5
    if rpm >= 15:
        return 2, 5
    return 2, 4


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem = divmod(seconds, 60)
    if rem == 0:
        return f"{minutes}m"
    return f"{minutes}m {rem}s"


@st.cache_data(show_spinner=False)
def load_presets() -> List[Dict[str, Any]]:
    path = DATA_DIR / "test_cases.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def render_task_input(rpm_budget: int) -> Tuple[str, Dict[str, Any] | None, int, int, bool, bool]:
    """Render the sidebar controls and return the current run configuration."""
    presets = load_presets()
    preset_names = ["Custom"] + [preset["task"][:60] for preset in presets]
    default_branches, default_steps = _recommend_live_profile(rpm_budget)

    st.sidebar.header("Configuration")
    st.sidebar.caption(f"Configured provider budget: up to {rpm_budget} requests per minute")
    st.sidebar.caption(
        f"Recommended live profile: {default_branches} branches x {default_steps} steps for a smoother run"
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

    n_branches = st.sidebar.slider("Number of branches", 2, 6, default_branches)
    max_steps = st.sidebar.slider("Max steps per branch", 3, 12, default_steps)

    use_llm = not demo_mode
    estimated_calls = _estimate_total_api_calls(
        n_branches=n_branches,
        max_steps=max_steps,
        use_llm=use_llm,
        use_llm_agent=use_llm,
    )
    estimated_seconds = _estimate_min_runtime_seconds(estimated_calls, rpm_budget)

    if use_llm:
        st.sidebar.caption(f"Estimated provider calls: {estimated_calls}")
        st.sidebar.caption(
            f"Minimum time at {rpm_budget} rpm: about {_format_duration(estimated_seconds)}"
        )
        if estimated_calls > rpm_budget:
            st.sidebar.warning("This setup is likely to take more than a minute at the current RPM budget.")
    else:
        st.sidebar.success("Demo mode uses zero provider calls.")

    run_clicked = st.sidebar.button("Run Evaluation", type="primary", use_container_width=True)
    return task, env_state, n_branches, max_steps, use_llm, run_clicked
