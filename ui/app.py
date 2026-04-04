"""TTE Streamlit application."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.config import config
from core.models import EvalResult
from main import run_tte
from ui.components.metrics_panel import render_metrics_panel
from ui.components.task_input import render_task_input
from ui.components.trace_viewer import render_trace_viewer
from ui.components.tree_view import render_tree_view

st.set_page_config(
    page_title="Time-Travel Evals (TTE)",
    page_icon="TTE",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0;">
        <h1 style="font-size:2.5rem; margin-bottom:0.2rem;">Time-Travel Evals</h1>
        <p style="color:#94a3b8; font-size:1.1rem; margin-top:0;">
            Evaluate AI agents not only on what happened, but on what could have happened.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Configured budget: {config.MAX_REQUESTS_PER_MINUTE} rpm | "
    f"Request timeout: {getattr(config, 'LLM_REQUEST_TIMEOUT_SECONDS', 20.0):.0f}s | "
    f"LLM judge: {'on' if config.ENABLE_LLM_JUDGE else 'off'}"
)

with st.expander("Core Research: Why Multiverse Evals?"):
    st.markdown(
        """
        **1. The Trajectory Illusion:** Traditional benchmarks evaluate agents on a single happy path.
        This creates a false sense of security, because agents can memorize a path yet fail at the first anomaly.

        **2. Compounding Errors:** In autonomous systems, one unhandled error can snowball into an unrecoverable state.
        Resilience is only proven by perturbing the environment and measuring recovery.

        **3. State-Space Branching:** Time-Travel Evals forks the environment into multiple alternative realities,
        replacing binary pass/fail rates with richer robustness metrics.
        """
    )

st.markdown("---")

task, env_state, n_branches, max_steps, use_llm, run_clicked = render_task_input(
    config.MAX_REQUESTS_PER_MINUTE,
    config.ENABLE_LLM_JUDGE,
)

if run_clicked:
    if use_llm and not config.validate():
        st.error("`NVIDIA_API_KEY` is not set. Enable Demo Mode or set the key in `.env`.")
        st.stop()

    with st.status("Running Time-Travel Evaluation...", expanded=True) as status:
        status.write("Generating branches and executing the selected profile...")
        result: EvalResult = asyncio.run(
            run_tte(
                task=task,
                env_state=env_state,
                n_branches=n_branches,
                max_steps=max_steps,
                use_llm=use_llm,
                use_llm_agent=use_llm,
            )
        )
        status.update(label="Evaluation complete", state="complete", expanded=False)

    st.session_state["tte_result"] = result

if "tte_result" in st.session_state:
    result = st.session_state["tte_result"]

    render_metrics_panel(result)
    st.markdown("---")
    st.markdown("### Branch Timeline")
    render_tree_view(result)
    render_trace_viewer(result)

    st.markdown("---")
    col_export, col_info = st.columns([1, 3])
    with col_export:
        st.download_button(
            "Export Results (JSON)",
            data=result.model_dump_json(indent=2),
            file_name="tte_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            f"Evaluated {len(result.branches)} branches | "
            f"{result.total_api_calls} API calls | "
            f"{result.total_tokens:,} tokens"
        )
else:
    st.info("Configure a task in the sidebar and click Run Evaluation to begin.")
    st.markdown(
        """
        ### How it works

        1. Enter a task for the agent.
        2. Generate multiple alternate branches.
        3. Run the agent across those branches.
        4. Compare robustness, success rate, stability, and panic in one view.
        """
    )
