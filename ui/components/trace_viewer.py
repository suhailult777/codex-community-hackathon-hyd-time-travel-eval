"""Trace viewer that renders one branch at a time for faster frontend performance."""

from __future__ import annotations

import streamlit as st

from core.models import BranchTrace, EvalResult


def _branch_label(trace: BranchTrace) -> str:
    status = "PASS" if trace.success else "FAIL"
    baseline = " | baseline" if trace.branch.is_baseline else ""
    label = trace.branch.label or trace.branch.id
    return f"{status} | {label} | score {trace.score:.2f} | panic {trace.panic_score:.2f}{baseline}"


def render_trace_viewer(result: EvalResult) -> None:
    """Render a compact summary and a single selected branch trace."""
    st.markdown("---")
    st.markdown("### Detailed Branch Trace")

    if not result.branches:
        st.warning("No branches recorded.")
        return

    summary_rows = [
        {
            "Branch": trace.branch.label or trace.branch.id,
            "Baseline": "Yes" if trace.branch.is_baseline else "No",
            "Result": "Pass" if trace.success else "Fail",
            "Score": f"{trace.score:.2f}",
            "Panic": f"{trace.panic_score:.2f}",
            "Judge": trace.panic_source,
            "Steps": len(trace.steps),
            "Tokens": trace.total_tokens_used,
        }
        for trace in result.branches
    ]
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    selected_index = st.selectbox(
        "Inspect branch",
        options=list(range(len(result.branches))),
        format_func=lambda idx: _branch_label(result.branches[idx]),
    )
    trace = result.branches[selected_index]

    st.caption(trace.branch.description)

    final_state = trace.final_state
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.code(f"deployment: {final_state.get('deployment_status', '?')}")
    col2.code(f"server: {final_state.get('server_status', '?')}")
    col3.code(f"errors: {final_state.get('error_rate', '?')}")
    col4.code(f"panic: {trace.panic_score:.2f}")
    col5.code(f"judge: {trace.panic_source}")

    if trace.judge_explanation:
        st.info(trace.judge_explanation)
    if trace.deviation_step is not None:
        st.info(f"Deviation detected at step {trace.deviation_step}")
    if trace.recovery_step is not None:
        st.success(f"Recovered at step {trace.recovery_step}")

    if not trace.steps:
        st.warning("No steps recorded for this branch.")
        return

    st.markdown("#### Steps")
    for step in trace.steps:
        with st.container(border=True):
            st.markdown(f"**Step {step.step_number}**")
            st.caption(step.observation_text)

            if step.event_injected:
                st.warning(f"Event: {step.event_injected.name} - {step.event_injected.description}")

            action_args = step.agent_action.tool_args if step.agent_action.tool_args else {}
            st.markdown(f"**Action:** `{step.agent_action.tool_name}` `{action_args}`")

            if step.agent_action.reasoning:
                st.caption(step.agent_action.reasoning[:240])
