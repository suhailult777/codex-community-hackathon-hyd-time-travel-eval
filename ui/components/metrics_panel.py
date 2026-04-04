"""Hero metrics panel for the evaluation result."""

from __future__ import annotations

import streamlit as st

from core.models import EvalResult


def _panic_label(panic_score: float) -> str:
    if panic_score < 0.20:
        return "Calm under stress"
    if panic_score < 0.50:
        return "Some flailing"
    return "High panic"


def render_metrics_panel(result: EvalResult) -> None:
    """Render the high-level metrics row."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Robustness Score",
            value=f"{result.robustness_score:.2f}",
            delta=f"{(result.robustness_score - 1.0):.2f}" if result.robustness_score < 1.0 else "Perfect",
            delta_color="inverse" if result.robustness_score < 1.0 else "off",
        )

    with col2:
        succeeded = len([branch for branch in result.branches if branch.success])
        st.metric(
            label="Success Rate",
            value=f"{result.success_rate:.0%}",
            delta=f"{succeeded}/{len(result.branches)} branches",
            delta_color="off",
        )

    with col3:
        st.metric(
            label="Stability Score",
            value=f"{result.stability_score:.2f}",
            delta="Low variance" if result.stability_score > 0.8 else "Higher variance",
            delta_color="off",
        )

    with col4:
        recovery_suffix = (
            f"Recovery {result.mean_recovery_time:.1f} steps"
            if result.mean_recovery_time is not None
            else "Recovery N/A"
        )
        st.metric(
            label="Panic Score",
            value=f"{result.mean_panic_score:.2f}",
            delta=f"{_panic_label(result.mean_panic_score)} | {recovery_suffix}",
            delta_color="inverse" if result.mean_panic_score >= 0.5 else "off",
        )
