"""Hero metrics panel for the evaluation result."""

from __future__ import annotations

import streamlit as st

from core.models import EvalResult


def render_metrics_panel(result: EvalResult) -> None:
    """Render the high-level metrics row."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Robustness Score",
            value=f"{result.robustness_score:.2f}",
            delta=f"{(result.robustness_score - 1.0):.2f}" if result.robustness_score < 1.0 else "Perfect",
            delta_color="inverse" if result.robustness_score < 1.0 else "off",
        )

    with c2:
        st.metric(
            label="Success Rate",
            value=f"{result.success_rate:.0%}",
            delta=f"{len([branch for branch in result.branches if branch.success])}/{len(result.branches)} branches",
            delta_color="off",
        )

    with c3:
        st.metric(
            label="Stability Score",
            value=f"{result.stability_score:.2f}",
            delta="Low variance" if result.stability_score > 0.8 else "Higher variance",
            delta_color="off",
        )

    with c4:
        recovery_text = f"{result.mean_recovery_time:.1f} steps" if result.mean_recovery_time else "N/A"
        st.metric(
            label="Average Recovery",
            value=recovery_text,
            delta=f"{result.total_tokens:,} tokens",
            delta_color="off",
        )
