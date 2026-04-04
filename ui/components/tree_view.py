"""Branch tree visualization using streamlit-echarts when available."""

from __future__ import annotations

import streamlit as st

from core.models import EvalResult

try:
    from streamlit_echarts import st_echarts

    HAS_ECHARTS = True
except ImportError:
    HAS_ECHARTS = False


def render_tree_view(result: EvalResult) -> None:
    """Render the branching result tree."""
    if not HAS_ECHARTS:
        _render_fallback(result)
        return

    children = []
    for trace in result.branches:
        color = "#22c55e" if trace.success else "#ef4444"
        status = "PASS" if trace.success else "FAIL"
        label = trace.branch.label or trace.branch.id
        children.append(
            {
                "name": f"{status} {label}\nscore: {trace.score:.2f}",
                "value": trace.score,
                "itemStyle": {"color": color, "borderColor": color},
                "label": {"color": "#fff"},
            }
        )

    tree_data = [
        {
            "name": f"Task: {result.base_task[:48]}",
            "children": children,
            "itemStyle": {"color": "#3b82f6"},
            "label": {"color": "#fff"},
        }
    ]

    option = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "tree",
                "data": tree_data,
                "top": "5%",
                "left": "15%",
                "bottom": "5%",
                "right": "15%",
                "symbolSize": 14,
                "orient": "TB",
                "label": {
                    "position": "bottom",
                    "verticalAlign": "middle",
                    "align": "center",
                    "fontSize": 12,
                    "fontFamily": "monospace",
                },
                "leaves": {
                    "label": {
                        "position": "bottom",
                        "verticalAlign": "middle",
                        "align": "center",
                    }
                },
                "emphasis": {"focus": "descendant"},
                "expandAndCollapse": False,
                "animationDuration": 400,
                "animationDurationUpdate": 500,
                "lineStyle": {"color": "#64748b", "width": 2, "curveness": 0.4},
            }
        ],
    }

    st_echarts(options=option, height="360px")


def _render_fallback(result: EvalResult) -> None:
    """Fallback summary when ECharts is unavailable."""
    st.markdown("### Branch Results")
    for trace in result.branches:
        status = "PASS" if trace.success else "FAIL"
        label = trace.branch.label or trace.branch.id
        baseline = " | baseline" if trace.branch.is_baseline else ""
        st.markdown(
            f"**{status}** {label}{baseline} | score `{trace.score:.2f}` | "
            f"steps {len(trace.steps)} | tokens {trace.total_tokens_used}"
        )
