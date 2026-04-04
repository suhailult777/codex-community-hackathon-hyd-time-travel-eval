"""Interactive multiverse DAG visualization for branch traces."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from core.models import BranchTrace, EvalResult, StepRecord

try:
    from streamlit_echarts import st_echarts

    HAS_ECHARTS = True
except Exception:  # noqa: BLE001
    HAS_ECHARTS = False

BASELINE_COLOR = "#38bdf8"
SUCCESS_COLOR = "#22c55e"
FAIL_COLOR = "#ef4444"
PANIC_COLOR = "#f59e0b"
EVENT_COLOR = "#fb923c"
ROOT_COLOR = "#6366f1"
TEXT_COLOR = "#e2e8f0"
LINE_COLOR = "#64748b"


def _trace_divergence_step(trace: BranchTrace) -> int:
    for step in trace.steps:
        if step.event_injected is not None:
            return step.step_number
    if trace.deviation_step is not None:
        return trace.deviation_step
    return 0


def _global_max_step(result: EvalResult) -> int:
    max_step = 0
    for trace in result.branches:
        for step in trace.steps:
            max_step = max(max_step, step.step_number)
    return max_step


def _baseline_lookup(result: EvalResult) -> dict[int, StepRecord]:
    baseline = next((trace for trace in result.branches if trace.branch.is_baseline), None)
    if baseline is None:
        return {}
    return {step.step_number: step for step in baseline.steps}


def _branch_row_offset(order: int) -> int:
    level = (order // 2) + 1
    distance = level * 150
    return -distance if order % 2 == 0 else distance


def _status_style(trace: BranchTrace) -> tuple[str, str]:
    if trace.panic_score >= 0.65:
        return "High Panic", PANIC_COLOR
    if trace.success:
        return "Recovered", SUCCESS_COLOR
    return "Failed", FAIL_COLOR


def _state_summary(step: StepRecord) -> str:
    state = step.world_state_after
    return (
        f"server={state.get('server_status', '?')} | "
        f"deploy={state.get('deployment_status', '?')} | "
        f"cpu={state.get('cpu_utilization', '?')}% | "
        f"mem={state.get('memory_utilization', '?')}% | "
        f"err={state.get('error_rate', '?')}"
    )


def _tooltip_for_step(trace: BranchTrace, step: StepRecord) -> str:
    event_text = ""
    if step.event_injected:
        event_text = f"\nEvent: {step.event_injected.name} - {step.event_injected.description}"
    return (
        f"{trace.branch.label or trace.branch.id}\n"
        f"Step {step.step_number}\n"
        f"Action: {step.agent_action.tool_name}{event_text}\n"
        f"{_state_summary(step)}\n"
        f"Score={trace.score:.2f} | Panic={trace.panic_score:.2f}"
    )


def build_multiverse_dag(result: EvalResult) -> Dict[str, Any]:
    """Build an explicit multiverse DAG for baseline and alternate timelines."""
    max_step = _global_max_step(result)
    baseline_steps = _baseline_lookup(result)
    alternate_traces = [trace for trace in result.branches if not trace.branch.is_baseline]
    chart_height = max(420, 260 + len(alternate_traces) * 130)
    center_y = chart_height // 2
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    categories = [
        {"name": "Baseline"},
        {"name": "Recovered"},
        {"name": "Failed"},
        {"name": "High Panic"},
    ]

    root_id = "task_root"
    nodes.append(
        {
            "id": root_id,
            "name": "Base Task",
            "x": 20,
            "y": center_y,
            "symbolSize": 56,
            "category": 0,
            "itemStyle": {"color": ROOT_COLOR, "borderColor": "#a5b4fc", "borderWidth": 3},
            "label": {"color": TEXT_COLOR, "fontWeight": "bold"},
            "tooltip": {"formatter": result.base_task},
        }
    )

    previous_id = root_id
    for step_number in range(max_step + 1):
        baseline_step = baseline_steps.get(step_number)
        action_label = baseline_step.agent_action.tool_name if baseline_step else "timeline"
        node_id = f"baseline_step_{step_number}"
        nodes.append(
            {
                "id": node_id,
                "name": f"Happy Path\nT{step_number}\n{action_label}",
                "x": 170 + (step_number * 180),
                "y": center_y,
                "symbolSize": 42,
                "category": 0,
                "itemStyle": {"color": BASELINE_COLOR, "borderColor": "#bae6fd", "borderWidth": 2},
                "label": {"color": TEXT_COLOR},
                "tooltip": {
                    "formatter": _tooltip_for_step(
                        next((trace for trace in result.branches if trace.branch.is_baseline), result.branches[0]),
                        baseline_step,
                    )
                    if baseline_step
                    else f"Happy Path\nStep {step_number}\nShared timeline before or after divergence."
                },
            }
        )
        links.append(
            {
                "source": previous_id,
                "target": node_id,
                "lineStyle": {"color": BASELINE_COLOR, "width": 4, "curveness": 0.0},
            }
        )
        previous_id = node_id

    for branch_index, trace in enumerate(alternate_traces):
        divergence_step = min(_trace_divergence_step(trace), max_step)
        status_name, branch_color = _status_style(trace)
        source_id = f"baseline_step_{divergence_step}"
        previous_branch_id: str | None = None
        branch_y = center_y + _branch_row_offset(branch_index)
        branch_steps = [step for step in trace.steps if step.step_number >= divergence_step]

        for step in branch_steps:
            node_id = f"{trace.branch.id}_step_{step.step_number}"
            is_first_branch_step = previous_branch_id is None
            is_last_branch_step = step == branch_steps[-1]
            title_lines = []
            if is_first_branch_step:
                title_lines.append(trace.branch.label or trace.branch.id)
            else:
                title_lines.append(f"{trace.branch.id.upper()}")
            title_lines.append(f"T{step.step_number}")
            if step.event_injected is not None:
                title_lines.append(step.event_injected.name)
            elif is_last_branch_step:
                title_lines.append("PASS" if trace.success else "FAIL")
            else:
                title_lines.append(step.agent_action.tool_name)

            border_color = EVENT_COLOR if step.event_injected is not None else branch_color
            symbol_size = 50 if step.event_injected is not None else 42
            category_index = next(
                index for index, category in enumerate(categories) if category["name"] == status_name
            )

            nodes.append(
                {
                    "id": node_id,
                    "name": "\n".join(title_lines),
                    "x": 170 + (step.step_number * 180),
                    "y": branch_y,
                    "symbolSize": symbol_size,
                    "category": category_index,
                    "itemStyle": {"color": branch_color, "borderColor": border_color, "borderWidth": 3},
                    "label": {"color": TEXT_COLOR},
                    "tooltip": {"formatter": _tooltip_for_step(trace, step)},
                }
            )

            links.append(
                {
                    "source": previous_branch_id or source_id,
                    "target": node_id,
                    "lineStyle": {
                        "color": branch_color if previous_branch_id else border_color,
                        "width": 3,
                        "curveness": 0.28 if previous_branch_id is None else 0.08,
                    },
                }
            )
            previous_branch_id = node_id

    return {
        "nodes": nodes,
        "links": links,
        "categories": categories,
        "height_px": chart_height,
    }


def build_multiverse_dot(result: EvalResult) -> str:
    """Build a Graphviz DOT fallback for environments without ECharts."""
    graph = build_multiverse_dag(result)
    category_colors = {
        "Baseline": BASELINE_COLOR,
        "Recovered": SUCCESS_COLOR,
        "Failed": FAIL_COLOR,
        "High Panic": PANIC_COLOR,
    }

    lines = [
        "digraph multiverse {",
        '  rankdir=LR;',
        '  graph [bgcolor="transparent", pad="0.4", nodesep="0.5", ranksep="1.0"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="white"];',
        '  edge [color="#64748b", penwidth=2.0];',
    ]

    category_lookup = {index: category["name"] for index, category in enumerate(graph["categories"])}
    for node in graph["nodes"]:
        category_name = category_lookup.get(node.get("category", 0), "Baseline")
        fill = node.get("itemStyle", {}).get("color", category_colors.get(category_name, BASELINE_COLOR))
        label = node["name"].replace("\n", "\\n")
        lines.append(f'  "{node["id"]}" [label="{label}", fillcolor="{fill}"];')

    for link in graph["links"]:
        lines.append(f'  "{link["source"]}" -> "{link["target"]}";')

    lines.append("}")
    return "\n".join(lines)


def render_tree_view(result: EvalResult) -> None:
    """Render the multiverse DAG."""
    st.caption(
        "Interactive multiverse DAG: the blue happy path stays centered while alternate timelines fork at the "
        "exact step where anomalies hit. Drag, zoom, and hover to inspect each reality."
    )
    graph = build_multiverse_dag(result)

    if HAS_ECHARTS:
        option = {
            "backgroundColor": "transparent",
            "legend": [
                {
                    "data": [category["name"] for category in graph["categories"]],
                    "textStyle": {"color": TEXT_COLOR},
                    "top": 0,
                }
            ],
            "tooltip": {"trigger": "item", "triggerOn": "mousemove", "confine": True},
            "animationDuration": 700,
            "series": [
                {
                    "type": "graph",
                    "layout": "none",
                    "data": graph["nodes"],
                    "links": graph["links"],
                    "categories": graph["categories"],
                    "roam": True,
                    "draggable": False,
                    "focusNodeAdjacency": True,
                    "edgeSymbol": ["none", "arrow"],
                    "edgeSymbolSize": [0, 10],
                    "lineStyle": {"color": LINE_COLOR, "opacity": 0.9},
                    "label": {
                        "show": True,
                        "position": "right",
                        "formatter": "{b}",
                        "color": TEXT_COLOR,
                        "fontSize": 11,
                        "lineHeight": 14,
                    },
                    "emphasis": {
                        "focus": "adjacency",
                        "lineStyle": {"width": 4},
                        "label": {"fontWeight": "bold"},
                    },
                }
            ],
        }
        st_echarts(options=option, height=f"{graph['height_px']}px")
        return

    st.graphviz_chart(build_multiverse_dot(result), use_container_width=True)
