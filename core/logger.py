"""Structured trace logging that writes JSONL files and prints rich console output."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import config

console = Console()


def _safe_console_text(text: str) -> str:
    """Best-effort console-safe text for terminals with limited encodings."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _should_emit_console_output() -> bool:
    """Avoid writing to captured stdout during automated test runs."""
    return "PYTEST_CURRENT_TEST" not in os.environ


class TraceLogger:
    """Writes one JSONL file per evaluation run."""

    def __init__(self, run_id: Optional[str] = None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_id = run_id or f"run_{ts}"
        self.log_path: Path = config.LOG_DIR / f"{self.run_id}.jsonl"
        self._file = open(self.log_path, "a", encoding="utf-8")

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append a structured log line."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "type": event_type,
            **data,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def log_step(self, branch_id: str, step: int, observation: str, action: str) -> None:
        self.log("step", {"branch_id": branch_id, "step": step, "observation": observation, "action": action})

    def log_branch_result(self, branch_id: str, success: bool, score: float) -> None:
        self.log("branch_result", {"branch_id": branch_id, "success": success, "score": score})

    def log_eval_result(self, data: Dict[str, Any]) -> None:
        self.log("eval_result", data)

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def print_header(title: str) -> None:
    if not _should_emit_console_output():
        return
    console.print(Panel(f"[bold cyan]{_safe_console_text(title)}[/bold cyan]", expand=False))


def print_step(branch_id: str, step: int, observation: str, action: str) -> None:
    if not _should_emit_console_output():
        return
    safe_observation = _safe_console_text(observation[:80])
    safe_action = _safe_console_text(action)
    console.print(
        f"  [dim]Step {step}[/dim]  > obs: [yellow]{safe_observation}[/yellow]  > act: [green]{safe_action}[/green]"
    )


def print_branch_result(branch_id: str, success: bool, score: float) -> None:
    if not _should_emit_console_output():
        return
    icon = "OK" if success else "FAIL"
    console.print(f"  {icon} Branch [bold]{_safe_console_text(branch_id)}[/bold]  score={score:.2f}")


def print_eval_summary(rs: float, success_rate: float, stability: float) -> None:
    if not _should_emit_console_output():
        return
    table = Table(title="TTE Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Robustness Score", f"{rs:.3f}")
    table.add_row("Success Rate", f"{success_rate:.1%}")
    table.add_row("Stability Score", f"{stability:.3f}")
    console.print(table)
