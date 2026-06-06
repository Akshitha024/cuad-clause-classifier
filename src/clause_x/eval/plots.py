"""Per-clause F1 bar chart, sorted descending."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_per_clause(summary_json: Path, out_path: Path) -> Path:
    rows = json.loads(summary_json.read_text())
    rows.sort(key=lambda r: r["f1"], reverse=True)
    clauses = [r["clause"] for r in rows]
    f1s = [r["f1"] for r in rows]
    fig_w = max(8.0, 0.25 * len(clauses))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(range(len(clauses)), f1s)
    ax.set_xticks(range(len(clauses)))
    ax.set_xticklabels(clauses, rotation=80, ha="right", fontsize=7)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("clause-x: per-clause F1 on CUAD val split")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
