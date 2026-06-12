from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid_q.statistics import planned_contrasts, summarize


ROOT = Path(__file__).resolve().parents[1]


def read_seed_metrics(*paths: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def write_result_report(
    seed_metrics: pd.DataFrame,
    result_dir: Path,
    table_path: Path,
    figure_path: Path,
    *,
    agents: list[str],
    names: dict[str, str],
    table_metrics: list[str],
    contrasts: list[dict],
    figure_metric: str = "return_auc",
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    seed_metrics = seed_metrics[seed_metrics["agent"].isin(agents)].copy()
    seed_metrics.to_csv(result_dir / "seed_metrics.csv", index=False)
    summary = summarize(seed_metrics)
    summary.to_csv(result_dir / "summary.csv", index=False)
    planned = planned_contrasts(seed_metrics, contrasts)
    planned.to_csv(result_dir / "planned_contrasts.csv", index=False)
    table = summary[summary["metric"].isin(table_metrics)].copy()
    table["method"] = table["agent"].map(names)
    table.to_csv(table_path, index=False)

    selected = summary[summary["metric"] == figure_metric].set_index("agent")
    available = [agent for agent in agents if agent in selected.index]
    means = selected.loc[available, "mean"].to_numpy(dtype=float)
    low = selected.loc[available, "bootstrap_ci_low"].to_numpy(dtype=float)
    high = selected.loc[available, "bootstrap_ci_high"].to_numpy(dtype=float)
    x = np.arange(len(available))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        x,
        means,
        yerr=np.vstack((means - low, high - means)),
        capsize=4,
        color="#4c78a8",
    )
    axis.set_xticks(x, [names[agent] for agent in available], rotation=20)
    axis.set_ylabel(figure_metric.replace("_", " "))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
