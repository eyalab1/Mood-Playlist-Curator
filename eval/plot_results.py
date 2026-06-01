"""Plot evaluation results saved by eval/run_eval.py.

Reads the latest eval_run from the DB (or a specific run id) and produces:
  - eval/plots/metric_ablation.png       (bar chart of 4 metrics x 2 configs)
  - eval/plots/cross_prompt_diversity.png (bar chart of % unique tracks)

Run after eval/run_eval.py has completed at least once:
    python -m eval.plot_results
    python -m eval.plot_results --run-id 2     # specific run
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from db import get_conn

METRICS = ["artist_diversity", "theme_coverage", "explanation_quality", "mood_fit"]
PLOT_DIR = Path(__file__).parent / "plots"


def _latest_run_id() -> int | None:
    with get_conn() as conn:
        row = conn.execute("SELECT MAX(id) AS id FROM eval_runs").fetchone()
    return row["id"] if row and row["id"] is not None else None


def _load_results(run_id: int) -> tuple[dict, dict]:
    """Return (scores_by_config, tracks_by_config)."""
    scores: dict[str, list[dict]] = {}
    tracks: dict[str, list[tuple[str, str]]] = {}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT prompt_id, metrics_json, playlist_json FROM eval_results "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    for r in rows:
        cfg, _, _ = r["prompt_id"].partition(":")
        metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
        playlist = json.loads(r["playlist_json"]) if r["playlist_json"] else {}
        scores.setdefault(cfg, []).append(metrics)
        for t in playlist.get("tracks", []):
            tracks.setdefault(cfg, []).append(
                (t.get("track_name", ""), t.get("artist", ""))
            )
    return scores, tracks


def _averages(scores: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cfg, rows in scores.items():
        if not rows:
            continue
        out[cfg] = {
            m: sum(r.get(m, 0.0) for r in rows) / len(rows) for m in METRICS
        }
    return out


def _plot_metric_ablation(averaged: dict[str, dict[str, float]]) -> Path:
    configs = list(averaged.keys())
    x = list(range(len(METRICS)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, cfg in enumerate(configs):
        vals = [averaged[cfg][m] for m in METRICS]
        offset = (i - (len(configs) - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            label=cfg.replace("_", "-"),
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in METRICS], fontsize=9
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("RAG ablation: metric comparison across 15 prompts")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = PLOT_DIR / "metric_ablation.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_cross_prompt_diversity(tracks_by_config: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = []
    pcts = []
    annotations = []
    for cfg, items in tracks_by_config.items():
        if not items:
            continue
        keys = {(n.strip().lower(), a.strip().lower()) for n, a in items}
        total = len(items)
        unique = len(keys)
        labels.append(cfg.replace("_", "-"))
        pcts.append(unique / total * 100 if total else 0.0)
        annotations.append(f"{unique} / {total}")

    bars = ax.bar(labels, pcts, color=["#4C72B0", "#DD8452"])
    for b, p, ann in zip(bars, pcts, annotations):
        ax.text(
            b.get_x() + b.get_width() / 2,
            p + 1.0,
            f"{p:.0f}%\n({ann})",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("% unique tracks across all prompts")
    ax.set_title("Cross-prompt diversity: with vs without RAG")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = PLOT_DIR / "cross_prompt_diversity.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    run_id: int | None = None
    for i, a in enumerate(sys.argv):
        if a == "--run-id" and i + 1 < len(sys.argv):
            run_id = int(sys.argv[i + 1])
    if run_id is None:
        run_id = _latest_run_id()
    if run_id is None:
        print("No eval runs found. Run `python -m eval.run_eval --full` first.")
        return

    scores, tracks = _load_results(run_id)
    if not scores:
        print(f"No results found for run_id={run_id}.")
        return

    averaged = _averages(scores)
    print(f"Plotting eval run_id={run_id}")
    print(f"Configs: {', '.join(averaged.keys())}")
    for cfg, av in averaged.items():
        print(
            f"  {cfg:<14} runs={len(scores[cfg])}  "
            + "  ".join(f"{m[:9]}={v:.3f}" for m, v in av.items())
        )

    p1 = _plot_metric_ablation(averaged)
    p2 = _plot_cross_prompt_diversity(tracks)
    print(f"\nWrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
