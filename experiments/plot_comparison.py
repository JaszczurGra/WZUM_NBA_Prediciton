"""
Compare all experiments that have a results/<name>.json.

Always:
  * prints a markdown summary table
  * writes experiments/results/summary.csv
If matplotlib is installed, also writes TWO points-only charts:
  * experiments/results/comparison_avg.png      -- mean over all LOSO seasons (1996-97..2024-25)
  * experiments/results/comparison_current.png  -- the held-out 2025-26 season

Both charts: zoomed y-axis (so small differences are visible), value labels on
each bar, the chosen production model highlighted, and NO titles -- only axis
labels, per request.

Run:  python experiments/plot_comparison.py
      (charts need matplotlib:  uv pip install --python .venv/bin/python matplotlib)
"""

import os
import json
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# The production model is stored under the name "baseline"; show it as the
# chosen/final model and highlight it in the charts.
CHOSEN_KEY = "baseline"
CHOSEN_LABEL = "final (chosen)"
HIGHLIGHT = "#C44E52"   # chosen model
NORMAL = "#4C72B0"      # other variants


def display_name(name):
    return CHOSEN_LABEL if name == CHOSEN_KEY else name


def load_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if "combined" not in r:        # skip anything that isn't an experiment result
            continue
        cur = r.get("current_season") or {}
        cur_pts = cur.get("combined", {}).get("points") if cur else None
        nba_ov = r["tasks"]["all_nba"]["mean_overlap"]
        rk_ov = r["tasks"]["all_rookie"]["mean_overlap"]
        rows.append({
            "name": r["name"],
            "avg_pts": r["combined"]["mean_points"],
            "avg_std": r["combined"]["std_points"],
            "avg_max": r["combined"]["max_points_per_season"],
            "current_pts": cur_pts,
            "current_max": cur.get("combined", {}).get("max_points") if cur else None,
            "allnba_pts": r["tasks"]["all_nba"]["mean_points"],
            "rookie_pts": r["tasks"]["all_rookie"]["mean_points"],
            "allnba_overlap": nba_ov,
            "rookie_overlap": rk_ov,
            "mean_overlap": (nba_ov + rk_ov) / 2,
            "description": r.get("description", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("avg_pts", ascending=False).reset_index(drop=True)
    return df


def print_table(df):
    if df.empty:
        print("No results found. Run the experiments first (e.g. python experiments/run_all.py).")
        return
    max_total = int(df["avg_max"].iloc[0])
    print(f"\nComparison (official points, max {max_total}/season):\n")
    print(f"| {'variant':16} | {'avg points':>12} | {'mean overlap':>12} | {'current 2025-26':>15} |")
    print("|" + "-" * 18 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 17 + "|")
    for _, r in df.iterrows():
        cur = f"{int(r['current_pts'])}" if pd.notna(r["current_pts"]) else "n/a"
        print(f"| {display_name(r['name']):16} | {r['avg_pts']:12.1f} "
              f"| {r['mean_overlap']:11.1%} | {cur:>15} |")
    print()


def _zoom_limits(vals):
    """Y-axis limits that zoom in so small differences look large.

    Floor sits below the smallest bar (not at 0), ceiling a bit above the top.
    """
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) if hi > lo else max(hi * 0.1, 1.0)
    ymin = max(0.0, lo - rng * 0.8 - 3)
    ymax = hi + rng * 0.45 + 3
    return ymin, ymax


def _bar_chart(names, values, ylabel, out_path, value_fmt="{:.0f}"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [display_name(n) for n in names]
    colors = [HIGHLIGHT if n == CHOSEN_KEY else NORMAL for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(x, values, color=colors, width=0.7)

    ymin, ymax = _zoom_limits(values)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    # value labels just above each bar
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (ymax - ymin) * 0.01,
                value_fmt.format(v), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_charts(df):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[info] matplotlib not installed -> skipping PNGs. Install with:\n"
              "       uv pip install --python .venv/bin/python matplotlib")
        return

    # Graph 1 -- POINTS: average official points per season, sorted best-first.
    pts_df = df.sort_values("avg_pts", ascending=False)
    _bar_chart(
        pts_df["name"].tolist(),
        pts_df["avg_pts"].tolist(),
        ylabel="average points per season (max 450)",
        out_path=os.path.join(RESULTS_DIR, "comparison_points.png"),
        value_fmt="{:.0f}",
    )
    print(f"[ok] points chart  -> {os.path.relpath(os.path.join(RESULTS_DIR, 'comparison_points.png'), os.path.dirname(HERE))}")

    # Graph 2 -- OVERLAP: mean top-N overlap with the real teams (% ), sorted best-first.
    ov_df = df.sort_values("mean_overlap", ascending=False)
    _bar_chart(
        ov_df["name"].tolist(),
        [v * 100 for v in ov_df["mean_overlap"].tolist()],
        ylabel="mean top-N overlap with real teams (%)",
        out_path=os.path.join(RESULTS_DIR, "comparison_overlap.png"),
        value_fmt="{:.1f}",
    )
    print(f"[ok] overlap chart -> {os.path.relpath(os.path.join(RESULTS_DIR, 'comparison_overlap.png'), os.path.dirname(HERE))}")


def main():
    df = load_results()
    print_table(df)
    if not df.empty:
        csv = os.path.join(RESULTS_DIR, "summary.csv")
        df.to_csv(csv, index=False)
        print(f"[ok] summary saved -> {os.path.relpath(csv, os.path.dirname(HERE))}")
        save_charts(df)


if __name__ == "__main__":
    main()
