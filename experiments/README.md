# Model comparison experiments

Each `exp_*.py` trains a **variant** of the NBA award model and evaluates it with
**Leave-One-Season-Out cross-validation**, reusing the production pipeline in
`src/` (same features, same official scoring). Results are written to
`experiments/results/<name>.json`; `plot_comparison.py` turns them into a chart.

## What is measured

Per held-out season, two metrics on both tasks (All-NBA = 3 teams of 5,
All-Rookie = 2 teams of 5):

- **official points** — the competition rule (10/8/6 by tier distance + team
  bonus 0/0/5/10/20/40). Max 270 All-NBA + 180 rookie = **450/season**.
- **top-N overlap** — fraction of the real team landing in our top-N.

## Variants

| script | what changes |
|--------|--------------|
| `exp_baseline.py`        | production config (XGB depth 5, lr 0.05, 300 trees, monotone, dynamic `scale_pos_weight`) |
| `exp_shallow_fast.py`    | shallow trees (depth 3, lr 0.10, 200 trees) |
| `exp_deep_slow.py`       | deep trees, small step (depth 8, lr 0.02, 800 trees) — slowest |
| `exp_strong_reg.py`      | heavy regularization (`reg_lambda`, `min_child_weight`, `gamma`, subsample/colsample) |
| `exp_no_class_weight.py` | baseline but `scale_pos_weight=1` (no imbalance handling) |
| `exp_xgb_no_monotone.py` | ablation: baseline but monotone constraints OFF |
| `exp_random_forest.py`   | different model family (RandomForest, no monotone) |

## How to run

```bash
# one at a time
python experiments/exp_baseline.py

# or all of them
python experiments/run_all.py

# then chart the results
python experiments/plot_comparison.py     # pip install matplotlib for the PNG
```

Outputs land in `experiments/results/`:
`<name>.json` (per run), `summary.csv`, and `comparison.png` (if matplotlib is
installed).

Tweak a variant by editing its `PARAMS` dict; add a new one by copying any
`exp_*.py` and adding it to the list in `run_all.py`.
