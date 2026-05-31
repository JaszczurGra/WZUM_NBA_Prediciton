"""
Shared core for the model-comparison experiments.

Everything here REUSES the production pipeline in src/ so that each experiment is
a true apples-to-apples variation of the real model (only the estimator / its
hyper-parameters change):

  * features -- src.features.build_feature_matrix / FEATURE_COLS
                (season-percentile of all 17 stats, identical to train.py)
  * scoring  -- the official competition rule, mirrored from
                src/predict.py::final_score:
                    per player : 10 / 8 / 6 pts for tier distance 0 / 1 / 2
                    per team   : bonus 0,0,5,10,20,40 by number of exact-tier hits

Each experiment is evaluated with Leave-One-Season-Out cross-validation on BOTH
tasks (All-NBA = 3 teams of 5, All-Rookie = 2 teams of 5) and reports two
metrics per held-out season:

  * official points  (max 90/team -> 270 All-NBA + 180 rookie = 450 total)
  * top-N overlap    (fraction of the real team that lands in our top-N)

`run_experiment(...)` writes experiments/results/<name>.json, which
plot_comparison.py turns into the comparison chart.
"""

import os
import sys
import glob
import json
import time
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

# --- make the project root importable so we can reuse src/ ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import build_feature_matrix, FEATURE_COLS, normalize_name  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
N_FEATURES = len(FEATURE_COLS)
# Monotone-constraint string matching the production model: every feature is
# oriented "higher = better", so each gets a +1 constraint.
MONOTONE = "(" + ",".join(["1"] * N_FEATURES) + ")"

# Task geometry. rookies_only restricts both train and test to is_rookie rows.
CURRENT_SEASON = "2025-26"
TASKS = {
    "all_nba": {
        "n_teams": 3, "team_size": 5,
        "labels": "allnba_labels.csv",
        "current_labels": "allnba_labels_current.csv",
        "rookies_only": False,
    },
    "all_rookie": {
        "n_teams": 2, "team_size": 5,
        "labels": "allrookie_labels.csv",
        "current_labels": "allrookie_labels_current.csv",
        "rookies_only": True,
    },
}

# ----------------------------------------------------------------------------
# Official scoring (mirror of src/predict.py::final_score)
# ----------------------------------------------------------------------------
_TIER_POINTS = {0: 10, 1: 8, 2: 6}        # points by |predicted_tier - actual_tier|
_TEAM_BONUS = [0, 0, 5, 10, 20, 40]       # bonus indexed by # of exact-tier hits in a team


def score_team(pred_names, actual_by_tier, pred_tier):
    """Score one predicted team of names against the actual tier map of a season."""
    base, exact = 0, 0
    for name in pred_names:
        n = normalize_name(name)
        actual_tier = next((t for t, names in actual_by_tier.items() if n in names), None)
        if actual_tier is None:
            continue
        diff = abs(pred_tier - actual_tier)
        if diff in _TIER_POINTS:
            base += _TIER_POINTS[diff]
            if diff == 0:
                exact += 1
    return base + (_TEAM_BONUS[exact] if exact <= 5 else 0)


def max_team_points(team_size=5):
    """Best possible score for a single team (all picks exact tier)."""
    return team_size * 10 + _TEAM_BONUS[min(team_size, 5)]


# ----------------------------------------------------------------------------
# Data loading (mirrors train.py::load_all_seasons)
# ----------------------------------------------------------------------------
def load_all_seasons(exclude_current=True):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "player_stats_*.csv")))
    if exclude_current:
        files = [f for f in files if "player_stats_2025-26.csv" not in f]
    if not files:
        raise FileNotFoundError(
            f"No player_stats_*.csv in {DATA_DIR}; run data/collect_stats.py first."
        )
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "SEASON" not in df.columns:
            df["SEASON"] = os.path.basename(f).replace("player_stats_", "").replace(".csv", "")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _actual_tiers_by_season(labels_df):
    """season -> {tier: set(normalized player names)}."""
    out = {}
    for _, r in labels_df.iterrows():
        season = str(r["season"])
        out.setdefault(season, {}).setdefault(int(r["tier"]), set()).add(
            normalize_name(r["player_name"])
        )
    return out


def _pos_weight(y_tr, policy):
    """Resolve scale_pos_weight for a fold given the policy.

    'balanced' -> neg/pos (>=1), matching train.py
    'none'     -> 1.0 (no imbalance handling)
    <number>   -> that fixed value
    """
    if policy == "none":
        return 1.0
    n_pos = max(1, int((y_tr == 1).sum()))
    n_neg = int((y_tr == 0).sum())
    balanced = n_neg / n_pos
    if policy == "balanced":
        return max(1.0, balanced)
    return float(policy)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def _evaluate_task(model_factory, pos_weight_policy, task_key, all_stats, rookie_year_df):
    cfg = TASKS[task_key]
    labels_df = pd.read_csv(os.path.join(DATA_DIR, cfg["labels"]))

    # NOTE: features are built once over all seasons, but each percentile is
    # computed *within its own season* (groupby("SEASON").rank) in
    # build_feature_matrix -- so there is no cross-season leakage into the
    # held-out fold. This matches the production pipeline in src/train.py.
    X, y, meta = build_feature_matrix(all_stats, labels_df, rookie_year_df)
    if cfg["rookies_only"]:
        mask = meta["is_rookie"].values
        X, y, meta = X[mask], y[mask], meta[mask].reset_index(drop=True)

    seasons_arr = meta["SEASON"].values
    names_arr = meta["PLAYER_NAME"].values
    actual_tiers = _actual_tiers_by_season(labels_df)

    n_teams, team_size = cfg["n_teams"], cfg["team_size"]
    n_select = n_teams * team_size
    max_pts = n_teams * max_team_points(team_size)

    logo = LeaveOneGroupOut()
    per_season = {}
    for tr_idx, te_idx in logo.split(X, y, seasons_arr):
        season = seasons_arr[te_idx][0]
        if y[te_idx].sum() == 0 or season not in actual_tiers:
            continue

        model = model_factory(_pos_weight(y[tr_idx], pos_weight_policy))
        model.fit(X[tr_idx], y[tr_idx])
        probs = model.predict_proba(X[te_idx])[:, 1]
        order = np.argsort(probs)[::-1]            # best first
        te_names = names_arr[te_idx]

        # Official points: fill teams in rank order (top 5 -> tier 1, etc.)
        pts = 0
        for t in range(n_teams):
            sel = order[t * team_size:(t + 1) * team_size]
            pts += score_team([te_names[i] for i in sel], actual_tiers[season], t + 1)

        # Top-N overlap
        top_n = set(order[:n_select].tolist())
        actual_idx = set(np.where(y[te_idx] == 1)[0].tolist())
        overlap = len(top_n & actual_idx) / n_select

        per_season[season] = {"points": int(pts), "overlap": round(float(overlap), 4)}

    pts_list = [v["points"] for v in per_season.values()]
    ov_list = [v["overlap"] for v in per_season.values()]
    return {
        "n_teams": n_teams,
        "team_size": team_size,
        "max_points_per_season": max_pts,
        "n_seasons": len(per_season),
        "mean_points": round(float(np.mean(pts_list)), 2) if pts_list else 0.0,
        "std_points": round(float(np.std(pts_list)), 2) if pts_list else 0.0,
        "mean_overlap": round(float(np.mean(ov_list)), 4) if ov_list else 0.0,
        "per_season": dict(sorted(per_season.items())),
    }


def _evaluate_current(model_factory, pos_weight_policy, all_stats, rookie_year_df):
    """Train on all history, predict CURRENT_SEASON, score vs *_current.csv.

    This mirrors src/predict.py exactly (full fit on history -> rank current
    players -> fill teams in order -> official scoring), so baseline here should
    reproduce the ~322/450 the production model gets on 2025-26.
    Returns None if the current-season files aren't present.
    """
    current_path = os.path.join(DATA_DIR, f"player_stats_{CURRENT_SEASON}.csv")
    if not os.path.exists(current_path):
        return None
    current_stats = pd.read_csv(current_path, low_memory=False)

    out, total, max_total = {}, 0, 0
    for task_key in ("all_nba", "all_rookie"):
        cfg = TASKS[task_key]
        cur_labels_path = os.path.join(DATA_DIR, cfg["current_labels"])
        if not os.path.exists(cur_labels_path):
            return None

        # --- train on all history ---
        labels_df = pd.read_csv(os.path.join(DATA_DIR, cfg["labels"]))
        X, y, meta = build_feature_matrix(all_stats, labels_df, rookie_year_df)
        if cfg["rookies_only"]:
            mask = meta["is_rookie"].values
            X, y = X[mask], y[mask]
        model = model_factory(_pos_weight(y, pos_weight_policy))
        model.fit(X, y)

        # --- predict current season ---
        Xc, _, metac = build_feature_matrix(current_stats, None, rookie_year_df)
        if cfg["rookies_only"]:
            cmask = metac["is_rookie"].values
            Xc, metac = Xc[cmask], metac[cmask].reset_index(drop=True)
        probs = model.predict_proba(Xc)[:, 1]
        order = np.argsort(probs)[::-1]
        names = metac["PLAYER_NAME"].values

        # --- score vs actual current-season tiers ---
        cur_labels = pd.read_csv(cur_labels_path)
        actual = {}
        for _, r in cur_labels.iterrows():
            actual.setdefault(int(r["tier"]), set()).add(normalize_name(r["player_name"]))

        n_teams, team_size = cfg["n_teams"], cfg["team_size"]
        pts = 0
        for t in range(n_teams):
            sel = order[t * team_size:(t + 1) * team_size]
            pts += score_team([names[i] for i in sel], actual, t + 1)
        mx = n_teams * max_team_points(team_size)
        out[task_key] = {"points": int(pts), "max_points": mx}
        total += pts
        max_total += mx

    out["combined"] = {"points": total, "max_points": max_total}
    return out


def run_experiment(name, model_factory, pos_weight_policy="balanced",
                   description="", params=None, save=True):
    """Run one experiment (both tasks, LOSO-CV) and write results/<name>.json.

    model_factory(pos_weight) -> a fresh, unfitted sklearn-style estimator with
    .fit / .predict_proba. pos_weight is the resolved scale_pos_weight for the
    fold (estimators that don't use it, e.g. RandomForest, may ignore it).
    """
    print(f"\n=== Experiment: {name} ===")
    if description:
        print(f"    {description}")

    t0 = time.time()
    all_stats = load_all_seasons(exclude_current=True)
    rookie_year_df = pd.read_csv(os.path.join(DATA_DIR, "players_rookie_year.csv"))

    tasks_out = {}
    for task_key in ("all_nba", "all_rookie"):
        print(f"  -> {task_key}: Leave-One-Season-Out CV ...", flush=True)
        tasks_out[task_key] = _evaluate_task(
            model_factory, pos_weight_policy, task_key, all_stats, rookie_year_df
        )

    print(f"  -> current season ({CURRENT_SEASON}): train-on-history, predict, score ...",
          flush=True)
    current_out = _evaluate_current(model_factory, pos_weight_policy, all_stats, rookie_year_df)

    # Combined official total (/450) over seasons present in BOTH tasks.
    nba_ps = tasks_out["all_nba"]["per_season"]
    rk_ps = tasks_out["all_rookie"]["per_season"]
    common_seasons = sorted(set(nba_ps) & set(rk_ps))
    combined = {s: nba_ps[s]["points"] + rk_ps[s]["points"] for s in common_seasons}
    comb_list = list(combined.values())
    max_total = (tasks_out["all_nba"]["max_points_per_season"]
                 + tasks_out["all_rookie"]["max_points_per_season"])

    result = {
        "name": name,
        "description": description,
        "params": params or {},
        "pos_weight_policy": pos_weight_policy,
        "n_features": N_FEATURES,
        "seed": SEED,
        "tasks": tasks_out,
        "combined": {
            "max_points_per_season": max_total,
            "n_seasons": len(combined),
            "mean_points": round(float(np.mean(comb_list)), 2) if comb_list else 0.0,
            "std_points": round(float(np.std(comb_list)), 2) if comb_list else 0.0,
            "per_season": combined,
        },
        "current_season": {"season": CURRENT_SEASON, **current_out} if current_out else None,
        "runtime_sec": round(time.time() - t0, 1),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    nba, rk, comb = tasks_out["all_nba"], tasks_out["all_rookie"], result["combined"]
    print(f"\n  All-NBA   : {nba['mean_points']:6.1f} / {nba['max_points_per_season']} pts"
          f"   | overlap {nba['mean_overlap']:.1%}  ({nba['n_seasons']} seasons)")
    print(f"  All-Rookie: {rk['mean_points']:6.1f} / {rk['max_points_per_season']} pts"
          f"   | overlap {rk['mean_overlap']:.1%}  ({rk['n_seasons']} seasons)")
    print(f"  COMBINED  : {comb['mean_points']:6.1f} / {max_total} pts per season"
          f"   ({comb['n_seasons']} seasons)")
    if current_out:
        cc = current_out["combined"]
        print(f"  CURRENT {CURRENT_SEASON}: {cc['points']} / {cc['max_points']} pts"
              f"   (All-NBA {current_out['all_nba']['points']}, "
              f"Rookie {current_out['all_rookie']['points']})")
    print(f"  done in {result['runtime_sec']}s")

    if save:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  saved -> {os.path.relpath(path, ROOT)}")

    return result
