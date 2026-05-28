"""
Train All-NBA and All-Rookie XGBoost models.

Run once after data collection:
  python src/train.py

Saves:
  models/allnba_model.pkl
  models/rookie_model.pkl
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import build_feature_matrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


#Skipping the last season data 
def load_all_seasons() -> pd.DataFrame:
    pattern = os.path.join(DATA_DIR, "player_stats_*.csv")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "player_stats_2025-26.csv" not in f]
    if not files:
        raise FileNotFoundError(f"No stats CSVs found at {pattern}. Run collect_stats.py first.")
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "SEASON" not in df.columns:
            season = os.path.basename(f).replace("player_stats_", "").replace(".csv", "")
            df["SEASON"] = season
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _xgb_model(pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def evaluate_logo(X, y, groups, n_select: int, label: str):
    """Leave-one-season-out CV; prints avg top-N overlap with ground truth."""
    logo = LeaveOneGroupOut()
    scores = []
    for tr_idx, te_idx in logo.split(X, y, groups):
        if y[te_idx].sum() == 0:
            continue
        pos_w = max(1.0, (y[tr_idx] == 0).sum() / max(1, (y[tr_idx] == 1).sum()))
        m = _xgb_model(pos_w)
        m.fit(X[tr_idx], y[tr_idx])
        probs = m.predict_proba(X[te_idx])[:, 1]
        top_n = set(np.argsort(probs)[-n_select:])
        actual = set(np.where(y[te_idx] == 1)[0])
        scores.append(len(top_n & actual) / n_select)
    avg = np.mean(scores) if scores else 0.0
    print(f"  {label} CV avg top-{n_select} overlap: {avg:.2%} over {len(scores)} seasons")
    return avg


def train_allnba(all_stats: pd.DataFrame, rookie_year_df: pd.DataFrame) -> XGBClassifier:
    labels = pd.read_csv(os.path.join(DATA_DIR, "allnba_labels.csv"))
    X, y, meta = build_feature_matrix(all_stats, labels, rookie_year_df)

    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    print(f"  All-NBA class ratio: {pos_weight:.1f}:1")

    groups = meta["SEASON"].values
    evaluate_logo(X, y, groups, n_select=15, label="All-NBA")

    model = _xgb_model(pos_weight)
    model.fit(X, y)
    return model


def train_rookie(all_stats: pd.DataFrame, rookie_year_df: pd.DataFrame) -> XGBClassifier:
    labels = pd.read_csv(os.path.join(DATA_DIR, "allrookie_labels.csv"))

    # Filter stats to rookie rows only
    X_all, y_all, meta_all = build_feature_matrix(all_stats, labels, rookie_year_df)
    mask = meta_all["is_rookie"].values
    X, y, meta = X_all[mask], y_all[mask], meta_all[mask].reset_index(drop=True)

    if len(y) == 0 or y.sum() == 0:
        print("  [warn] No rookie label rows found — check data/raw/allrookie_labels.csv")
        return _xgb_model(30.0)

    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    print(f"  Rookie class ratio: {pos_weight:.1f}:1")

    groups = meta["SEASON"].values
    evaluate_logo(X, y, groups, n_select=10, label="All-Rookie")

    model = _xgb_model(pos_weight)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    print("=== Loading stats ===")
    all_stats = load_all_seasons()
    print(f"  Loaded {len(all_stats)} player-season rows across {all_stats['SEASON'].nunique()} seasons")

    rookie_year_path = os.path.join(DATA_DIR, "players_rookie_year.csv")
    if os.path.exists(rookie_year_path):
        rookie_year_df = pd.read_csv(rookie_year_path)
    else:
        print("  [warn] players_rookie_year.csv not found — rookie model will have no is_rookie info")
        rookie_year_df = None

    print("\n=== Training All-NBA model ===")
    allnba_model = train_allnba(all_stats, rookie_year_df)
    allnba_path = os.path.join(MODELS_DIR, "allnba_model.pkl")
    with open(allnba_path, "wb") as f:
        pickle.dump(allnba_model, f)
    print(f"  Saved → {allnba_path}")

    print("\n=== Training All-Rookie model ===")
    rookie_model = train_rookie(all_stats, rookie_year_df)
    rookie_path = os.path.join(MODELS_DIR, "rookie_model.pkl")
    with open(rookie_path, "wb") as f:
        pickle.dump(rookie_model, f)
    print(f"  Saved → {rookie_path}")

    print("\nTraining complete.")
