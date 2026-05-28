"""
Generate NBA award predictions for the 2025-26 season.

Usage:
    python src/predict.py /path/to/Mikolajczak_Julian.json

Outputs a JSON file with 5 keys (each a list of 5 player name strings):
    all_nba_first_team, all_nba_second_team, all_nba_third_team,
    all_rookie_first_team, all_rookie_second_team

NOTE: Verify that key names match the schema in lab_projekt_2026.ipynb.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Allow running from project root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import build_feature_matrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CURRENT_SEASON = "2025-26"


def load_current_stats() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"player_stats_{CURRENT_SEASON}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data/collect_stats.py first."
        )
    return pd.read_csv(path, low_memory=False)


def load_rookie_year() -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, "players_rookie_year.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def top_names(probs: np.ndarray, meta: pd.DataFrame, start: int, end: int) -> list:
    sorted_idx = np.argsort(probs)[::-1]  # descending
    selected = sorted_idx[start:end]
    return meta.iloc[selected]["PLAYER_NAME"].tolist()


def predict(output_path: str):
    stats = load_current_stats()
    rookie_year_df = load_rookie_year()

    X, _, meta = build_feature_matrix(stats, labels_df=None, rookie_year_df=rookie_year_df)

    # --- All-NBA ---
    allnba_model_path = os.path.join(MODELS_DIR, "allnba_model.pkl")
    if not os.path.exists(allnba_model_path):
        raise FileNotFoundError(f"{allnba_model_path} not found. Run src/train.py first.")
    with open(allnba_model_path, "rb") as f:
        allnba_model = pickle.load(f)

    allnba_probs = allnba_model.predict_proba(X)[:, 1]

    # --- All-Rookie ---
    rookie_model_path = os.path.join(MODELS_DIR, "rookie_model.pkl")
    if not os.path.exists(rookie_model_path):
        raise FileNotFoundError(f"{rookie_model_path} not found. Run src/train.py first.")
    with open(rookie_model_path, "rb") as f:
        rookie_model = pickle.load(f)

    rookie_mask = meta["is_rookie"].values
    rookie_count = rookie_mask.sum()
    print(f"  Rookies in 2025-26: {rookie_count}")

    if rookie_count < 10:
        print(f"  [warn] Only {rookie_count} rookies found — All-Rookie teams may be incomplete")

    rookie_X = X[rookie_mask]
    rookie_meta = meta[rookie_mask].reset_index(drop=True)
    rookie_probs = rookie_model.predict_proba(rookie_X)[:, 1] if len(rookie_X) > 0 else np.array([])

    output = {
        "all_nba_first_team":    top_names(allnba_probs, meta, 0, 5),
        "all_nba_second_team":   top_names(allnba_probs, meta, 5, 10),
        "all_nba_third_team":    top_names(allnba_probs, meta, 10, 15),
        "all_rookie_first_team": top_names(rookie_probs, rookie_meta, 0, 5) if len(rookie_probs) >= 5 else [],
        "all_rookie_second_team":top_names(rookie_probs, rookie_meta, 5, 10) if len(rookie_probs) >= 10 else [],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nPredictions written to: {output_path}")
    for key, players in output.items():
        print(f"  {key}: {players}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <output_path>")
        sys.exit(1)
    predict(sys.argv[1])
