"""
Generate NBA award predictions for the 2025-26 season.

Usage:
    python src/predict.py /path/to/Mikolajczak_Julian.json

Outputs a JSON file with 5 keys (each a list of 5 player name strings),
matching the format specified in Lab_projekt_2026.ipynb (Dane Wyjściowe):
    "first all-nba team", "second all-nba team", "third all-nba team",
    "first rookie all-nba team", "second rookie all-nba team"
"""

import os
import re
import sys
import json
import pickle
import argparse
import unicodedata
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
    if rookie_count < 10:
        print(f"  [warn] Only {rookie_count} rookies found — All-Rookie teams may be incomplete")

    rookie_X = X[rookie_mask]
    rookie_meta = meta[rookie_mask].reset_index(drop=True)
    rookie_probs = rookie_model.predict_proba(rookie_X)[:, 1] if len(rookie_X) > 0 else np.array([])

    output = {
        "first all-nba team":         top_names(allnba_probs, meta, 0, 5),
        "second all-nba team":        top_names(allnba_probs, meta, 5, 10),
        "third all-nba team":         top_names(allnba_probs, meta, 10, 15),
        "first rookie all-nba team":  top_names(rookie_probs, rookie_meta, 0, 5) if len(rookie_probs) >= 5 else [],
        "second rookie all-nba team": top_names(rookie_probs, rookie_meta, 5, 10) if len(rookie_probs) >= 10 else [],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nPredictions written to: {output_path}")
    for key, players in output.items():
        print(f"  {key}: {players}")


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower().strip())









def final_score(prediction_path: str) -> dict:
    """Score the predictions against the actual labels for the latest season."""
    def _score_team(predicted: list, actual_by_tier: dict, pred_tier: int) -> int:
        """Score one predicted team of 5 against the actual tier map."""
        base = 0
        exact = 0
        for player in predicted:
            n = _norm(player)
            actual_tier = next((t for t, names in actual_by_tier.items() if n in names), None)
            if actual_tier is None:
                continue
            diff = abs(pred_tier - actual_tier)
            if diff == 0:
                base += 10
                exact += 1
            elif diff == 1:
                base += 8
            elif diff == 2:
                base += 6
        bonus = [0, 0, 5, 10, 20, 40][exact] if exact <= 5 else 0
        return base + bonus

    with open(prediction_path, "r", encoding="utf-8") as f:
        pred = json.load(f)



    def _tier_map(csv_path: str) -> dict:
        df = pd.read_csv(csv_path)
        tiers = {}
        for _, row in df.iterrows():
            tiers.setdefault(int(row["tier"]), set()).add(_norm(row["player_name"]))
        return tiers



    nba_tiers  = _tier_map(os.path.join(DATA_DIR, "allnba_labels_current.csv"))
    rook_tiers = _tier_map(os.path.join(DATA_DIR, "allrookie_labels_current.csv"))


    teams = {
        "first all-nba team":         (pred.get("first all-nba team",         []), nba_tiers,  1),
        "second all-nba team":        (pred.get("second all-nba team",        []), nba_tiers,  2),
        "third all-nba team":         (pred.get("third all-nba team",         []), nba_tiers,  3),
        "first rookie all-nba team":  (pred.get("first rookie all-nba team",  []), rook_tiers, 1),
        "second rookie all-nba team": (pred.get("second rookie all-nba team", []), rook_tiers, 2),
    }

    scores = {}
    for key, (players, tier_map, pred_tier) in teams.items():
        scores[key] = _score_team(players, tier_map, pred_tier)
    scores["total"] = sum(v for k, v in scores.items() if k != "total")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NBA award predictions.")
    parser.add_argument("output_path", help="Path to write the output JSON file.")
    parser.add_argument("--score", action="store_true", default=False)
    args = parser.parse_args()

    predict(args.output_path)

    if args.score:
        score = final_score(args.output_path)
        print(f"\nFinal Score: {score['total']}/450 (details: {score})")
            
