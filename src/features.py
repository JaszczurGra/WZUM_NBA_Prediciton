"""
Feature engineering shared by train.py and predict.py.

build_feature_matrix(stats_df, labels_df=None, rookie_year_df=None)
  → (X, y, meta_df)
"""

import re
import unicodedata
import numpy as np
import pandas as pd


PCTILE_COLS = [
    "PTS", "REB", "AST", "STL", "BLK", "GP", "MIN",
    "NET_RATING", "PIE", "TEAM_WINS",
    "FG_PCT", "FG3_PCT", "FT_PCT", "TS_PCT", "USG_PCT", "AST_PCT", "REB_PCT",
]
RAW_COLS = []
FEATURE_COLS = [f"{c}_pctile" for c in PCTILE_COLS] + RAW_COLS


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower().strip())


def build_feature_matrix(
    stats_df: pd.DataFrame,
    labels_df: pd.DataFrame = None,
    rookie_year_df: pd.DataFrame = None,
) -> tuple:
    """
    combines player stats with award labesls, return feature matrix X, label vector y, and meta for PLAYER_ID, PLAYER_NAME, SEASON, is_rookie

    """
    df = stats_df.copy()

    for col in PCTILE_COLS + RAW_COLS:
        if col not in df.columns:
            df[col] = np.nan

    for col in PCTILE_COLS:
        df[f"{col}_pctile"] = df.groupby("SEASON")[col].rank(pct=True)

    if rookie_year_df is not None:
        df["SEASON_YEAR"] = df["SEASON"].str[:4].astype(int)
        df = df.merge(
            rookie_year_df[["PLAYER_ID", "DEBUT_YEAR"]],
            on="PLAYER_ID",
            how="left",
        )
        df["is_rookie"] = df["SEASON_YEAR"] == df["DEBUT_YEAR"]
    else:
        df["is_rookie"] = False

    X = df[FEATURE_COLS].fillna(0).values.astype(float)

    meta = df[["PLAYER_ID", "PLAYER_NAME", "SEASON", "is_rookie"]].reset_index(drop=True)

    if labels_df is None:
        return X, None, meta

    ldf = labels_df.copy()
    ldf["PLAYER_NAME_NORM"] = ldf["player_name"].apply(normalize_name)
    ldf["SEASON"] = ldf["season"]
    ldf = ldf.drop_duplicates(subset=["SEASON", "PLAYER_NAME_NORM"])

    df["PLAYER_NAME_NORM"] = df["PLAYER_NAME"].apply(normalize_name)
    merged = df.merge(
        ldf[["SEASON", "PLAYER_NAME_NORM", "tier"]],
        on=["SEASON", "PLAYER_NAME_NORM"],
        how="left",
    )
    merged = merged.drop_duplicates(subset=["PLAYER_ID", "SEASON"])
    merged["tier"] = merged["tier"].fillna(0).astype(int)

    y = (merged["tier"] > 0).astype(int).values

    return X, y, meta
