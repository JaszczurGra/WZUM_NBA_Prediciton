"""
Collects per-game traditional + advanced player stats and team standings
for every season from 2000-01 through 2025-26 via nba_api.

Also builds data/raw/players_rookie_year.csv which maps player_id → debut season.

Run once; already-cached CSV files are skipped.
"""

import os
import time
import glob
import pandas as pd
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueStandingsV3,
    CommonPlayerInfo,
)
from nba_api.stats.static import players as static_players

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(DATA_DIR, exist_ok=True)

SEASONS = [f"{y}-{str(y + 1)[-2:]:0>2}" for y in range(2000, 2026)]


def _sleep():
    time.sleep(1.2)


def collect_season(season: str) -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, f"player_stats_{season}.csv")
    if os.path.exists(out_path):
        print(f"  [skip] {season} already cached")
        return pd.read_csv(out_path)

    print(f"  Fetching traditional stats for {season}...")
    trad = LeagueDashPlayerStats(
        season=season, per_mode_simple="PerGame"
    ).get_data_frames()[0]
    _sleep()

    print(f"  Fetching advanced stats for {season}...")
    adv = LeagueDashPlayerStats(
        season=season,
        per_mode_simple="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]
    _sleep()

    print(f"  Fetching standings for {season}...")
    try:
        standings = LeagueStandingsV3(season=season).get_data_frames()[0]
        wins = standings[["TeamID", "WINS"]].rename(
            columns={"TeamID": "TEAM_ID", "WINS": "TEAM_WINS"}
        )
    except Exception as e:
        print(f"  [warn] standings failed for {season}: {e}")
        wins = pd.DataFrame(columns=["TEAM_ID", "TEAM_WINS"])
    _sleep()

    # Keep only the advanced columns we need (avoid duplicate column names)
    adv_cols = ["PLAYER_ID"] + [
        c for c in ["NET_RATING", "AST_PCT", "REB_PCT", "USG_PCT", "PIE", "TS_PCT"]
        if c in adv.columns
    ]
    adv_sub = adv[adv_cols]

    df = trad.merge(adv_sub, on="PLAYER_ID", suffixes=("", "_adv"))
    if not wins.empty:
        df = df.merge(wins, on="TEAM_ID", how="left")
    else:
        df["TEAM_WINS"] = 0

    df["SEASON"] = season
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def collect_rookie_years() -> pd.DataFrame:
    """
    Build a player_id → debut_season_year mapping using DraftHistory (fast: 1 API call).
    Falls back to CommonPlayerInfo for undrafted players found in our stats CSVs.
    """
    out_path = os.path.join(DATA_DIR, "players_rookie_year.csv")
    if os.path.exists(out_path):
        print("  [skip] players_rookie_year.csv already cached")
        return pd.read_csv(out_path)

    from nba_api.stats.endpoints import DraftHistory, CommonPlayerInfo

    print("  Fetching DraftHistory (all drafts)...")
    dh = DraftHistory(league_id="00").get_data_frames()[0]
    _sleep()

    # DraftHistory SEASON field is the draft year (e.g., "2024" for 2024 NBA Draft).
    # Draftee joins team in the following season: draft year == debut season start year.
    df = dh[["PERSON_ID", "PLAYER_NAME", "SEASON"]].copy()
    df.rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "DEBUT_YEAR"}, inplace=True)
    df["DEBUT_YEAR"] = df["DEBUT_YEAR"].astype(int)
    df = df.drop_duplicates("PLAYER_ID")

    # Supplement with undrafted players: find player_ids in our stats CSVs not in draft df
    import glob
    stat_files = glob.glob(os.path.join(DATA_DIR, "player_stats_*.csv"))
    if stat_files:
        all_ids = set()
        all_id_name = {}
        for f in stat_files:
            season = os.path.basename(f).replace("player_stats_", "").replace(".csv", "")
            season_year = int(season[:4])
            tmp = pd.read_csv(f, usecols=["PLAYER_ID", "PLAYER_NAME"], low_memory=False)
            for _, row in tmp.iterrows():
                pid = int(row["PLAYER_ID"])
                if pid not in all_id_name:
                    all_id_name[pid] = (row["PLAYER_NAME"], season_year)

        missing_ids = set(all_id_name.keys()) - set(df["PLAYER_ID"].astype(int))
        print(f"  Undrafted / missing players to look up: {len(missing_ids)}")
        extra_rows = []
        for i, pid in enumerate(sorted(missing_ids)):
            if i % 50 == 0 and i > 0:
                print(f"    {i}/{len(missing_ids)}...")
            try:
                info = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
                from_year = int(info["FROM_YEAR"].iloc[0])
                extra_rows.append({"PLAYER_ID": pid,
                                   "PLAYER_NAME": all_id_name[pid][0],
                                   "DEBUT_YEAR": from_year})
            except Exception:
                # Use first season year we saw them as fallback
                extra_rows.append({"PLAYER_ID": pid,
                                   "PLAYER_NAME": all_id_name[pid][0],
                                   "DEBUT_YEAR": all_id_name[pid][1]})
            _sleep()

        if extra_rows:
            df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    print("=== Collecting season stats ===")
    for season in SEASONS:
        collect_season(season)

    print("\n=== Building rookie year index ===")
    collect_rookie_years()

    print("\nDone.")
