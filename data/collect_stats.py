"""
Collects per-game traditional + advanced player stats and team standings
for every season from 2000-01 through 2025-26 via nba_api.

Also builds data/raw/players_rookie_year.csv which maps player_id → debut season.

Run once; already-cached CSV files are skipped.
"""

import os
import time
import pandas as pd
import glob
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueStandingsV3,
    DraftHistory,
    CommonPlayerInfo
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(DATA_DIR, exist_ok=True)

SEASONS = [f"{y}-{str(y + 1)[-2:]:0>2}" for y in range(1996, 2026)]


def collect_season(season: str, sleep_length: float = 0.5) -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, f"player_stats_{season}.csv")

    print(f"  Fetching traditional stats for {season}...")
    trad = LeagueDashPlayerStats(
        season=season, per_mode_detailed="PerGame"
    ).get_data_frames()[0]
    time.sleep(sleep_length)

    print(f"  Fetching advanced stats for {season}...")
    adv = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]
    time.sleep(sleep_length)

    print(f"  Fetching standings for {season}...")
    standings = LeagueStandingsV3(season=season).get_data_frames()[0]
    wins = standings[["TeamID", "WINS"]].rename(
        columns={"TeamID": "TEAM_ID", "WINS": "TEAM_WINS"}
    )

    time.sleep(sleep_length)

    #Add data from advaced stats that is not in the traidinal ones 
    adv_sub = adv[ ["PLAYER_ID"] + [  c for c in adv.columns  if c not in trad.columns]]

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
    out_path = os.path.join(DATA_DIR, "players_rookie_year.csv")

    print("  Fetching DraftHistory (all drafts)...")
    dh = DraftHistory(league_id="00").get_data_frames()[0]

    df = dh[["PERSON_ID", "PLAYER_NAME", "SEASON"]].copy()
    df.rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "DEBUT_YEAR"}, inplace=True)
    df["DEBUT_YEAR"] = df["DEBUT_YEAR"].astype(int)
    df = df.drop_duplicates("PLAYER_ID")

    stat_files = glob.glob(os.path.join(DATA_DIR, "player_stats_*.csv"))
    if stat_files:
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
                extra_rows.append({"PLAYER_ID": pid,
                                   "PLAYER_NAME": all_id_name[pid][0],
                                   "DEBUT_YEAR": all_id_name[pid][1]})

        if extra_rows:
            df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    for season in SEASONS:
        collect_season(season)
    collect_rookie_years()
