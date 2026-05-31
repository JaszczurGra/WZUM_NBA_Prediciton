"""
Scrapes historical All-NBA and All-Rookie award data from Basketball Reference.

Outputs:
  data/raw/allnba_labels.csv          — all seasons except the latest, season, player_name, tier (1/2/3)
  data/raw/allrookie_labels.csv       — all seasons except the latest, season, player_name, tier (1/2)
  data/raw/current_allnba_labels.csv   — latest season only
  data/raw/current_allrookie_labels.csv — latest season only

Run once; cached files are skipped.
"""

import os
import re
import time
import difflib
import unicodedata
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}


def scrape_allnba() -> pd.DataFrame:
    print("Scraping All-NBA labels")
    out_path = os.path.join(DATA_DIR, "allnba_labels.csv")

    url = "https://www.basketball-reference.com/awards/all_league.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, "html.parser")

    table = soup.find("table", {"id": re.compile(r"awards", re.I)})

    records = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        texts = [c.get_text(strip=True) for c in cells]

        season_match = re.match(r"\d{4}-\d{2}", texts[0])
        if not season_match:
            continue

        season = texts[0]
        player_anchors = [
            a for c in cells
            for a in c.find_all("a", href=re.compile(r"/players/"))
        ]
        if not player_anchors:
            continue
        tier = None
        for t in texts:
            if t == "1st":
                tier = 1
            elif t == "2nd":
                tier = 2
            elif t == "3rd":
                tier = 3
        if tier is None:
            continue

        for a in player_anchors:
            records.append({"season": season, "player_name": a.get_text(strip=True), "tier": tier})

    df = pd.DataFrame(records)
    df = df[df["season"].str.match(r"\d{4}-\d{2}$")]

    latest_season = df["season"].max()
    current_df = df[df["season"] == latest_season]
    historical_df = df[df["season"] != latest_season]

    historical_df.to_csv(out_path, index=False)
    current_path = os.path.join(DATA_DIR, "allnba_labels_current.csv")
    current_df.to_csv(current_path, index=False)
    print(f"Saved {len(historical_df)} rows → {out_path}")
    print(f"Saved {len(current_df)} rows (season {latest_season}) → {current_path}")
    return df


def scrape_allrookie() -> pd.DataFrame:
    print("Scraping All-Rookie labels")
    out_path = os.path.join(DATA_DIR, "allrookie_labels.csv")

    url = "https://www.basketball-reference.com/awards/all_rookie.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, "html.parser")

    table = soup.find("table", {"id": re.compile(r"awards", re.I)})
    if table is None:
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    records = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        texts = [c.get_text(strip=True) for c in cells]

        season_match = re.match(r"\d{4}-\d{2}", texts[0])
        if not season_match:
            continue

        season = texts[0]
        player_anchors = [
            a for c in cells
            for a in c.find_all("a", href=re.compile(r"/players/"))
        ]
        if not player_anchors:
            continue

        tier = None
        for t in texts:
            if t == "1st":
                tier = 1
            elif t == "2nd":
                tier = 2
        if tier is None:
            continue

        for a in player_anchors:
            records.append({"season": season, "player_name": a.get_text(strip=True), "tier": tier})

    df = pd.DataFrame(records)
    df = df[df["season"].str.match(r"\d{4}-\d{2}$")]

    latest_season = df["season"].max()
    current_df = df[df["season"] == latest_season]
    historical_df = df[(df["season"] >= '1996-97') & (df["season"] < latest_season)]

    historical_df.to_csv(out_path, index=False)
    current_path = os.path.join(DATA_DIR, "allrookie_labels_current.csv")
    current_df.to_csv(current_path, index=False)
    print(f"Saved {len(historical_df)} rows → {out_path}")
    print(f"Saved {len(current_df)} rows (season {latest_season}) → {current_path}")
    return df



if __name__ == "__main__":
    scrape_allnba()
    scrape_allrookie()
