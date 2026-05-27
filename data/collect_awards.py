"""
Scrapes historical All-NBA and All-Rookie award data from Basketball Reference.

Outputs:
  data/raw/allnba_labels.csv   — season, player_name, tier (1/2/3)
  data/raw/allrookie_labels.csv — season, player_name, tier (1/2)

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

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove non-alpha chars for fuzzy matching."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower().strip())


def _season_str_to_key(text: str) -> str:
    """Convert bref season like '2023-24' to nba_api style '2023-24'."""
    return text.strip()


def scrape_allnba() -> pd.DataFrame:
    """
    Basketball Reference All-NBA page layout:
    Each award year has rows like:
      Season | Lg | Tm | Player | ...
    The 'Tm' column cycles 1st/2nd/3rd Team across consecutive rows.
    """
    out_path = os.path.join(DATA_DIR, "allnba_labels.csv")
    if os.path.exists(out_path):
        print("  [skip] allnba_labels.csv already cached")
        return pd.read_csv(out_path)

    url = "https://www.basketball-reference.com/awards/all_nba.html"
    print(f"  GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Find the awards table — bref uses id="awards_all_nba" or similar
    table = soup.find("table", {"id": re.compile(r"awards", re.I)})
    if table is None:
        # Fallback: first large table on the page
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    records = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        texts = [c.get_text(strip=True) for c in cells]

        # Row has a season like "2023-24" in first cell
        season_match = re.match(r"\d{4}-\d{2}", texts[0])
        if not season_match:
            continue

        season = texts[0]
        # Find the player link (anchor tag with /players/ href)
        player_anchor = None
        for c in cells:
            a = c.find("a", href=re.compile(r"/players/"))
            if a:
                player_anchor = a
                break
        if not player_anchor:
            continue

        player_name = player_anchor.get_text(strip=True)

        # Tier: look for text like "1st", "2nd", "3rd" in any cell
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

        records.append({"season": season, "player_name": player_name, "tier": tier})

    df = pd.DataFrame(records)
    if df.empty:
        # Try alternate parsing: table has header row with year spanning multiple data rows
        df = _scrape_award_table_grouped(soup, "all_nba")

    df = df[df["season"].str.match(r"\d{4}-\d{2}$")]
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def scrape_allrookie() -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, "allrookie_labels.csv")
    if os.path.exists(out_path):
        print("  [skip] allrookie_labels.csv already cached")
        return pd.read_csv(out_path)

    url = "https://www.basketball-reference.com/awards/all_rookie.html"
    print(f"  GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

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
        player_anchor = None
        for c in cells:
            a = c.find("a", href=re.compile(r"/players/"))
            if a:
                player_anchor = a
                break
        if not player_anchor:
            continue

        player_name = player_anchor.get_text(strip=True)

        tier = None
        for t in texts:
            if t == "1st":
                tier = 1
            elif t == "2nd":
                tier = 2
        if tier is None:
            continue

        records.append({"season": season, "player_name": player_name, "tier": tier})

    df = pd.DataFrame(records)
    if df.empty:
        df = _scrape_award_table_grouped(soup, "all_rookie")

    df = df[df["season"].str.match(r"\d{4}-\d{2}$")]
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def _scrape_award_table_grouped(soup: BeautifulSoup, award_type: str) -> pd.DataFrame:
    """
    Fallback parser for bref tables where the season column only appears on
    the first row of a group and subsequent rows for the same season are blank.
    """
    records = []
    current_season = None
    current_tier = None

    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame(columns=["season", "player_name", "tier"])

    table = tables[0]

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        texts = [c.get_text(strip=True) for c in cells]

        if re.match(r"\d{4}-\d{2}", texts[0]):
            current_season = texts[0]

        if not current_season:
            continue

        # Detect tier from cell text
        for t in texts:
            if t in ("1st Team", "1st"):
                current_tier = 1
            elif t in ("2nd Team", "2nd"):
                current_tier = 2
            elif t in ("3rd Team", "3rd"):
                current_tier = 3

        for c in cells:
            a = c.find("a", href=re.compile(r"/players/"))
            if a:
                records.append({
                    "season": current_season,
                    "player_name": a.get_text(strip=True),
                    "tier": current_tier,
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=== Scraping All-NBA labels ===")
    df_nba = scrape_allnba()
    print(df_nba.tail())

    print("\n=== Scraping All-Rookie labels ===")
    df_rook = scrape_allrookie()
    print(df_rook.tail())

    print("\nDone.")
