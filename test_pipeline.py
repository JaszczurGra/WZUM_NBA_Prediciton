"""
Smoke test: generates synthetic data, trains, predicts, checks output.
Run with: python test_pipeline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────
# Player name helpers (alphabetic only so normalize_name works)
# ──────────────────────────────────────────────────────────────
FIRST = [chr(65 + i) * 4 for i in range(26)]   # "Aaaa" … "Zzzz"
LAST  = [chr(65 + i) * 5 for i in range(26)]   # "Aaaaa" … "Zzzzz"

def pname(i):
    """Unique name for player index i (0-675 range)."""
    return f"{FIRST[i // 26]} {LAST[i % 26]}"

def rname(i):
    """Unique rookie name for player index i."""
    return f"Rook{FIRST[i // 26]} {LAST[i % 26]}"

# Layout per season (400 players total):
#   indices 0–59   : veterans / regulars
#   indices 60–74  : All-NBA stars (non-rookie)
#   indices 75–399 : rest of roster
# Rookie pool: separate 60 rookie players per season, indices 0–59 in rookie namespace

SEASONS = [f"{y}-{str(y+1)[-2:]:0>2}" for y in range(2000, 2026)]
PLAYERS_PER_SEASON = 200
ROOKIES_PER_SEASON = 50
ALLNBA_STAR_START  = 60   # All-NBA stars occupy indices 60-74
ALLROOK_STAR_START = 0    # All-Rookie stars occupy rookie indices 0-9

rng = np.random.default_rng(42)

stat_rows = []
debut_records = []

for season in SEASONS:
    season_year = int(season[:4])

    # Veteran / regular players (non-rookies)
    for i in range(PLAYERS_PER_SEASON):
        pid = season_year * 100000 + i
        star = ALLNBA_STAR_START <= i < ALLNBA_STAR_START + 15
        pts  = rng.normal(28 if star else 10, 2)
        reb  = rng.normal(9  if star else 4, 1)
        ast  = rng.normal(7  if star else 2, 1)
        stl  = rng.normal(1.5 if star else 0.8, 0.2)
        blk  = rng.normal(1.2 if star else 0.4, 0.2)
        gp   = rng.integers(65, 82) if star else rng.integers(20, 82)
        mins = rng.normal(34 if star else 20, 3)
        stat_rows.append({
            "PLAYER_ID": pid, "PLAYER_NAME": pname(i), "TEAM_ID": rng.integers(1, 31),
            "GP": int(gp), "MIN": round(float(mins), 1),
            "PTS": round(float(pts), 1), "REB": round(float(reb), 1),
            "AST": round(float(ast), 1), "STL": round(float(stl), 1),
            "BLK": round(float(blk), 1),
            "FG_PCT":  round(float(rng.normal(0.52 if star else 0.44, 0.04)), 3),
            "FG3_PCT": round(float(rng.normal(0.38 if star else 0.33, 0.05)), 3),
            "FT_PCT":  round(float(rng.normal(0.85 if star else 0.72, 0.06)), 3),
            "TS_PCT":  round(float(rng.normal(0.60 if star else 0.50, 0.04)), 3),
            "USG_PCT": round(float(rng.normal(0.30 if star else 0.18, 0.04)), 3),
            "NET_RATING": round(float(rng.normal(8 if star else 0, 3)), 1),
            "AST_PCT":    round(float(rng.normal(0.35 if star else 0.15, 0.07)), 3),
            "REB_PCT":    round(float(rng.normal(0.12 if star else 0.07, 0.02)), 3),
            "PIE":        round(float(rng.normal(0.15 if star else 0.07, 0.03)), 3),
            "TEAM_WINS":  int(rng.integers(40, 65) if star else rng.integers(25, 60)),
            "SEASON": season,
        })

    # Rookie players (separate player IDs)
    for j in range(ROOKIES_PER_SEASON):
        pid = season_year * 100000 + PLAYERS_PER_SEASON + j
        star = j < 10
        debut_records.append({"PLAYER_ID": pid, "PLAYER_NAME": rname(j), "DEBUT_YEAR": season_year})
        pts  = rng.normal(20 if star else 8, 2)
        reb  = rng.normal(7  if star else 3, 1)
        ast  = rng.normal(5  if star else 2, 1)
        gp   = rng.integers(50, 82) if star else rng.integers(10, 82)
        mins = rng.normal(28 if star else 16, 4)
        stat_rows.append({
            "PLAYER_ID": pid, "PLAYER_NAME": rname(j), "TEAM_ID": rng.integers(1, 31),
            "GP": int(gp), "MIN": round(float(mins), 1),
            "PTS": round(float(pts), 1), "REB": round(float(reb), 1),
            "AST": round(float(ast), 1),
            "STL": round(float(rng.normal(1.2 if star else 0.7, 0.2)), 1),
            "BLK": round(float(rng.normal(0.8 if star else 0.3, 0.2)), 1),
            "FG_PCT":  round(float(rng.normal(0.50 if star else 0.42, 0.05)), 3),
            "FG3_PCT": round(float(rng.normal(0.36 if star else 0.31, 0.05)), 3),
            "FT_PCT":  round(float(rng.normal(0.82 if star else 0.70, 0.07)), 3),
            "TS_PCT":  round(float(rng.normal(0.57 if star else 0.48, 0.05)), 3),
            "USG_PCT": round(float(rng.normal(0.26 if star else 0.16, 0.04)), 3),
            "NET_RATING": round(float(rng.normal(5 if star else -1, 3)), 1),
            "AST_PCT":    round(float(rng.normal(0.25 if star else 0.12, 0.06)), 3),
            "REB_PCT":    round(float(rng.normal(0.10 if star else 0.06, 0.02)), 3),
            "PIE":        round(float(rng.normal(0.12 if star else 0.06, 0.03)), 3),
            "TEAM_WINS":  int(rng.integers(30, 60)),
            "SEASON": season,
        })

all_stats = pd.DataFrame(stat_rows)
# Clip numeric columns only
num_cols = all_stats.select_dtypes(include="number").columns
all_stats[num_cols] = all_stats[num_cols].clip(lower=0)

# All-NBA labels: players at indices 60-74 in non-rookie pool
allnba_rows = []
for season in SEASONS[:-1]:
    for i in range(ALLNBA_STAR_START, ALLNBA_STAR_START + 15):
        tier = 1 if i < ALLNBA_STAR_START + 5 else (2 if i < ALLNBA_STAR_START + 10 else 3)
        allnba_rows.append({"season": season, "player_name": pname(i), "tier": tier})
allnba_labels = pd.DataFrame(allnba_rows)

# All-Rookie labels: rookie players at indices 0-9 in rookie pool
allrookie_rows = []
for season in SEASONS[:-1]:
    for j in range(10):
        allrookie_rows.append({"season": season, "player_name": rname(j), "tier": 1 if j < 5 else 2})
allrookie_labels = pd.DataFrame(allrookie_rows)

debut_df = pd.DataFrame(debut_records).drop_duplicates("PLAYER_ID")

# ──────────────────────────────────────────────────────────────
# 2. Write synthetic CSVs to a temp dir
# ──────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    raw_dir = os.path.join(tmpdir, "data", "raw")
    os.makedirs(raw_dir)
    models_dir = os.path.join(tmpdir, "models")
    os.makedirs(models_dir)

    for season in SEASONS:
        df = all_stats[all_stats["SEASON"] == season]
        df.to_csv(os.path.join(raw_dir, f"player_stats_{season}.csv"), index=False)

    allnba_labels.to_csv(os.path.join(raw_dir, "allnba_labels.csv"), index=False)
    allrookie_labels.to_csv(os.path.join(raw_dir, "allrookie_labels.csv"), index=False)
    debut_df.to_csv(os.path.join(raw_dir, "players_rookie_year.csv"), index=False)

    # ──────────────────────────────────────────────────────────────
    # 3. Train
    # ──────────────────────────────────────────────────────────────
    import glob as _glob
    from xgboost import XGBClassifier
    import src.features as feat_mod

    def load_seasons():
        files = sorted(_glob.glob(os.path.join(raw_dir, "player_stats_*.csv")))
        return pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)

    def quick_xgb(pos_w):
        return XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )

    print("Loading stats...")
    all_s = load_seasons()
    rook_df = pd.read_csv(os.path.join(raw_dir, "players_rookie_year.csv"))
    nba_lbl  = pd.read_csv(os.path.join(raw_dir, "allnba_labels.csv"))
    rook_lbl = pd.read_csv(os.path.join(raw_dir, "allrookie_labels.csv"))

    print("Building All-NBA features...")
    X, y, meta = feat_mod.build_feature_matrix(all_s, nba_lbl, rook_df)
    print(f"  samples={len(y)}, positive={y.sum()}")
    assert y.sum() > 0, f"No positive All-NBA labels! Check name join. Sample names: {meta['PLAYER_NAME'].head()}"

    pos_w = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    print(f"  pos_weight={pos_w:.1f}")
    # Train on first 20 seasons
    mask = meta["SEASON"].isin(SEASONS[:20]).values
    allnba_model = quick_xgb(pos_w)
    allnba_model.fit(X[mask], y[mask])

    print("Building All-Rookie features...")
    Xr, yr, metar = feat_mod.build_feature_matrix(all_s, rook_lbl, rook_df)
    rook_mask = metar["is_rookie"].values
    Xr2, yr2, metar2 = Xr[rook_mask], yr[rook_mask], metar[rook_mask].reset_index(drop=True)
    print(f"  rookie samples={len(yr2)}, positive={yr2.sum()}")
    assert yr2.sum() > 0, "No positive All-Rookie labels!"

    pos_wr = max(1.0, (yr2 == 0).sum() / max(1, (yr2 == 1).sum()))
    mask_r = metar2["SEASON"].isin(SEASONS[:20]).values
    rookie_model = quick_xgb(pos_wr)
    rookie_model.fit(Xr2[mask_r], yr2[mask_r])

    with open(os.path.join(models_dir, "allnba_model.pkl"), "wb") as f:
        pickle.dump(allnba_model, f)
    with open(os.path.join(models_dir, "rookie_model.pkl"), "wb") as f:
        pickle.dump(rookie_model, f)

    # ──────────────────────────────────────────────────────────────
    # 4. Predict on 2025-26
    # ──────────────────────────────────────────────────────────────
    print("\nPredicting 2025-26...")
    cur = all_s[all_s["SEASON"] == "2025-26"].copy()
    Xc, _, mc = feat_mod.build_feature_matrix(cur, labels_df=None, rookie_year_df=rook_df)

    allnba_probs = allnba_model.predict_proba(Xc)[:, 1]

    rook_mask_c = mc["is_rookie"].values
    rook_Xc = Xc[rook_mask_c]
    rook_mc  = mc[rook_mask_c].reset_index(drop=True)
    rook_probs = rookie_model.predict_proba(rook_Xc)[:, 1] if len(rook_Xc) >= 10 else np.zeros(len(rook_Xc))

    def top_names(probs, meta_df, start, end):
        idx = np.argsort(probs)[::-1][start:end]
        return meta_df.iloc[idx]["PLAYER_NAME"].tolist()

    output = {
        "all_nba_first_team":    top_names(allnba_probs, mc, 0, 5),
        "all_nba_second_team":   top_names(allnba_probs, mc, 5, 10),
        "all_nba_third_team":    top_names(allnba_probs, mc, 10, 15),
        "all_rookie_first_team": top_names(rook_probs, rook_mc, 0, 5),
        "all_rookie_second_team":top_names(rook_probs, rook_mc, 5, 10),
    }

    out_path = os.path.join(tmpdir, "Mikolajczak_Julian.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ──────────────────────────────────────────────────────────────
    # 5. Validate
    # ──────────────────────────────────────────────────────────────
    print("\n=== Validation ===")
    with open(out_path) as f:
        result = json.load(f)

    expected_keys = ["all_nba_first_team", "all_nba_second_team", "all_nba_third_team",
                     "all_rookie_first_team", "all_rookie_second_team"]
    errors = []
    for key in expected_keys:
        if key not in result:
            errors.append(f"MISSING KEY: {key}")
        elif len(result[key]) != 5:
            errors.append(f"WRONG COUNT for {key}: {len(result[key])}")
        else:
            print(f"  {key}: {result[key]}")

    all_players = [p for lst in result.values() for p in lst]
    if len(all_players) != len(set(all_players)):
        errors.append("DUPLICATE players across teams!")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
