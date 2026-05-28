# NBA Awards Prediction — Solution Description

## Problem Formulation

The task is to predict which NBA players will be named to the All-NBA First, Second, and Third Teams and the All-Rookie First and Second Teams for the 2025-26 season. Each team has exactly 5 players, giving 15 All-NBA selections and 10 All-Rookie selections.

This is treated as a **ranking problem**: train a binary classifier (selected vs. not selected) and rank all players by their predicted probability of being selected. The top-ranked players fill the teams in order.

## Data Sources

**Player statistics** are collected via the `nba_api` library, which provides official NBA stats through the `LeagueDashPlayerStats` endpoint. For each season from 2000-01 through 2025-26, both traditional per-game stats (points, rebounds, assists, steals, blocks, shooting percentages, games played, minutes) and advanced metrics (Net Rating, AST%, REB%, USG%, PIE, TS%) are collected. Team win totals are collected via `LeagueStandingsV3`.

**Historical award labels** (which players were named All-NBA or All-Rookie each season) are scraped from Basketball Reference using `requests` and `BeautifulSoup`, covering the same date range. Player names are normalized to UTF-8 and matched to the stats by an accent-insensitive key, so accented names (e.g. Jokić, Dončić, Šarić) join correctly to their award labels rather than being silently dropped from the positive examples.

All raw data is cached to CSV files in `data/raw/` so the collection step only runs once.

## Feature Engineering

Every statistic is **percentile-ranked within each season** before being used as a feature. This normalization makes the model season-agnostic — a 25 PPG scorer in 2003 (a low-scoring era) gets the same feature value as a 25 PPG scorer in 2023 if they both rank at the same percentile among their peers. Without this, the model would need to learn era-specific thresholds.

The final feature set (17 features) consists of season percentile ranks of:
- **Counting / impact stats**: PTS, REB, AST, STL, BLK, GP, MIN, Net Rating, PIE, Team Wins
- **Efficiency stats**: FG%, 3P%, FT%, TS%, USG%, AST%, REB%

The efficiency stats are percentile-ranked as well (rather than used raw) because league-wide efficiency has drifted upward over the training span — average true-shooting and three-point rates in 2025 are materially higher than in 2000. Ranking within season removes this drift so a future season is comparable to the historical training data, and it puts every feature on a common [0, 1] scale.

## Model

**Algorithm**: `XGBClassifier` (gradient boosted trees) with binary cross-entropy loss.

Gradient boosted trees were chosen because:
1. They handle the extreme class imbalance well via `scale_pos_weight` (~33:1 ratio of non-selected to selected players)
2. They capture non-linear interactions between features (e.g., high usage + high efficiency is multiplicatively better than either alone)
3. They are interpretable via feature importance
4. They satisfy the constraint of not using deep neural networks

The model predicts the probability P(player is All-NBA selected). Players are ranked by this probability; ranks 1–5 become First Team, 6–10 become Second Team, 11–15 become Third Team.

**Monotonic constraints.** Each feature is constrained so that the predicted selection probability is *non-decreasing* in it (`monotone_constraints = (1, …, 1)`). Because every feature is a percentile rank oriented "higher = better", this encodes a single domain-agnostic prior: more production never lowers a player's score. This is standard regularization — it is applied uniformly through the features and does not name, weight, or hand-pick any individual player; the model still does all of the selecting. The constraint is necessary because the unconstrained model learned a spurious *backwards* relationship — high rebound% and assist% (which historically correlate with non-star role-playing bigs) were treated as negative signals, which buried genuinely elite playmaking bigs such as Nikola Jokić. Adding the constraint corrects this without any manual intervention and improves cross-validated accuracy across seasons.

A separate model is trained for All-Rookie on the same features and with the same constraints, but fit only on rows where `is_rookie == True`.

## Training

Training data spans the 2000-01 through 2024-25 seasons (~25 years × ~450 players/year ≈ 11,250 player-season observations). The 2025-26 season is held out entirely for prediction.

**Validation**: Leave-one-season-out cross-validation is used during development. For each held-out season, the model trained on all other seasons predicts rankings, and the metric is the fraction of top-15 predictions that overlap with the actual 15 All-NBA selections. A random baseline would score ~3%, a good model achieves 60%+.

## Prediction

For the 2025-26 season:
1. Current season stats (regular season final) are loaded
2. Features are computed with the same percentile normalization applied within the 2025-26 season
3. The trained All-NBA model scores every player; top 15 by probability fill the three teams
4. The trained All-Rookie model scores only rookie players; top 10 fill the two rookie teams

Rookies are identified by cross-referencing each player's debut season (from `CommonPlayerInfo`) with the current season year.

## Limitations

- The model ranks by overall probability without enforcing position balance. The NBA dropped mandatory position requirements for All-NBA teams in 2023, so this is appropriate for recent seasons but slightly reduces comparability to earlier eras used in training.
- Stats from the regular season only are used; playoff performance is not a factor.
- Player injuries, suspensions, or eligibility rules that may be known to voters but not reflected in stats are not modeled.
- For newly debuted players or players with unusual name spellings, the join between nba_api names and Basketball Reference names may have occasional mismatches.
