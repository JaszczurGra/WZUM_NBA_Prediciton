"""
xgb_no_monotone -- ablation: baseline XGBoost with monotone constraints OFF.

Identical to exp_baseline (depth=5, lr=0.05, 300 trees, dynamic
scale_pos_weight) EXCEPT monotone_constraints is removed. This isolates the
single effect of the "higher = better" constraint.

Reading the chart:
  * if this sits clearly below baseline  -> monotone helps (as the report claims)
  * if it sits between random_forest and baseline -> monotone is a big part of
    why the RandomForest variant (which can't use monotone) is the weakest.

Run:  python experiments/exp_xgb_no_monotone.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xgboost import XGBClassifier
from experiments.common import run_experiment, SEED

PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05)


def make_model(pos_weight):
    # No monotone_constraints argument -> XGBoost is free to learn any
    # direction per feature (the rest matches baseline exactly).
    return XGBClassifier(
        **PARAMS,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


if __name__ == "__main__":
    run_experiment(
        "xgb_no_monotone", make_model, pos_weight_policy="balanced",
        description="Ablation: baseline XGB (depth=5, lr=0.05, 300 trees) with monotone constraints OFF.",
        params=PARAMS,
    )
