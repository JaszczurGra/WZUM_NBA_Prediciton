"""
baseline -- the current production model.

XGBoost, depth=5, lr=0.05, 300 trees, monotone constraints, dynamic
scale_pos_weight. This is the reference point every other variant is compared
against (it reproduces the configuration in src/train.py).

Run:  python experiments/exp_baseline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xgboost import XGBClassifier
from experiments.common import run_experiment, MONOTONE, SEED

PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05)


def make_model(pos_weight):
    return XGBClassifier(
        **PARAMS,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        monotone_constraints=MONOTONE,
    )


if __name__ == "__main__":
    run_experiment(
        "baseline", make_model, pos_weight_policy="balanced",
        description="Production config: XGB depth=5, lr=0.05, 300 trees, monotone, dynamic scale_pos_weight.",
        params=PARAMS,
    )
