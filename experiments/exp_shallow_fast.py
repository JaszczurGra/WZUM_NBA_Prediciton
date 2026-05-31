"""
shallow_fast -- shallow, higher-learning-rate trees.

XGBoost, depth=3, lr=0.10, 200 trees, monotone, dynamic scale_pos_weight.
Tests whether a simpler, faster, lower-variance model keeps up with the
deeper baseline (less risk of overfitting the small positive class).

Run:  python experiments/exp_shallow_fast.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xgboost import XGBClassifier
from experiments.common import run_experiment, MONOTONE, SEED

PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.10)


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
        "shallow_fast", make_model, pos_weight_policy="balanced",
        description="XGB depth=3, lr=0.10, 200 trees, monotone, dynamic scale_pos_weight.",
        params=PARAMS,
    )
