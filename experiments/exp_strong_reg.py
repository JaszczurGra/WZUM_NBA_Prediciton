"""
strong_reg -- heavily regularized XGBoost.

XGBoost, depth=5, lr=0.05, 500 trees, monotone, dynamic scale_pos_weight, plus
strong regularization: reg_lambda=5, min_child_weight=5, gamma=1.0,
subsample=0.8, colsample_bytree=0.8. Tests whether explicit regularization +
row/column subsampling generalizes better across seasons than the baseline.

Run:  python experiments/exp_strong_reg.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xgboost import XGBClassifier
from experiments.common import run_experiment, MONOTONE, SEED

PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    reg_lambda=5.0,
    min_child_weight=5,
    gamma=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
)


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
        "strong_reg", make_model, pos_weight_policy="balanced",
        description="XGB depth=5, lr=0.05, 500 trees + strong regularization "
                    "(reg_lambda=5, min_child_weight=5, gamma=1, subsample/colsample=0.8).",
        params=PARAMS,
    )
