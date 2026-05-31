"""
deep_slow -- deeper trees, small learning rate, many rounds.

XGBoost, depth=8, lr=0.02, 800 trees, monotone, dynamic scale_pos_weight.
More capacity to model feature interactions, with a small step size and many
rounds to keep it stable. This is the slowest variant to run.

Run:  python experiments/exp_deep_slow.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xgboost import XGBClassifier
from experiments.common import run_experiment, MONOTONE, SEED

PARAMS = dict(n_estimators=800, max_depth=8, learning_rate=0.02)


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
        "deep_slow", make_model, pos_weight_policy="balanced",
        description="XGB depth=8, lr=0.02, 800 trees, monotone, dynamic scale_pos_weight.",
        params=PARAMS,
    )
