"""
no_class_weight -- baseline XGBoost WITHOUT imbalance handling.

Identical to baseline (depth=5, lr=0.05, 300 trees, monotone) but with
scale_pos_weight = 1 (pos_weight_policy="none"). All-NBA picks are ~3% of
players (~33:1 imbalance), so this isolates how much the dynamic
scale_pos_weight in the baseline is actually worth.

Run:  python experiments/exp_no_class_weight.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xgboost import XGBClassifier
from experiments.common import run_experiment, MONOTONE, SEED

PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05)


def make_model(pos_weight):
    # pos_weight is forced to 1.0 by pos_weight_policy="none"; kept in the
    # signature so the factory matches the interface common.run_experiment expects.
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
        "no_class_weight", make_model, pos_weight_policy="none",
        description="Baseline XGB but scale_pos_weight=1 (no class-imbalance handling).",
        params={**PARAMS, "scale_pos_weight": 1},
    )
