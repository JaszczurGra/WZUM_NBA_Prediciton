"""
random_forest -- different model family (bagging instead of boosting).

RandomForestClassifier, 400 trees, class_weight="balanced". A non-boosting tree
ensemble for contrast with the XGBoost variants. Note: random forests do not
support monotone constraints, so this variant trades the baseline's "higher =
better" guarantee for bagged variance reduction.

Run:  python experiments/exp_random_forest.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.ensemble import RandomForestClassifier
from experiments.common import run_experiment, SEED

PARAMS = dict(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
)


def make_model(pos_weight):
    # RandomForest handles imbalance via class_weight, so the resolved
    # pos_weight is intentionally ignored here.
    return RandomForestClassifier(
        **PARAMS,
        random_state=SEED,
        n_jobs=-1,
    )


if __name__ == "__main__":
    run_experiment(
        "random_forest", make_model, pos_weight_policy="balanced",
        description="RandomForest 400 trees, class_weight='balanced' (no monotone constraints).",
        params={**PARAMS, "model": "RandomForestClassifier"},
    )
