"""
Run every experiment in sequence. Each is run in its own process and writes
experiments/results/<name>.json. Afterwards, run plot_comparison.py to chart.

Run:  python experiments/run_all.py
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    "exp_baseline.py",
    "exp_shallow_fast.py",
    "exp_deep_slow.py",
    "exp_strong_reg.py",
    "exp_no_class_weight.py",
    "exp_xgb_no_monotone.py",
    "exp_random_forest.py",
]


def main():
    for script in EXPERIMENTS:
        print(f"\n{'#' * 70}\n# {script}\n{'#' * 70}")
        rc = subprocess.run([sys.executable, os.path.join(HERE, script)]).returncode
        if rc != 0:
            print(f"  [warn] {script} exited with code {rc}")
    print("\nAll experiments finished. Now run:  python experiments/plot_comparison.py")


if __name__ == "__main__":
    main()
