"""Sweep SVO values for the partner and measure behavioral signatures
and posterior recovery.

Run from inside ``gym_cooking/``::

    python -m experiments.run_svo --level partial-divider_salad --seeds 5

Maps to project research questions:

    Q1 (recovery):   true theta_2 vs final posterior mean of theta_2
    Q2 (convergence): posterior std as a function of timestep
    Q3 (behavior):   fraction of (0,0) actions by partner per theta_2

Outputs a pickle per run under ``misc/metrics/pickles/svo/`` and prints a
small summary table at the end.
"""
import argparse
import itertools
import math
import os
import pickle
import subprocess
import sys


PARTNER_SVOS_DEG = [0, 22.5, 45, 67.5, 90]


def parse_args():
    p = argparse.ArgumentParser("SVO experiment sweep")
    p.add_argument("--level", type=str, default="partial-divider_salad")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ego-svo", type=float, default=45.0,
                   help="The inferring agent's own SVO (degrees).")
    p.add_argument("--infer", action="store_true", default=True,
                   help="Enable particle-filter inference of the partner's SVO.")
    p.add_argument("--n-particles", type=int, default=64)
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def main():
    args = parse_args()

    runs = list(itertools.product(PARTNER_SVOS_DEG, range(1, args.seeds + 1)))
    summary = []

    for partner_svo, seed in runs:
        cmd = [
            args.python, "main.py",
            "--num-agents", "2",
            "--level", args.level,
            "--model1", "bd",
            "--model2", "bd",
            "--svo1", str(args.ego_svo),
            "--svo2", str(partner_svo),
            "--seed", str(seed),
        ]
        if args.infer:
            cmd += ["--infer-svo", "--n-particles", str(args.n_particles)]
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=False)
        summary.append((partner_svo, seed))

    print("\nFinished sweep:")
    for partner_svo, seed in summary:
        print("  partner_svo={:>5} deg, seed={}".format(partner_svo, seed))
    print("Pickles in misc/metrics/pickles/. Use a notebook to load and plot.")


if __name__ == "__main__":
    main()
