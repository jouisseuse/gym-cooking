"""Plot SVO posterior trajectories from a saved Bag pickle.

Two modes:
  default: stack one panel per (observer, partner) pair so you can see both
           directions of inference at once (each agent's belief about each
           other agent).
  --observer/--partner: plot only that single pair (original behavior).

Usage (from gym_cooking/):

    # Both directions
    python -m misc.metrics.plot_svo_inference \
        --pickle misc/metrics/pickles/<run>.pkl \
        --out ../images_svo/inference_traj.png

    # One direction only
    python -m misc.metrics.plot_svo_inference \
        --pickle misc/metrics/pickles/<run>.pkl \
        --observer agent-1 --partner agent-2 \
        --out ../images_svo/inference_traj.png
"""
import argparse
import math
import os
import sys

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np


def to_deg(x):
    return np.degrees(np.asarray(x))


def parse_args():
    p = argparse.ArgumentParser("plot SVO posterior trajectories")
    p.add_argument("--pickle", required=True, help="Path to Bag pkl from a run.")
    p.add_argument("--observer", default=None,
                   help="Single observer; if omitted, all pairs are plotted.")
    p.add_argument("--partner", default=None,
                   help="Single partner; if omitted, all pairs are plotted.")
    p.add_argument("--out", default=None,
                   help="Output PNG path. If omitted, shows interactively.")
    return p.parse_args()


def plot_pair(ax_main, ax_ess, trace, observer, partner, true_deg):
    """Render one observer→partner posterior trace into the given axes."""
    ts    = np.array([d["t"]   for d in trace])
    means = to_deg([d["mean"]  for d in trace])
    stds  = to_deg([d["std"]   for d in trace])
    esss  = np.array([d["ess"] for d in trace])

    for d in trace:
        xs = np.full_like(d["particles"], d["t"], dtype=float)
        ys = to_deg(d["particles"])
        ax_main.scatter(xs, ys, c="C0", s=3, alpha=0.04, edgecolors="none")

    ax_main.fill_between(ts, means - stds, means + stds, alpha=0.2,
                         label="$\\hat{\\theta} \\pm 1\\sigma$")
    ax_main.plot(ts, means, label="posterior mean", linewidth=2)
    ax_main.axhline(true_deg, ls="--", color="k", alpha=0.7,
                    label="true $\\theta_{{{}}}$ = {:.1f}$^\\circ$".format(
                        partner, true_deg))

    ax_main.set_ylabel("SVO (deg)")
    ax_main.set_ylim(-95, 95)
    ax_main.set_title("{} inferring {}'s SVO".format(observer, partner))
    ax_main.legend(loc="lower right", fontsize=8)
    ax_main.grid(alpha=0.3)

    ax_ess.plot(ts, esss, color="C1")
    ax_ess.set_ylabel("ESS")
    ax_ess.grid(alpha=0.3)


def main():
    args = parse_args()
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    if not data.get("infer_svo", False):
        sys.exit("This run did not enable --infer-svo; no posterior to plot.")

    # Build the list of (observer, partner, trace) to plot.
    pairs = []
    if args.observer and args.partner:
        trace = data["svo_posterior"].get(args.observer, {}).get(args.partner)
        if not trace:
            sys.exit("No SVO trace for observer={} partner={}".format(
                args.observer, args.partner))
        pairs.append((args.observer, args.partner, trace))
    else:
        for observer, by_partner in sorted(data["svo_posterior"].items()):
            for partner, trace in sorted(by_partner.items()):
                if trace:
                    pairs.append((observer, partner, trace))

    if not pairs:
        sys.exit("No PF traces found in this pickle.")

    # Two rows per pair (posterior + ESS), shared x axis.
    n = len(pairs)
    fig, axes = plt.subplots(
            2 * n, 1, figsize=(8, 3.0 * n + 0.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1] * n})
    if n == 1:
        axes = list(axes)

    for i, (observer, partner, trace) in enumerate(pairs):
        ax_main = axes[2 * i]
        ax_ess  = axes[2 * i + 1]
        true_theta = data["true_svo"].get(partner, math.pi / 4)
        plot_pair(ax_main, ax_ess, trace, observer, partner,
                  math.degrees(true_theta))

    axes[-1].set_xlabel("timestep")
    fig.tight_layout()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print("Saved to {}".format(args.out))
    else:
        plt.show()


if __name__ == "__main__":
    main()
