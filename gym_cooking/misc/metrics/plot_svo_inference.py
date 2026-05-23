"""Plot the partner-SVO posterior trajectory from a saved Bag pickle.

Usage (from gym_cooking/):

    python -m misc.metrics.plot_svo_inference \
        --pickle misc/metrics/pickles/<run>.pkl \
        --observer agent-1 --partner agent-2 \
        --out ../images_svo/inference_traj.png

The plot shows:
  - posterior mean theta_hat over time (degrees)
  - +/- 1 std envelope
  - true theta_partner as a horizontal dashed line
  - particle cloud as faint scatter

Requires matplotlib and numpy.
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
    p = argparse.ArgumentParser("plot SVO posterior trajectory")
    p.add_argument("--pickle", required=True, help="Path to Bag pkl from a run.")
    p.add_argument("--observer", default="agent-1",
                   help="Inferring agent whose posterior to plot.")
    p.add_argument("--partner", default="agent-2",
                   help="Partner whose theta is being inferred.")
    p.add_argument("--out", default=None,
                   help="Output PNG path. If omitted, shows interactively.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    if not data.get("infer_svo", False):
        sys.exit("This run did not enable --infer-svo; no posterior to plot.")

    posterior_by_partner = data["svo_posterior"].get(args.observer, {})
    trace = posterior_by_partner.get(args.partner)
    if not trace:
        sys.exit("No SVO trace for observer={} partner={}".format(
            args.observer, args.partner))

    ts      = np.array([d["t"]    for d in trace])
    means   = to_deg([d["mean"]   for d in trace])
    stds    = to_deg([d["std"]    for d in trace])
    esss    = np.array([d["ess"]  for d in trace])

    true_partner_theta = data["true_svo"].get(args.partner, math.pi / 4)
    true_deg = math.degrees(true_partner_theta)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # Particle cloud (faint scatter).
    for d in trace:
        xs = np.full_like(d["particles"], d["t"], dtype=float)
        ys = to_deg(d["particles"])
        ax1.scatter(xs, ys, c="C0", s=4, alpha=0.05, edgecolors="none")

    # Posterior mean +/- 1 std.
    ax1.fill_between(ts, means - stds, means + stds, alpha=0.2,
                     label="$\\hat{\\theta} \\pm 1\\sigma$")
    ax1.plot(ts, means, label="posterior mean", linewidth=2)
    ax1.axhline(true_deg, ls="--", color="k", alpha=0.7,
                label="true $\\theta_{{{}}}$ = {:.1f}$^\\circ$".format(
                    args.partner, true_deg))

    ax1.set_ylabel("partner SVO (deg)")
    ax1.set_ylim(-95, 95)
    ax1.set_title("{} inferring {}'s SVO".format(args.observer, args.partner))
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    # ESS panel.
    ax2.plot(ts, esss, color="C1")
    ax2.set_ylabel("ESS")
    ax2.set_xlabel("timestep")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print("Saved to {}".format(args.out))
    else:
        plt.show()


if __name__ == "__main__":
    main()
