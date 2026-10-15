"""Probability of frozen ground of the two-state HMM after Walker et al. (2022) and of the
three-class model on the thawed, partial and frozen days of the development cells north of 45 N
(Supplementary Material).

Same days as 03_model/18_hmm_baseline.py: clean labelled days whose HMM inputs were observed,
with the held-out P(frozen) of the model (cv_predictions). Histograms of the share of days per
probability bin, every day weighted so that each spatial block carries the same total weight
within the set scored (model.block_weights).

Figure: paper/figs/hmm_probabilities.pdf

Usage: python 11_plot_hmm.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbft import config
from nbft.model import block_weights
from nbft.plotting import savefig, set_rse_style

WALKER_MIN_LATITUDE = 45.0
BINS = np.linspace(0, 1, 21)
PANELS = {"thawed": "Thawed days", "partial": "Partial days", "frozen": "Frozen days"}
RETRIEVALS = {
    "hmm_prob": ("Two-state HMM", "#7b3294"),
    "p_frozen": ("Three-class model", "#2b2b2b"),
}


def main():
    set_rse_style()
    pred = pd.read_parquet(config.path("model.cv_predictions"))
    hmm = pd.read_parquet(config.path("baseline.hmm_posteriors", pixels="insitu", am_pm="am"))
    x = pred.merge(hmm[["cell_id", "date", "hmm_prob", "observed"]], on=["cell_id", "date"])
    x = x[x["observed"] & (x["latitude"] >= WALKER_MIN_LATITUDE)]
    x = x.dropna(subset=["hmm_prob", "p_frozen"]).reset_index(drop=True)
    w = pd.Series(block_weights(x["block"]), index=x.index)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for i, (ax, (label, title)) in enumerate(zip(axes, PANELS.items(), strict=True)):
        m = x["label"] == label
        for col, (name, color) in RETRIEVALS.items():
            h, _ = np.histogram(x.loc[m, col].clip(0, 1), BINS, weights=w[m])
            ax.stairs(h / h.sum(), BINS, color=color, lw=2, label=name)
        ax.set_title(f"({'abc'[i]}) {title}", loc="left", fontsize=13)
        ax.set_xlabel("$P$(frozen)", fontsize=12)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("Share of days", fontsize=12)
    axes[0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    savefig("hmm_probabilities", fig=fig)


if __name__ == "__main__":
    main()
