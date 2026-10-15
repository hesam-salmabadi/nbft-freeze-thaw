"""Decision space of the three-class model: Z_NPR against Z_NDV, one panel per tree-cover class
(Results 4.3).

Background: the most probable class of the production model (model_coefficients.csv) at the median
tree cover of the class's training days, shaded by its probability; lines where two classes are
equally probable. Points: every clean training day of the class, coloured by its in situ label and
drawn in random order so that no label hides another. Z_NDV is on a symmetric log axis (linear
within +-Z_LINEAR), as in the site examples.

Figure: paper/figs/decision_space.png

Usage: python 05_plot_decision_space.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from nbft import config, paths
from nbft.evaluation import weighted_quantile
from nbft.model import CLASSES, TERMS, block_weights, class_probabilities, design
from nbft.plotting import savefig, set_rse_style

COLORS = {"thawed": "#d6604d", "transitional": "#f2c14e", "frozen": "#4575b4"}
LABEL_CLASS = {"thawed": "thawed", "partial": "transitional", "frozen": "frozen"}
TREE_ORDER = ["<10%", "10-30%", "30-60%", ">60%"]
NPR_RANGE, NDV_RANGE = (-8.0, 6.0), (-6.0, 100.0)
Z_LINEAR = 3.0
SEED = 0
POINT_SIZE = 10
POINT_ALPHA = 0.5


def background(ax, coef, tree):
    npr = np.linspace(*NPR_RANGE, 400)
    # grid even in the symlog coordinate so the shading is smooth on the axis
    u = np.linspace(*(symlog(np.array(NDV_RANGE))), 400)
    ndv = inverse_symlog(u)
    gx, gy = np.meshgrid(npr, ndv)
    probs = class_probabilities(coef, design(gx, gy, np.full_like(gx, tree)))
    p = np.stack([np.asarray(probs[k]) for k in CLASSES])
    best = p.argmax(0)
    rgb = np.array([to_rgb(COLORS[k]) for k in CLASSES])[best]
    alpha = 0.06 + 0.3 * (p.max(0) - 1 / 3) / (2 / 3)
    img = np.concatenate([rgb, alpha[..., None]], axis=-1)
    ax.pcolormesh(gx, gy, img[:-1, :-1], shading="flat", rasterized=True)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        diff = p[a] - p[b]
        other = np.delete(p, [a, b], axis=0)[0]
        ax.contour(
            gx,
            gy,
            np.where((p[a] > other) & (p[b] > other), diff, np.nan),
            levels=[0],
            colors="k",
            linewidths=1.0,
        )


def symlog(v):
    return np.sign(v) * np.where(
        np.abs(v) <= Z_LINEAR, np.abs(v), Z_LINEAR * (1 + np.log10(np.abs(v) / Z_LINEAR))
    )


def inverse_symlog(u):
    return np.sign(u) * np.where(
        np.abs(u) <= Z_LINEAR, np.abs(u), Z_LINEAR * 10 ** (np.abs(u) / Z_LINEAR - 1)
    )


def main():
    set_rse_style()
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    coef = coef[["intercept", *TERMS]]
    x = pd.read_parquet(config.path("model.training_table"))
    x = x[x["clean"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, (ax, tc) in enumerate(zip(axes, TREE_ORDER, strict=True)):
        g = x[x["tree_class"] == tc]
        tree = weighted_quantile(g["tree"], block_weights(g["block"]), 0.5)
        background(ax, coef, tree)
        h = g.sample(frac=1, random_state=SEED)  # random drawing order
        ax.scatter(
            h["z_npr"],
            h["z_ndv"],
            s=POINT_SIZE,
            color=h["label"].map(LABEL_CLASS).map(COLORS),
            edgecolors="none",
            alpha=POINT_ALPHA,
            rasterized=True,
        )
        ax.set_yscale("symlog", linthresh=Z_LINEAR)
        ax.set_xlim(*NPR_RANGE)
        ax.set_ylim(*NDV_RANGE)
        ticks = [-3, 0, 3, 10, 30, 100]
        ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_title(f"({'abcd'[i]}) Tree cover {tc} (T = {tree:.2f})", fontsize=14, loc="left")
        if i >= 2:
            ax.set_xlabel("$Z_{NPR}$", fontsize=14)
        ax.tick_params(labelsize=12)
    for ax in axes[::2]:
        ax.set_ylabel("$Z_{NDV}$", fontsize=14)
    handles = [Patch(color=COLORS[k], label=k.capitalize()) for k in CLASSES]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout()
    savefig("decision_space", formats=("png",), fig=fig)


if __name__ == "__main__":
    main()
