"""Flowchart of the processing chain (Section 3.1, Fig. workflow).

Two tracks start from the data: the in situ track builds the ground-state labels, the satellite
track the standardized indices. They meet in feature selection and the model, whose maps and
held-out predictions go to the evaluation. Box titles carry the section numbers of the paper.

Usage: python 03_plot_workflow.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nbft.plotting import savefig, set_rse_style

INK, MUTED = "#2b2b2b", "#6b6a66"
# One hue per track; light fill, full-strength edge
TRACKS = {
    "insitu": ("#eb6834", "#fdeee6"),
    "satellite": ("#2a78d6", "#e7f0fb"),
    "static": ("#8a8984", "#f1f0ec"),
    "model": ("#2b2b2b", "#ffffff"),
    "evaluation": ("#1baf7a", "#e5f6ef"),
}
W, H = 30.0, 11.0  # default box size (axis units)


def box(ax, x, y, title, body, track, w=W, h=H):
    edge, fill = TRACKS[track]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.6 if track == "model" else 1.2,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        x,
        y + h / 2 - 2.0,
        title,
        ha="center",
        va="top",
        fontsize=12.5,
        weight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x,
        y + h / 2 - 5.0,
        body,
        ha="center",
        va="top",
        fontsize=10.5,
        color=INK,
        linespacing=1.35,
        zorder=3,
    )
    return {
        "top": (x, y + h / 2 + 0.4),
        "bottom": (x, y - h / 2 - 0.4),
        "left": (x - w / 2 - 0.4, y),
        "right": (x + w / 2 + 0.4, y),
    }


def arrow(ax, start, end, label=None, rad=0.0, label_xy=None, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>,head_length=6,head_width=3.5",
            connectionstyle=f"arc3,rad={rad}",
            color=MUTED,
            linewidth=1.3,
            linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            color=MUTED,
            style="italic",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            zorder=4,
        )


def main():
    set_rse_style()
    fig, ax = plt.subplots(figsize=(12.5, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Data (Section 2)
    insitu = box(
        ax,
        18,
        92,
        "In situ soil (2.2)",
        "temperature and moisture\nISMN, AmeriFlux, field networks",
        "insitu",
        h=9.5,
    )
    tb = box(
        ax,
        52,
        92,
        "Brightness temperatures (2.3)",
        "SMAP L-band V, H\nAMSR2 18.7 and 36.5 GHz V, H",
        "satellite",
        h=9.5,
    )
    static = box(
        ax,
        86,
        92,
        "Static maps (2.5)",
        "tree cover, roughness,\nopen water, biome",
        "static",
        w=24,
        h=9.5,
    )

    # Methods (Section 3)
    labels = box(
        ax,
        18,
        70,
        "Ground-state labels (3.2)",
        "freezing curve of each sensor\n"
        r"$\rightarrow$ thresholds $T_{on}$ and $T_{fr}$"
        "\nstate of each cell at the morning\noverpass: thawed, partial, frozen",
        "insitu",
        h=14,
    )
    indices = box(
        ax,
        52,
        70,
        "Indices and standardization (3.3)",
        "NPR, NDV, NDH, WD\n"
        r"$Z = (X - \mu_{thaw}) \, / \, \sigma_{Jul\text{-}Aug}$ per pixel",
        "satellite",
        h=13,
    )
    select = box(
        ax,
        52,
        49,
        "Feature selection (3.4)",
        "rank by frozen/thawed separation\n"
        r"drop an index if $|\rho| > 0.7$ with a kept one",
        "satellite",
        h=11,
    )
    model = box(
        ax,
        35,
        27,
        "Three-class logistic model (3.5)",
        r"selected indices, tree cover $T$, $T \times$ index"
        "\nequal weight per spatial block and class\n"
        "five-fold cross-validation by 200 km blocks",
        "model",
        w=36,
        h=13,
    )
    eval_inputs = box(
        ax,
        86,
        27,
        "Evaluation data (2.4, 2.5)",
        "FT-SMOS\nERA5-Land air temperature and SWE\nIMS snow cover",
        "static",
        w=26,
        h=12,
    )

    # Products and evaluation
    maps = box(
        ax,
        18,
        6,
        "Daily morning maps, 2015–2023",
        "P(thawed), P(transitional), P(frozen)\nmost probable state",
        "model",
        w=30,
        h=10,
    )
    evaluation = box(
        ax,
        70,
        7,
        "Evaluation (3.6)",
        "held-out labels by tree cover, roughness, biome\n"
        "FT-SMOS, soil and air temperature, snow\nexternal test outside North America",
        "evaluation",
        w=40,
        h=12,
    )

    arrow(ax, insitu["bottom"], labels["top"])
    arrow(ax, tb["bottom"], indices["top"])
    arrow(ax, indices["bottom"], select["top"])
    arrow(
        ax, (28, labels["bottom"][1]), select["left"], rad=0.25, label="labels", label_xy=(32.5, 55)
    )
    arrow(
        ax,
        labels["bottom"],
        (model["top"][0] - 9, model["top"][1]),
        rad=0.15,
        label="training labels",
        label_xy=(21.5, 41),
    )
    arrow(ax, select["bottom"], (model["top"][0] + 9, model["top"][1]), rad=-0.15)
    arrow(ax, static["bottom"], model["right"], rad=-0.25, label="tree cover", label_xy=(77, 52))
    arrow(ax, model["bottom"], (maps["top"][0] + 6, maps["top"][1]), rad=0.15)
    arrow(
        ax,
        (model["right"][0] - 3, model["bottom"][1] + 0.4),
        evaluation["top"],
        rad=-0.15,
        label="held-out predictions",
        label_xy=(63, 17),
    )
    arrow(ax, eval_inputs["bottom"], (evaluation["top"][0] + 14, evaluation["top"][1]), rad=0.0)

    savefig("workflow", fig=fig)


if __name__ == "__main__":
    main()
