"""Figure style for Remote Sensing of Environment."""

import matplotlib.pyplot as plt

from nbft.constants import FEATURE_LABELS
from nbft.paths import PAPER_FIGS

PALETTE_BINARY = {0: "#d6604d", 1: "#4393c3"}  # unfrozen, frozen
# Three-state palette used by the validation figures
STATE_PALETTE = {"Unfrozen": "#d73027", "Transitional": "#fee090", "Frozen": "#4575b4"}


def set_rse_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
            "axes.labelsize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.linewidth": 1.0,
            "mathtext.fontset": "stix",
            # TrueType (42) instead of Type 3: figure text stays selectable and searchable
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label(feature):
    return FEATURE_LABELS.get(feature, feature)


def savefig(name, formats=("pdf",), fig=None):
    """Save the current (or given) figure to paper/figs/<name>.<fmt> and close it."""
    fig = fig or plt.gcf()
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(PAPER_FIGS / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
