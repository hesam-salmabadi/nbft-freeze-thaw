"""Are the development cells typical of the North American landscape?

Compares the development cells with every North American land cell (open water at most 15 %,
ESA CCI) on tree cover, terrain roughness, elevation and slope (two-sample Kolmogorov-Smirnov D,
medians, and
the share of the domain inside the range the development cells span), and on ESA CCI class,
Koppen-Geiger main group and RESOLVE biome (shares). Complements the radiometric
representativeness test (K-S, Wasserstein, MMD on Z_NPR / Z_NDV).

Usage: python 07_representativeness.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nbft import config, insitu, paths
from nbft.plotting import savefig, set_rse_style

CONTINUOUS = {
    "tree_cover_pct": ("treecover", "mean", "Tree cover (%)"),
    "elevation_std_m": ("terrain", "elevation_std", "Roughness, elevation std (m)"),
    "elevation_m": ("terrain", "elevation_mean", "Elevation (m)"),
    "slope_deg": ("terrain", "slope_mean", "Slope (deg)"),
}
CATEGORICAL = {
    "cci_class": (
        "landcover_cci2015",
        "dominant_class",
        {1: "forest", 2: "low_vegetation", 3: "wetland", 4: "agriculture", 5: "water", 6: "other"},
    ),
    "koppen_group": ("koppen", "dominant_class", None),
    "biome": ("resolve_biome", "dominant_class", None),
}
KOPPEN = [
    "Af",
    "Am",
    "Aw",
    "BWh",
    "BWk",
    "BSh",
    "BSk",
    "Csa",
    "Csb",
    "Csc",
    "Cwa",
    "Cwb",
    "Cwc",
    "Cfa",
    "Cfb",
    "Cfc",
    "Dsa",
    "Dsb",
    "Dsc",
    "Dsd",
    "Dwa",
    "Dwb",
    "Dwc",
    "Dwd",
    "Dfa",
    "Dfb",
    "Dfc",
    "Dfd",
    "ET",
    "EF",
]
DOMAIN_COLOR, DEV_COLOR = "#8a8984", "#2a78d6"


def domain_mask():
    """North American land cells with at most 15 % open water (ESA CCI, as in the screen)."""
    return insitu.north_america_mask() & (insitu.cci_water_fraction() <= insitu.MAX_WATER_FRACTION)


def code_names(name):
    import rasterio

    with rasterio.open(config.path("sfcc.landcover_raster", name=name, grid=insitu.GRID)) as src:
        tag = src.tags().get("codes") or src.tags().get("classes")
    return {int(k): v for k, v in (p.split("=", 1) for p in tag.split(";"))}


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cell_table.csv")
    groups = {
        "development": cells[cells["set"] == "development"],
    }
    mask = domain_mask()

    rows = []
    domain_values = {}
    for col, (raster, band, _) in CONTINUOUS.items():
        grid = insitu.read_band(raster, band).astype(float)
        dom = grid[mask]
        dom = dom[np.isfinite(dom)]
        domain_values[col] = dom
        for name, g in groups.items():
            r, c = insitu.cell_rowcol(g["cell_id"])
            v = grid[r, c]
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, [5, 95])
            rows.append(
                {
                    "variable": col,
                    "group": name,
                    "n_cells": len(v),
                    "median": np.median(v),
                    "domain_median": np.median(dom),
                    "ks_d": stats.ks_2samp(v, dom).statistic,
                    "domain_within_p5_p95": np.mean((dom >= lo) & (dom <= hi)),
                }
            )
    pd.DataFrame(rows).round(3).to_csv(paths.TABLES / "insitu_representativeness.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    shares = []
    for col, (raster, band, names) in CATEGORICAL.items():
        grid = insitu.read_band(raster, band)
        names = names or code_names(raster)
        label = (
            (lambda k, names=names: names.get(int(k), "none"))
            if col != "koppen_group"
            else (lambda k: KOPPEN[int(k) - 1][:2] if k > 0 else "none")
        )
        dom = pd.Series(grid[mask]).map(label).value_counts(normalize=True).rename("domain")
        parts = [dom]
        for name, g in groups.items():
            r, c = insitu.cell_rowcol(g["cell_id"])
            parts.append(pd.Series(grid[r, c]).map(label).value_counts(normalize=True).rename(name))
        t = pd.concat(parts, axis=1).fillna(0).round(3).assign(attribute=col)
        shares.append(t.rename_axis("class").reset_index())
    shares = pd.concat(shares)
    shares.to_csv(paths.TABLES / "insitu_representativeness_classes.csv", index=False)
    print(shares.to_string(index=False))

    figure(cells, groups["development"], domain_values, shares)


LC_NAMES = {
    "forest": "Forest",
    "low_vegetation": "Low vegetation",
    "agriculture": "Agriculture",
    "wetland": "Wetland",
    "other": "Other",
}
KG_NAMES = {"B": "B (arid)", "C": "C (temperate)", "D": "D (cold)", "E": "E (polar)"}


def paired_bars(ax, table, names, title):
    """Share of development cells vs North American land per class (horizontal bars)."""
    t = table.set_index("class").reindex(list(names)).fillna(0)
    y = np.arange(len(t))[::-1]
    h = 0.38
    ax.barh(
        y + h / 2, t["domain"] * 100, h, color=DOMAIN_COLOR, alpha=0.6, label="North American land"
    )
    ax.barh(y - h / 2, t["development"] * 100, h, color=DEV_COLOR, label="Development cells")
    ax.set_yticks(y, [names[k] for k in t.index])
    ax.set_xlabel("Share of cells (%)")
    ax.set_title(title, loc="left", fontsize=15)


def figure(cells, dev, domain_values, shares):
    set_rse_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    panels = [
        ("tree_cover_pct", "(a)"),
        ("elevation_std_m", "(b)"),
        ("elevation_m", "(c)"),
    ]
    for ax, (col, tag) in zip(axes[0], panels, strict=True):
        dom, v = domain_values[col], dev[col].dropna().to_numpy()
        log = col == "elevation_std_m"
        edges = (
            np.logspace(0, np.log10(max(dom.max(), 1) + 1), 30)
            if log
            else np.linspace(0, max(np.percentile(dom, 99.5), np.percentile(v, 99.5)), 30)
        )
        # on the log axis, weight by bin width in log space so equal areas mean equal shares
        width = np.diff(np.log10(edges)) if log else np.diff(edges)
        for data, kw in [
            (dom, dict(color=DOMAIN_COLOR, alpha=0.45, label="North American land")),
            (v, dict(histtype="step", linewidth=2, color=DEV_COLOR, label="Development cells")),
        ]:
            counts, _ = np.histogram(np.clip(data, edges[0], edges[-1]), edges)
            ax.hist(edges[:-1], edges, weights=counts / counts.sum() / width, **kw)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(CONTINUOUS[col][2])
        ax.set_yticks([])
        ax.set_title(tag, loc="left", fontsize=15)
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=12, frameon=False)

    lc = shares[shares["attribute"] == "cci_class"]
    paired_bars(axes[1, 0], lc, LC_NAMES, "(d) Land cover (ESA CCI)")
    kg = shares[shares["attribute"] == "koppen_group"].copy()
    kg["class"] = kg["class"].str[0]
    kg = kg.groupby("class", as_index=False)[["domain", "development"]].sum()
    paired_bars(axes[1, 1], kg, KG_NAMES, "(e) K\u00f6ppen-Geiger main group")
    axes[1, 0].legend(fontsize=12, frameon=False, loc="lower right")

    ax = axes[1, 2]
    n = np.arange(0, int(cells["freeze_years"].max()) + 1)
    years = cells["freeze_years"].value_counts().reindex(n, fill_value=0)
    seasons = cells["complete_seasons"].value_counts().reindex(n, fill_value=0)
    w = 0.4
    ax.bar(n - w / 2, years, w, color=DEV_COLOR, label="Freeze seasons with data")
    ax.bar(n + w / 2, seasons, w, color="#eb6834", label="Complete cold seasons")
    ax.set_xticks(n)
    ax.set_xlabel("Number per cell")
    ax.set_ylabel("Development cells")
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.set_title("(f) Record length", loc="left", fontsize=15)
    savefig("insitu_representativeness", fig=fig)


if __name__ == "__main__":
    main()
