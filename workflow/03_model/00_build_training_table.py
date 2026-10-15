"""Training table: every labelled AM day of the development cells outside July and August.

Rows: cell-days labelled frozen, partial or thawed (sfcc-label, cut-off 0.75), with the
standardized indices, the tree-cover fraction and class of the cell, and its spatial block
(insitu.development_days). The model is trained on the frozen and thawed days; partial days are
kept to evaluate the transitional class.

Also writes results/tables/training_composition.csv: the share of the frozen and thawed training
days in each roughness and tree-cover class, raw and with the block weights of the model
(model.block_weights), against the share of the North American land cells of the product domain.

Usage: python 00_build_training_table.py
"""

import numpy as np
import pandas as pd

from nbft import config, insitu, paths
from nbft.model import block_weights


def domain_classes():
    tree = insitu.read_band("treecover", "mean")
    rough = insitu.read_band("terrain", "elevation_std")
    domain = insitu.north_america_mask()
    domain &= insitu.cci_water_fraction() <= insitu.MAX_WATER_FRACTION
    domain &= np.isfinite(tree)
    return pd.DataFrame(
        {
            "tree_class": insitu.tree_cover_class(tree[domain]).astype(str),
            "roughness_class": insitu.roughness_class(rough[domain]).astype(str),
        }
    )


def composition(x):
    train = x[x["label"] != "partial"].assign(w=lambda d: block_weights(d["block"]))
    land = domain_classes()
    parts = []
    for col in ("roughness_class", "tree_class"):
        g = train.groupby(col)
        parts.append(
            pd.DataFrame(
                {
                    "group": col,
                    "cells": g["cell_id"].nunique(),
                    "days_share": g.size() / len(train),
                    "weighted_share": g["w"].sum() / train["w"].sum(),
                    "land_share": land[col].value_counts(normalize=True),
                    "frozen_share": g["frozen"].mean(),
                }
            )
        )
    return pd.concat(parts).rename_axis("class").reset_index()


def main():
    x = insitu.development_days()
    out = config.path("model.training_table")
    out.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(out, index=False)
    print(f"{len(x):,} cell-days at {x['cell_id'].nunique()} cells, {x['block'].nunique()} blocks")
    print(x.groupby(["tree_class", "label"]).size().unstack().to_string())
    print(f"-> {out}")
    comp = composition(x).round(3)
    comp.to_csv(paths.TABLES / "training_composition.csv", index=False)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
