"""Fit the production model on every clean day of the development cells.

Writes results/tables/model_coefficients.csv: one row per class of nbft.model (thawed is the zero
reference row), intercept and TERMS in the units of the standardized indices and the tree-cover
fraction. 07_upscale.py applies them to the maps.

Usage: python 06_train_production.py
"""

import pandas as pd

from nbft import config, paths
from nbft.model import LABEL_CLASS, design_frame, fit_multinomial, model_weights


def main():
    x = pd.read_parquet(config.path("model.training_table"))
    x = x[x["clean"]]
    y = x["label"].map(LABEL_CLASS)
    coef = fit_multinomial(design_frame(x), y, model_weights(x["block"], y))
    coef["training_days"] = len(x)
    coef["training_cells"] = x["cell_id"].nunique()
    coef.to_csv(paths.TABLES / "model_coefficients.csv", index_label="class")
    print(coef.round(4).to_string())


if __name__ == "__main__":
    main()
