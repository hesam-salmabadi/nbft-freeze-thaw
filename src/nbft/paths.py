"""Locations inside the repository for tracked outputs."""

from nbft.config import REPO_ROOT

TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"
PAPER_FIGS = REPO_ROOT / "paper" / "figs"

SCALER_PARAMS = TABLES / "scaler_params.csv"
PRODUCTION_EQUATIONS = TABLES / "production_equations.csv"
STATION_METADATA = TABLES / "station_metadata.csv"
