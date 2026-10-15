"""Study constants shared across the workflow."""

import numpy as np

# Study period: eight complete freeze-thaw years, 2015/16 to 2022/23 (1 August to 31 July)
TIME_START = np.datetime64("2015-08-01")
TIME_END = np.datetime64("2023-08-01")

# Model features: standardized SMAP L-band NPR and AMSR2 18.7/36.5 GHz V-pol normalized difference
MODEL_FEATURES = ["npr", "ndv"]

# Target biomes (merged MODIS/UMD classes): model biome ID -> raw classes merged into it.
# Raw classes: MCD12C1 LC_Type2 (UMD). Class 11 (permanent wetlands) does not occur on the grid.
LC_GROUPS = {5: [1, 2, 3, 4, 5], 7: [6, 7], 8: [8, 9], 10: [10], 12: [12]}
TARGET_LCS = [12, 5, 8, 7, 10]
LC_NAMES = {
    12: "Cropland",
    5: "Mixed Forest",
    8: "Boreal Forest",
    7: "Boreal Tundra",
    10: "Tundra",
}

# Labels
TARGET = "target"  # 0 = unfrozen, 1 = frozen, 2 = transitional (ambiguous)
STRATIFY_COL = "landcover"

# Transition band: distances to these probability isolines, multiplied by the scale factor, give
# the transitional class bounds. Because D is a distance in logit space, scaling is equivalent to
# choosing a narrower pair: (0.1, 0.9) at scale 0.5 is the same band as roughly (0.27, 0.77) at
# scale 1.0. See results/tables/band_sensitivity.csv.
TRANSITION_PROBS = (0.1, 0.9)
TRANSITION_BAND_SCALE = 0.5

# Cross-validation
NUM_FOLDS = 6
POINTS_PER_BLOCK = 500

# Pixels with water fraction at or above this are masked in the product
MAX_WATER_FRACTION = 0.15

FEATURE_LABELS = {
    "npr": "$Z_{NPR}$",
    "ndv": "$Z_{NDV}$",
    "ndh": "$Z_{NDH}$",
    "qe": "$Q_{e}$",
    "qe1": "$Q_{e1}$",
    "amsr_36v": "$TB_{36.5V}$",
    "amsr_36h": "$TB_{36.5H}$",
    "amsr_18v": "$TB_{18.7V}$",
    "amsr_18h": "$TB_{18.7H}$",
    "smap_v": "$TB_{1.4V}$",
    "smap_h": "$TB_{1.4H}$",
    "DAV_AMSR": "$DAV_{AMSR}$",
    "DAV_SMAP": "$DAV_{SMAP}$",
}
