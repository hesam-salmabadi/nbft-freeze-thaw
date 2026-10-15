import numpy as np
import pandas as pd

from nbft.training_data import BLOCK_EPS_PIXELS, spatial_blocks


def test_spatial_blocks_group_nearby_pixels_and_separate_far_ones():
    """y/x must be pixel indices: two pixels 5 apart share a block, 500 apart do not."""
    df = pd.DataFrame({"y": [100, 105, 600], "x": [200, 200, 200]})
    blocks = spatial_blocks(df)["spatial_block"].to_numpy()
    assert blocks[0] == blocks[1]
    assert blocks[2] != blocks[0]


def test_block_radius_is_in_pixels_not_metres():
    """Guards the regression where y/x were EASE metres, making every pixel its own block."""
    metres = pd.DataFrame({"y": np.array([3163500.0, 3172500.0]), "x": [-1372500.0, -1372500.0]})
    assert spatial_blocks(metres)["spatial_block"].nunique() == 2, (
        "9 km apart in metres must NOT be treated as within "
        f"{BLOCK_EPS_PIXELS} pixels; y/x must be indices"
    )
