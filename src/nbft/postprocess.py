"""Processing-mask (PM) post-filter, following the SMOS L3 freeze-thaw protocol.

PM seasonal states: 1 summer, 2 late summer, 3 early freezing, 4 evolved freezing, 5 winter,
6 late winter, 7 melting, 8 end of melting.

Rules:
1. Summer (PM 1-2): force unfrozen.
2. Winter (PM 5-6): frozenness may not decrease from the previous day.
"""

import numpy as np
import xarray as xr

SUMMER = (1, 2)
WINTER = (5, 6)

# Product state encoding -> ordinal frozenness, and back
STATE_TO_ORDINAL = {0: 0, 2: 1, 1: 2}  # unfrozen, transitional, frozen
ORDINAL_TO_STATE = {v: k for k, v in STATE_TO_ORDINAL.items()}


def no_thaw_in_winter(state, winter):
    """Along the last axis (time), keep values at least the last valid value where it is winter.

    Missing days (NaN) stay missing and do not break the chain.
    """
    out = np.array(state, dtype=np.float32)
    last = out[..., 0].copy()
    for t in range(1, out.shape[-1]):
        cur = out[..., t]
        valid = ~np.isnan(cur)
        cur = np.where(winter[..., t] & valid & ~np.isnan(last), np.maximum(cur, last), cur)
        out[..., t] = cur
        last = np.where(valid, cur, last)
    return out


def _remap(da, mapping):
    """Map state codes; anything else (including NaN) becomes NaN."""
    out = xr.full_like(da, np.nan, dtype=np.float32)
    for src, dst in mapping.items():
        out = xr.where(da == src, dst, out)
    return out


def _apply_no_thaw(da, winter):
    return xr.apply_ufunc(
        no_thaw_in_winter,
        da,
        winter,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )


def apply_processing_mask(product, pm, chunks=None):
    """Filter the FT product with the PM. Returns filtered states and cleaned D / probability.

    D and probability are set to NaN where the PM overrode the model (summer freeze, or a winter
    thaw that was blocked).
    """
    chunks = chunks or {"time": -1, "y": 128, "x": 128}
    summer = pm.isin(SUMMER)
    winter = pm.isin(WINTER).chunk(chunks)

    # Summer forces unfrozen only where the model produced a value
    state = xr.where(summer & product["state"].notnull(), 0, product["state"]).chunk(chunks)
    binary = xr.where(summer & product["binary_state"].notnull(), 0, product["binary_state"])
    binary = binary.chunk(chunks)

    ordinal = _remap(state, STATE_TO_ORDINAL)
    ordinal_fixed = _apply_no_thaw(ordinal, winter)
    binary_fixed = _apply_no_thaw(binary, winter)

    overridden = (summer & (product["state"] != 0)) | (ordinal_fixed > ordinal)
    out = xr.Dataset(
        {
            "state_filtered": _remap(ordinal_fixed, ORDINAL_TO_STATE).astype(np.float32),
            "binary_state_filtered": binary_fixed.astype(np.float32),
            "D_cleaned": product["D"].where(~overridden),
            "probability_cleaned": product["probability"].where(~overridden),
        }
    )
    out["state_filtered"].attrs["description"] = "0: unfrozen, 1: frozen, 2: transitional (PM)"
    out["binary_state_filtered"].attrs["description"] = "0: unfrozen, 1: frozen (PM)"
    out["D_cleaned"].attrs["description"] = "D, NaN where the PM overrode the model"
    out["probability_cleaned"].attrs["description"] = "P(frozen), NaN where the PM overrode"
    return out
