"""Microwave indices and their standardization against a thaw-season reference.

Earlier names, kept in the reference file names: NDV was ``sr``, NDH was ``QER``, Qe was ``QE``.
"""


def normalized_difference(a, b):
    return (a - b) / (a + b)


def npr(smap_v, smap_h):
    """SMAP L-band normalized polarization ratio."""
    return normalized_difference(smap_v, smap_h)


def ndv(amsr_18v, amsr_36v):
    """AMSR2 normalized difference of the 18.7 and 36.5 GHz V-pol channels."""
    return normalized_difference(amsr_18v, amsr_36v)


def ndh(amsr_18h, amsr_36v):
    """AMSR2 normalized difference of 18.7 GHz H-pol and 36.5 GHz V-pol."""
    return normalized_difference(amsr_18h, amsr_36v)


def wd(smap_v, amsr_36v):
    """Frequency difference between SMAP V-pol and AMSR2 36.5 GHz V-pol."""
    return smap_v - amsr_36v


def zscore(raw, ref, std):
    """Standardized anomaly against the per-pixel thaw reference."""
    return (raw - ref) / std
