"""Non-parametric tests used for the thermal validation."""

import numpy as np
from scipy import stats


def kruskal_epsilon_squared(*groups):
    """Kruskal-Wallis H with the epsilon-squared effect size.

    Effect size is used alongside p-values because the samples are large enough that almost any
    difference is 'significant'.
    """
    groups = [np.asarray(g) for g in groups if len(g)]
    if len(groups) < 2:
        return 0.0, 1.0, 0.0
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = max(0.0, (h - k + 1) / (n - k))
    return float(h), float(p), float(eps2)


def jonckheere_terpstra(*groups):
    """Jonckheere-Terpstra test for an ordered alternative, groups given coldest first.

    For the FT states the alternative is T_frozen <= T_transitional <= T_unfrozen. The statistic
    sums the Mann-Whitney pair counts that run in the expected direction, and is compared with its
    null distribution (normal approximation, no tie correction).

    Returns (JT, Z, one-sided p-value).
    """
    groups = [np.asarray(g) for g in groups]
    sizes = [len(g) for g in groups]
    n = sum(sizes)

    jt = sum(
        stats.mannwhitneyu(groups[j], groups[i]).statistic
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    )
    expected = (n**2 - sum(s**2 for s in sizes)) / 4
    variance = (n**2 * (2 * n + 3) - sum(s**2 * (2 * s + 3) for s in sizes)) / 72
    z = (jt - expected) / np.sqrt(variance)
    return float(jt), float(z), float(1 - stats.norm.cdf(z))


def fraction_in_band(values, lower, upper):
    """Percentage of values inside a temperature band."""
    v = np.asarray(values)
    return 100 * float(np.mean((v >= lower) & (v <= upper))) if v.size else 0.0
