import numpy as np

from nbft.stats import fraction_in_band, jonckheere_terpstra, kruskal_epsilon_squared


def test_jonckheere_detects_ordering():
    rng = np.random.default_rng(0)
    cold, mid, warm = rng.normal(-10, 1, 200), rng.normal(0, 1, 200), rng.normal(10, 1, 200)
    _, z, p = jonckheere_terpstra(cold, mid, warm)
    assert z > 10 and p < 1e-6


def test_jonckheere_finds_no_ordering_in_noise():
    rng = np.random.default_rng(1)
    a, b, c = (rng.normal(0, 1, 200) for _ in range(3))
    _, _, p = jonckheere_terpstra(a, b, c)
    assert p > 0.01


def test_kruskal_effect_size_zero_for_identical_groups():
    x = np.arange(100.0)
    _, _, eps2 = kruskal_epsilon_squared(x, x.copy())
    assert eps2 == 0.0


def test_fraction_in_band():
    assert fraction_in_band([-8, -3, 0, 1, 5], -7, 2) == 60.0
    assert fraction_in_band([], -7, 2) == 0.0
