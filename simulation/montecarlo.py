import numpy as np


class MarketSimulator:
    """
    Simule N trajectoires d'UN sous-jacent par GBM, vectorise.
    Integre un taux de dividende continu q : le drift devient (r - q - 0.5*sigma^2).
    C'est essentiel pour les indices a fort rendement (Euro Stoxx 50, CAC 40).
    """

    def __init__(self, spot: float, r: float, sigma: float, div_yield: float = 0.0, seed=None):
        self.spot = float(spot)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(div_yield)
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, times, n_sims: int) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        dts = np.diff(np.concatenate([[0.0], times]))
        Z = self.rng.standard_normal((n_sims, len(times)))
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dts
        diff = self.sigma * np.sqrt(dts)[None, :] * Z
        log_cum = np.cumsum(drift[None, :] + diff, axis=1)
        return self.spot * np.exp(log_cum)