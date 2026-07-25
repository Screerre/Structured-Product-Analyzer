import numpy as np


class MarketSimulator:
    """
    Simule N trajectoires d'UN sous-jacent par GBM, vectorise.
    Integre un taux de dividende continu q : le drift devient (mu - q - 0.5*sigma^2).

    IMPORTANT : `r` et `drift` sont deux choses distinctes.
      * r      sert a l'ACTUALISATION des flux (ProductEngine lit simulator.r)
      * drift  est le rendement attendu utilise dans la diffusion

    drift=None  -> mesure RISQUE-NEUTRE (drift = r). Correct pour calculer une
                   valeur actualisee, mais les probabilites qui en sortent sont
                   des probabilites risque-neutres, pessimistes par construction
                   puisqu'elles incorporent la prime de risque actions.
    drift=0.07  -> mesure REELLE. Correct pour les probabilites montrees a un
                   client, mais la valeur actualisee n'a alors plus de sens.

    Ne jamais melanger les deux lectures dans un meme tableau de synthese.
    """

    def __init__(self, spot: float, r: float, sigma: float, div_yield: float = 0.0,
                 seed=None, drift: float | None = None):
        self.spot = float(spot)
        self.r = float(r)                                    # actualisation
        self.sigma = float(sigma)
        self.q = float(div_yield)
        self.mu = float(r) if drift is None else float(drift)  # diffusion
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, times, n_sims: int) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        dts = np.diff(np.concatenate([[0.0], times]))
        Z = self.rng.standard_normal((n_sims, len(times)))
        drift = (self.mu - self.q - 0.5 * self.sigma ** 2) * dts
        diff = self.sigma * np.sqrt(dts)[None, :] * Z
        log_cum = np.cumsum(drift[None, :] + diff, axis=1)
        return self.spot * np.exp(log_cum)
