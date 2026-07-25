"""
Simulateur Monte Carlo d'un indice reconstitue a partir de N constituants.

Interface volontairement identique a MarketSimulator :
    .spot, .r, .simulate_paths(times, n_sims) -> ndarray (n_sims, len(times))
ProductEngine peut donc consommer ce simulateur sans aucune modification.

Trois differences de fond avec une simulation mono-actif sur la vol du panier :

1. Le rebalancement. Un indice a ponderations fixes rebalancees trimestriellement
   n'a pas la meme dynamique qu'un panier buy and hold. Simuler chaque constituant
   et rebalancer capture cet effet ; simuler directement le niveau du panier ne le
   capture pas.

2. Le decrement en points. Il se calcule sur le NIVEAU, pas sur le rendement, donc
   il est path-dependent : il faut une grille fine, pas seulement les dates de
   constatation. Un decrement de 50 points coute 5% a un indice a 1000 et 7,1% a
   un indice a 700. C'est ce durcissement dans les scenarios baissiers qui compte.

3. Le rendement total. Un indice decrement est bati sur un indice de rendement
   total dont on retranche le prelevement synthetique. Le decrement REMPLACE le
   dividende, il ne s'y ajoute pas. En mode total_return=True, q est donc ignore.
"""
from __future__ import annotations

import numpy as np


class BasketSimulator:
    """
    Diffusion multi-actifs (GBM correle, Cholesky) puis reconstruction du niveau d'indice.

    Parametres
    ----------
    spot              niveau initial de l'indice (base 100 en general)
    r                 taux sans risque continu
    drift             None -> drift risque-neutre r (valorisation).
                      Sinon rendement attendu annuel sous la mesure REELLE
                      (mode scenario, pour les probabilites montrees au client).
    weights           ponderations cibles (k,), normalisees en interne
    sigmas            vols annualisees par constituant (k,)
    corr              matrice de correlation (k, k), doit etre PSD
    div_yields        rendements du dividende (k,). Ignores si total_return=True.
    total_return      True pour un indice decrement (le decrement remplace le dividende)
    rebalance_years   0.25 trimestriel, 1.0 annuel, None pour buy and hold
    decrement_points  prelevement en POINTS par an, retranche au niveau
    decrement_rate    prelevement en POURCENTAGE par an (exclusif du precedent)
    steps_per_year    finesse de la grille de diffusion. 52 suffit pour un decrement
                      en points ; monter a 252 si un jour tu ajoutes une barriere
                      americaine (observation continue).
    batch_size        nombre de trajectoires traitees en memoire simultanement
    """

    def __init__(self, spot: float, r: float, weights, sigmas, corr,
                 div_yields=None, drift: float | None = None,
                 total_return: bool = True, rebalance_years: float | None = 0.25,
                 decrement_points: float = 0.0, decrement_rate: float = 0.0,
                 steps_per_year: int = 52, seed=None, batch_size: int = 20000):
        self.spot = float(spot)
        self.r = float(r)
        self.mu = float(r) if drift is None else float(drift)

        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size < 1:
            raise ValueError("weights doit etre un vecteur non vide")
        if np.any(w < 0):
            raise ValueError("ponderations negatives non supportees")
        self.w = w / w.sum()

        self.sigma = np.asarray(sigmas, dtype=float)
        if self.sigma.shape != self.w.shape:
            raise ValueError("sigmas et weights de tailles differentes")

        self.corr = np.asarray(corr, dtype=float)
        k = self.w.size
        if self.corr.shape != (k, k):
            raise ValueError(f"corr doit etre {k}x{k}")

        q = np.zeros(k) if (div_yields is None or total_return) else np.asarray(div_yields, float)
        self.q = q

        self.total_return = bool(total_return)
        self.rebalance_years = rebalance_years
        self.dec_pts = float(decrement_points)
        self.dec_rate = float(decrement_rate)
        self.steps_per_year = int(steps_per_year)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

        try:
            self.L = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Matrice de correlation non definie positive. Passe-la par "
                "market.basket_data.nearest_correlation avant de construire le simulateur."
            ) from exc

    # ------------------------------------------------------------------
    @property
    def basket_vol(self) -> float:
        """Vol instantanee du panier a ponderations cibles. Indicatif, hors rebalancement."""
        cov = np.outer(self.sigma, self.sigma) * self.corr
        return float(np.sqrt(self.w @ cov @ self.w))

    def _build_grid(self, times: np.ndarray):
        """Grille de diffusion contenant 0, toutes les dates de constatation, et un pas regulier."""
        T = float(times[-1])
        n_reg = max(int(np.ceil(T * self.steps_per_year)), 1)
        regular = np.linspace(0.0, T, n_reg + 1)
        grid = np.unique(np.concatenate([[0.0], regular, times]))
        obs_pos = np.searchsorted(grid, times)
        return grid, obs_pos

    def _rebalance_mask(self, grid: np.ndarray) -> np.ndarray:
        """True aux pas de grille ou l'on remet les poids a la cible."""
        mask = np.zeros(len(grid), dtype=bool)
        if not self.rebalance_years:
            return mask
        step = float(self.rebalance_years)
        nxt = step
        for i in range(1, len(grid)):
            if grid[i] >= nxt - 1e-12:
                mask[i] = True
                while nxt <= grid[i] + 1e-12:
                    nxt += step
        return mask

    # ------------------------------------------------------------------
    def simulate_paths(self, times, n_sims: int) -> np.ndarray:
        """Retourne les niveaux d'indice aux dates de constatation, shape (n_sims, len(times))."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times doit etre un vecteur non vide")

        grid, obs_pos = self._build_grid(times)
        dts = np.diff(grid)
        rebal = self._rebalance_mask(grid)
        obs_at = {int(p): j for j, p in enumerate(obs_pos)}

        k = self.w.size
        mu_i = self.mu - self.q - 0.5 * self.sigma ** 2   # drift log par constituant
        out = np.empty((n_sims, times.size), dtype=float)

        for start in range(0, n_sims, self.batch_size):
            m = min(self.batch_size, n_sims - start)

            X = np.ones((m, k))                 # prix relatifs, base 1
            units = np.tile(self.w, (m, 1))     # unites detenues, valeur initiale du panier = 1
            gross = np.ones(m)                  # niveau du panier brut (rendement total)
            level = np.full(m, self.spot)       # niveau de l'indice, decrement inclus

            if 0 in obs_at:
                out[start:start + m, obs_at[0]] = level

            for j, dt in enumerate(dts):
                sdt = np.sqrt(dt)
                dW = self.rng.standard_normal((m, k)) @ self.L.T
                X *= np.exp(mu_i * dt + self.sigma * sdt * dW)

                gross_new = np.einsum("ij,ij->i", units, X)
                level = level * (gross_new / gross)

                if self.dec_pts:
                    level -= self.dec_pts * dt
                elif self.dec_rate:
                    level *= (1.0 - self.dec_rate) ** dt
                np.maximum(level, 0.0, out=level)

                gross = gross_new
                pos = j + 1
                if rebal[pos]:
                    units = self.w[None, :] * (gross[:, None] / X)
                if pos in obs_at:
                    out[start:start + m, obs_at[pos]] = level

        return out


# ======================================================================
# Controle de coherence
# ======================================================================
def sanity_check_vs_single(seed: int = 7, n: int = 200_000) -> dict:
    """
    Un panier a un seul constituant, sans decrement et sans rebalancement, doit
    reproduire MarketSimulator. Ecart attendu de l'ordre de l'erreur Monte Carlo.

    A brancher dans ta suite de validation mathematique.
    """
    from simulation.montecarlo import MarketSimulator

    times = np.array([0.5, 1.0, 2.0, 5.0])
    sigma, r, q, spot = 0.22, 0.025, 0.0, 100.0

    single = MarketSimulator(spot=spot, r=r, sigma=sigma, div_yield=q, seed=seed)
    basket = BasketSimulator(spot=spot, r=r, weights=[1.0], sigmas=[sigma],
                             corr=[[1.0]], total_return=True, rebalance_years=None,
                             steps_per_year=252, seed=seed)

    a = single.simulate_paths(times, n).mean(axis=0)
    b = basket.simulate_paths(times, n).mean(axis=0)
    theo = spot * np.exp((r - q) * times)

    return {"single": a, "basket": b, "theorique": theo,
            "ecart_relatif_max": float(np.max(np.abs(b - theo) / theo))}
