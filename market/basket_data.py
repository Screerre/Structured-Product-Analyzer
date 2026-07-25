"""
Reconstruction d'un sous-jacent synthetique a partir de plusieurs tickers.

Deux usages :
  * fetch_basket(...)        -> vols, matrice de correlation, dividendes, historiques
  * reconstruct_index(...)   -> serie historique de l'indice reconstitue (backtest)

Note importante sur les donnees : yfinance est appele avec auto_adjust=True, donc
les cloture sont ajustees des dividendes. Les series manipulees ici sont donc des
series de RENDEMENT TOTAL. C'est exactement ce qu'il faut pour un indice decrement,
dont le prelevement synthetique remplace le dividende reel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from market.data_provider import _download_close, hist_vol_ewma, hist_vol_simple

TRADING_DAYS = 252


# ==================================================================
# Recuperation
# ==================================================================
def fetch_basket(tickers, lookback_days: int = 730, method: str = "ewma", lam: float = 0.94):
    """
    Telecharge les historiques de plusieurs tickers et calibre les parametres du panier.

    Retourne un dict :
        closes    DataFrame des cloture alignees (dropna sur l'intersection des dates)
        log_ret   DataFrame des rendements log quotidiens
        vols      np.ndarray (k,) vol annualisee par ligne
        corr      np.ndarray (k, k) matrice de correlation des rendements quotidiens
        divs      np.ndarray (k,) rendement du dividende (0 si inconnu)
        ok        bool
        info      message lisible
        missing   liste des tickers qui n'ont pas pu etre telecharges
    """
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(tickers) < 2:
        return {"ok": False, "info": "Il faut au moins deux lignes dans le panier.", "missing": []}

    series, missing = {}, []
    for t in tickers:
        close, msg = _download_close(t, lookback_days)
        if close is None:
            missing.append(f"{t} ({msg})")
        else:
            series[t] = close.rename(t)

    if len(series) < 2:
        return {"ok": False, "info": "Moins de deux tickers exploitables. " + " / ".join(missing),
                "missing": missing}

    closes = pd.concat(series.values(), axis=1).dropna()
    if len(closes) < 60:
        return {"ok": False, "info": f"Seulement {len(closes)} dates communes, insuffisant.",
                "missing": missing}

    log_ret = np.log(closes / closes.shift(1)).dropna()

    if method == "ewma":
        vols = np.array([hist_vol_ewma(log_ret[c], lam) for c in closes.columns], dtype=float)
        mlabel = f"EWMA(lam={lam})"
    else:
        vols = np.array([hist_vol_simple(log_ret[c]) for c in closes.columns], dtype=float)
        mlabel = "equiponderee"

    corr = nearest_correlation(log_ret.corr().to_numpy(dtype=float))
    divs = np.zeros(len(closes.columns))  # cloture deja ajustees : rendement total

    info = (f"{len(closes.columns)} lignes / {len(log_ret)} jours communs "
            f"/ vol {mlabel} de {vols.min()*100:.1f}% a {vols.max()*100:.1f}%")
    if missing:
        info += " | ignores : " + ", ".join(missing)

    return {"ok": True, "closes": closes, "log_ret": log_ret, "vols": vols,
            "corr": corr, "divs": divs, "info": info, "missing": missing,
            "tickers": list(closes.columns)}


def nearest_correlation(c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Projette une matrice sur le cone des matrices semi-definies positives.

    Une matrice de correlation estimee sur un echantillon court, ou avec des donnees
    manquantes, peut ne pas etre PSD. La factorisation de Cholesky echoue alors.
    On ecrete les valeurs propres negatives et on renormalise la diagonale a 1.
    """
    c = np.asarray(c, dtype=float)
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    if vals.min() >= eps:
        np.fill_diagonal(c, 1.0)
        return c
    vals = np.clip(vals, eps, None)
    c2 = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(c2))
    c2 = c2 / np.outer(d, d)
    np.fill_diagonal(c2, 1.0)
    return c2


# ==================================================================
# Reconstruction historique (backtest)
# ==================================================================
def _rebalance_marks(index: pd.DatetimeIndex, freq: str | None) -> list[int]:
    """Indices (>=1) des premiers jours de chaque periode de rebalancement."""
    if freq is None:
        return [1]
    per = index.to_period(freq)
    changed = np.concatenate([[False], per[1:] != per[:-1]])
    marks = list(np.flatnonzero(changed))
    return [1] + [m for m in marks if m >= 1]


def _rebalanced_level(px: np.ndarray, w: np.ndarray, marks: list[int]) -> np.ndarray:
    """
    Niveau d'un panier a pondérations cibles fixes, rebalance aux dates `marks`.

    Entre deux rebalancements les poids derivent (buy and hold sur les unites).
    Le rebalancement remet les poids a la cible, ce qui cree l'effet de drag
    (ou de bonus) que le mode "panier agrege" ne capture pas.
    """
    n = len(px)
    level = np.empty(n)
    level[0] = 1.0
    bounds = sorted(set(marks)) + [n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a >= n:
            break
        b = min(b, n)
        level[a:b] = level[a - 1] * ((px[a:b] / px[a - 1]) @ w)
    return level


def reconstruct_index(closes: pd.DataFrame, weights, rebalance: str | None = "Q",
                      decrement_points: float = 0.0, decrement_rate: float = 0.0,
                      base: float = 100.0) -> pd.Series:
    """
    Reconstitue la serie historique de l'indice.

    rebalance          'Q' trimestriel, 'M' mensuel, 'A' annuel, None = buy and hold
    decrement_points   prelevement en POINTS par an, retranche au niveau (convention marche)
    decrement_rate     prelevement en POURCENTAGE par an (indices type "5% decrement")

    Les deux decrements sont exclusifs dans la pratique. Le points l'emporte s'il est non nul.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    px = closes.to_numpy(dtype=float)
    marks = _rebalance_marks(closes.index, rebalance)
    gross = _rebalanced_level(px, w, marks)

    # fraction d'annee ACT/365 entre deux observations
    days = np.diff(closes.index.to_julian_date().to_numpy(dtype=float))
    dt = np.concatenate([[0.0], days]) / 365.0

    lvl = np.empty(len(gross))
    lvl[0] = base
    for i in range(1, len(gross)):
        x = lvl[i - 1] * (gross[i] / gross[i - 1])
        if decrement_points:
            x -= decrement_points * dt[i]
        elif decrement_rate:
            x *= (1.0 - decrement_rate) ** dt[i]
        lvl[i] = max(x, 0.0)

    return pd.Series(lvl, index=closes.index, name="Indice reconstitue")


def basket_vol(vols, corr, weights) -> float:
    """Vol annualisee du panier : sqrt(w' Sigma w). Utile pour l'affichage et le controle."""
    w = np.asarray(weights, float); w = w / w.sum()
    s = np.asarray(vols, float)
    cov = np.outer(s, s) * np.asarray(corr, float)
    return float(np.sqrt(w @ cov @ w))
