import numpy as np
import streamlit as st

PRESET_TICKERS = {
    "Euro Stoxx 50": "^STOXX50E",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI",
    "S&P 500": "^GSPC",
    "Nikkei 225": "^N225",
    "Personnalise": "",
}
DEFAULT_DIV = {
    "^STOXX50E": 0.030, "^FCHI": 0.030, "^GDAXI": 0.025,
    "^GSPC": 0.014, "^N225": 0.018,
}


# ==================================================================
# Telechargement
# ==================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _download_close(ticker: str, lookback_days: int):
    """
    Telecharge les cloture. Retourne (Series pandas ou None, message).

    Le cache est indispensable : avec le rafraichissement automatique de la vol,
    cette fonction est appelee a chaque changement de champ. Les deux arguments
    sont dans la signature, sinon le cache renverrait eternellement le premier
    resultat quel que soit le ticker demande.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance non installe (pip install yfinance)"
    try:
        data = yf.download(ticker, period=f"{lookback_days}d", progress=False, auto_adjust=True)
        if data is None or len(data) < 30:
            return None, f"Pas assez de donnees pour {ticker}"
        close = data["Close"].dropna()
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        return close, "ok"
    except Exception as e:
        return None, f"Echec {ticker} : {type(e).__name__}"


# ==================================================================
# Estimateurs de volatilite
# ==================================================================
def hist_vol_simple(log_ret) -> float:
    """Vol historique annualisee, equiponderee."""
    return float(np.std(np.asarray(log_ret, dtype=float)) * np.sqrt(252))


def hist_vol_ewma(log_ret, lam: float = 0.94) -> float:
    """
    Vol EWMA annualisee (RiskMetrics). Vectorisee.

    ATTENTION AU CHOIX DE LAMBDA. lam=0.94 a une demi-vie d'environ 11 jours et
    une memoire effective d'environ 17 jours : la fenetre de calibration choisie
    dans l'UI n'a alors quasiment aucun effet, la vol est dictee par le dernier
    mois. C'est le bon estimateur pour une VaR a un jour, le mauvais pour un
    produit a 8 ans. Pour du long terme, utiliser lam=0.99 (demi-vie ~69 jours)
    ou hist_vol_blend ci-dessous.
    """
    r = np.asarray(log_ret, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    # poids decroissants du plus recent au plus ancien, normalises
    age = np.arange(r.size - 1, -1, -1, dtype=float)
    wts = (1.0 - lam) * lam ** age
    wts /= wts.sum()
    var = float(np.dot(wts, r ** 2))
    return float(np.sqrt(var * 252))


def hist_vol_blend(log_ret, lam: float = 0.94, w_short: float = 0.5) -> float:
    """
    Melange vol courte (EWMA, regime actuel) et vol longue (equiponderee sur toute
    la fenetre). Compromis raisonnable pour un produit a plusieurs annees : on ne
    veut ni figer la vol du dernier mois, ni ignorer le regime en cours.
    """
    s = hist_vol_ewma(log_ret, lam)
    l = hist_vol_simple(log_ret)
    if np.isnan(s):
        return l
    return float(w_short * s + (1.0 - w_short) * l)


VOL_METHODS = {"ewma": hist_vol_ewma, "simple": hist_vol_simple, "blend": hist_vol_blend}


# ==================================================================
# API
# ==================================================================
def fetch_market_data(ticker: str, lookback_days: int = 365,
                      method: str = "blend", lam: float = 0.94):
    """
    Recupere vol historique + dividende indicatif.
    Retourne (vol, div, info). vol=None si echec. Ne leve jamais d'exception.

    method : 'ewma' (court terme), 'simple' (equipondere), 'blend' (defaut).
    """
    close, msg = _download_close(ticker, lookback_days)
    if close is None:
        return None, None, msg
    log_ret = np.log(close / close.shift(1)).dropna()

    if method == "ewma":
        vol = hist_vol_ewma(log_ret, lam); mlabel = f"EWMA(lam={lam})"
    elif method == "simple":
        vol = hist_vol_simple(log_ret); mlabel = "equiponderee"
    else:
        vol = hist_vol_blend(log_ret, lam); mlabel = "mixte 50/50 EWMA + longue"

    last = float(close.iloc[-1])
    div = DEFAULT_DIV.get(ticker.upper(), 0.0)
    info = (f"{ticker} : vol {vol*100:.1f}% [{mlabel}] / {len(log_ret)}j "
            f"(cours {last:.0f}, div~{div*100:.1f}%)")
    if ticker.upper() not in DEFAULT_DIV:
        info += " | dividende inconnu, mis a 0, a saisir a la main"
    return vol, div, info


def correlation(ticker1: str, ticker2: str, lookback_days: int = 365):
    """
    Correlation HISTORIQUE des rendements quotidiens entre deux sous-jacents.
    Retourne (corr, info, df_aligned) ou (None, message, None).
    """
    c1, m1 = _download_close(ticker1, lookback_days)
    if c1 is None:
        return None, f"{ticker1} : {m1}", None
    c2, m2 = _download_close(ticker2, lookback_days)
    if c2 is None:
        return None, f"{ticker2} : {m2}", None
    import pandas as pd
    df = pd.concat([c1.rename("a"), c2.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return None, "Pas assez de dates communes entre les deux series", None
    ra = np.log(df["a"] / df["a"].shift(1))
    rb = np.log(df["b"] / df["b"].shift(1))
    rr = pd.concat([ra, rb], axis=1).dropna()
    corr = float(rr.corr().iloc[0, 1])
    info = f"Correlation {corr:+.2f} sur {len(rr)} jours communs"
    return corr, info, rr
