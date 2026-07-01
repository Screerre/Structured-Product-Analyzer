import numpy as np

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


def _download_close(ticker: str, lookback_days: int):
    """Telecharge les cloture. Retourne une Series pandas ou None + message."""
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


def hist_vol_simple(log_ret) -> float:
    """Vol historique annualisee, equiponderee (ecart-type classique)."""
    return float(log_ret.std() * np.sqrt(252))


def hist_vol_ewma(log_ret, lam: float = 0.94) -> float:
    """
    Vol EWMA annualisee (style RiskMetrics) : plus de poids aux jours recents.
    lam=0.94 est le standard quotidien RiskMetrics.
    """
    r = np.asarray(log_ret, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    # variance EWMA recursive
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x ** 2
    return float(np.sqrt(var * 252))


def fetch_market_data(ticker: str, lookback_days: int = 365, method: str = "ewma", lam: float = 0.94):
    """
    Recupere vol historique (methode 'simple' ou 'ewma') + dividende indicatif.
    Retourne (vol, div, info). vol=None si echec. Ne leve jamais d'exception.
    """
    close, msg = _download_close(ticker, lookback_days)
    if close is None:
        return None, None, msg
    log_ret = np.log(close / close.shift(1)).dropna()
    if method == "ewma":
        vol = hist_vol_ewma(log_ret, lam)
        mlabel = f"EWMA(lam={lam})"
    else:
        vol = hist_vol_simple(log_ret)
        mlabel = "equiponderee"
    last = float(close.iloc[-1])
    div = DEFAULT_DIV.get(ticker.upper(), 0.0)
    info = f"{ticker} : vol {vol*100:.1f}% [{mlabel}] / {len(log_ret)}j (cours {last:.0f}, div~{div*100:.1f}%)"
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