"""
Composant Streamlit : construction du sous-jacent.

Deux modes :
  * Indice unique    -> un ticker, comportement historique de l'outil
  * Panier construit -> N tickers ponderes, indice reconstitue

underlying_form() retourne un `spec` dict que build_simulator() transforme en
simulateur, indifferemment MarketSimulator ou BasketSimulator (meme interface,
ProductEngine ne voit aucune difference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from market.data_provider import PRESET_TICKERS, fetch_market_data
from market.basket_data import fetch_basket, reconstruct_index, basket_vol
from simulation.montecarlo import MarketSimulator
from simulation.basket import BasketSimulator

PRESET_KEYS = list(PRESET_TICKERS.keys())
LOOKBACKS = {"90 jours": 90, "250 jours (1 an)": 365, "500 jours (2 ans)": 730,
             "1250 jours (5 ans)": 1825}
REBAL = {"Trimestriel": 0.25, "Mensuel": 1 / 12, "Annuel": 1.0, "Aucun (buy and hold)": None}
REBAL_PANDAS = {0.25: "Q", 1 / 12: "M", 1.0: "A", None: None}


def _init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return key


# ==================================================================
# Callbacks
# ==================================================================
def _on_preset(prefix: str):
    p = st.session_state[f"{prefix}_preset"]
    if p != "Personnalise":
        st.session_state[f"{prefix}_ticker"] = PRESET_TICKERS[p]
    _refresh_single(prefix)


def _refresh_single(prefix: str):
    """Recalibre vol et dividende. Ecrit DIRECTEMENT dans les cles des widgets."""
    tkr = str(st.session_state.get(f"{prefix}_ticker", "")).strip()
    win = st.session_state.get(f"{prefix}_win", "500 jours (2 ans)")
    if not tkr:
        st.session_state[f"{prefix}_info"] = "Renseigne un ticker."
        st.session_state[f"{prefix}_ok"] = False
        return
    try:
        v, d, info = fetch_market_data(tkr, LOOKBACKS[win])
    except Exception as exc:
        st.session_state[f"{prefix}_info"] = f"Echec : {exc}"
        st.session_state[f"{prefix}_ok"] = False
        return
    if v is not None:
        st.session_state[f"{prefix}_voln"] = round(float(v), 5)
        if d is not None:
            st.session_state[f"{prefix}_divn"] = round(float(d), 5)
    st.session_state[f"{prefix}_info"] = info
    st.session_state[f"{prefix}_ok"] = v is not None


def _refresh_basket(prefix: str):
    """Telecharge le panier et stocke le resultat calibre en session."""
    df = st.session_state.get(f"{prefix}_basket_df")
    if df is None or len(df) == 0:
        return
    poids = pd.to_numeric(df["Poids (%)"], errors="coerce").fillna(0)
    rows = [(str(t).strip().upper(), float(p))
            for t, p in zip(df["Ticker"], poids)
            if str(t).strip() and float(p) > 0]
    if len(rows) < 2:
        st.session_state[f"{prefix}_basket_res"] = {
            "ok": False, "info": "Il faut au moins deux lignes avec un poids positif."}
        return
    win = st.session_state.get(f"{prefix}_bwin", "500 jours (2 ans)")
    try:
        res = fetch_basket(tuple(t for t, _ in rows), LOOKBACKS[win])
    except Exception as exc:
        res = {"ok": False, "info": f"Echec : {exc}"}
    if res.get("ok"):
        wmap = dict(rows)
        w = np.array([wmap[t] for t in res["tickers"]], dtype=float)
        res["weights"] = w / w.sum()
    st.session_state[f"{prefix}_basket_res"] = res


# ==================================================================
# Formulaire
# ==================================================================
def underlying_form(prefix: str, default_vol: float = 0.20) -> dict:
    """Affiche le constructeur de sous-jacent et retourne un `spec`."""
    _init(f"{prefix}_mode", "Indice unique")
    mode = st.radio("Type de sous-jacent", ["Indice unique", "Panier construit"],
                    horizontal=True, key=f"{prefix}_mode")

    st.caption("Dans un indice decrement, le prelevement REMPLACE le dividende. "
               "Renseigne un decrement OU un dividende, jamais les deux.")
    d1, d2 = st.columns(2)
    _init(f"{prefix}_decpts", 0.0)
    _init(f"{prefix}_decrate", 0.0)
    dec_pts = d1.number_input(
        "Decrement (points/an)", step=1.0, format="%.2f", key=f"{prefix}_decpts",
        help="Convention marche francaise. 50 points coutent 5% a un indice base 1000, "
             "mais 7,1% si l'indice tombe a 700 : le prelevement se durcit quand le "
             "marche baisse, donc quand la barriere de capital est menacee.")
    dec_rate = d2.number_input("Decrement (%/an)", step=0.001, format="%.5f",
                               key=f"{prefix}_decrate", disabled=(dec_pts > 0))

    # ---------------- Mode 1 : ticker unique ----------------
    if mode == "Indice unique":
        _init(f"{prefix}_preset", PRESET_KEYS[0])
        _init(f"{prefix}_ticker", PRESET_TICKERS[st.session_state[f"{prefix}_preset"]])
        _init(f"{prefix}_win", "500 jours (2 ans)")
        _init(f"{prefix}_voln", float(default_vol))
        _init(f"{prefix}_divn", 0.03)

        preset = st.selectbox("Preselection", PRESET_KEYS, key=f"{prefix}_preset",
                              on_change=_on_preset, args=(prefix,))
        ticker = st.text_input("Ticker", key=f"{prefix}_ticker",
                               on_change=_refresh_single, args=(prefix,))
        st.selectbox("Fenetre de calibration", list(LOOKBACKS.keys()), key=f"{prefix}_win",
                     on_change=_refresh_single, args=(prefix,))
        st.button("Forcer la recuperation", key=f"{prefix}_fetch",
                  on_click=_refresh_single, args=(prefix,))

        info = st.session_state.get(f"{prefix}_info")
        if info:
            (st.success if st.session_state.get(f"{prefix}_ok") else st.warning)(info)

        c1, c2 = st.columns(2)
        vol = c1.number_input("Volatilite", step=0.00001, format="%.5f", key=f"{prefix}_voln")
        div = c2.number_input("Dividende", step=0.00001, format="%.5f", key=f"{prefix}_divn",
                              disabled=(dec_pts > 0 or dec_rate > 0))

        name = preset if preset != "Personnalise" else (ticker or "Sous-jacent")
        return {"mode": "single", "ready": bool(str(ticker).strip()), "name": name,
                "ticker": ticker, "vol": float(vol),
                "div": 0.0 if (dec_pts or dec_rate) else float(div),
                "dec_pts": float(dec_pts), "dec_rate": float(dec_rate)}

    # ---------------- Mode 2 : panier ----------------
    st.caption("Reconstitue un indice a partir de plusieurs lignes. Utile quand l'indice "
               "du produit n'est pas cotable, ce qui est le cas de tous les indices sur "
               "mesure construits pour un emetteur.")

    _init(f"{prefix}_basket_df", pd.DataFrame({
        "Ticker": ["ASML.AS", "MC.PA", "NVDA", "JPM"],
        "Poids (%)": [25.0, 25.0, 25.0, 25.0],
    }))
    _init(f"{prefix}_bwin", "500 jours (2 ans)")
    _init(f"{prefix}_rebal", "Trimestriel")

    edited = st.data_editor(
        st.session_state[f"{prefix}_basket_df"], num_rows="dynamic",
        use_container_width=True, key=f"{prefix}_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Symbole Yahoo Finance"),
            "Poids (%)": st.column_config.NumberColumn("Poids (%)", min_value=0.0,
                                                       max_value=100.0, step=0.5, format="%.2f"),
        },
    )
    st.session_state[f"{prefix}_basket_df"] = edited

    total = float(pd.to_numeric(edited["Poids (%)"], errors="coerce").fillna(0).sum())
    cA, cB, cC = st.columns([2, 2, 1])
    cA.selectbox("Fenetre de calibration", list(LOOKBACKS.keys()), key=f"{prefix}_bwin")
    cB.selectbox("Rebalancement", list(REBAL.keys()), key=f"{prefix}_rebal",
                 help="Un panier rebalance n'a pas la meme dynamique qu'un buy and hold. "
                      "Les indices de marche sont quasi tous rebalances trimestriellement.")
    cC.metric("Somme", f"{total:.1f}%")
    if abs(total - 100.0) > 0.01:
        st.caption(f"Somme a {total:.2f}%, les poids seront renormalises a 100%.")

    st.button("Calibrer le panier", key=f"{prefix}_bfetch", type="secondary",
              on_click=_refresh_basket, args=(prefix,))

    res = st.session_state.get(f"{prefix}_basket_res")
    base_ko = {"mode": "basket", "ready": False, "name": "Panier",
               "dec_pts": float(dec_pts), "dec_rate": float(dec_rate)}
    if not res:
        st.info("Renseigne les lignes puis clique sur Calibrer le panier.")
        return base_ko
    if not res.get("ok"):
        st.warning(res.get("info", "Calibration impossible."))
        return base_ko

    st.success(res["info"])
    w = res["weights"]
    bvol = basket_vol(res["vols"], res["corr"], w)
    avg_vol = float(np.dot(w, res["vols"]))

    m = st.columns(3)
    m[0].metric("Vol du panier", f"{bvol*100:.1f}%")
    m[1].metric("Vol moyenne ponderee", f"{avg_vol*100:.1f}%")
    m[2].metric("Gain de diversification", f"{(1 - bvol/avg_vol)*100:.1f}%")
    st.caption("L'ecart entre les deux vols mesure la diversification reelle. S'il est "
               "faible, les secteurs affiches ne sont qu'un seul facteur de risque. "
               "Correlation historique de temps calme : elle monte vers 1 dans les krachs, "
               "donc ce gain est surestime dans les scenarios de perte.")

    with st.expander("Matrice de correlation"):
        st.dataframe(
            pd.DataFrame(res["corr"], index=res["tickers"], columns=res["tickers"])
            .style.format("{:+.2f}").background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
            use_container_width=True)

    with st.expander("Backtest de l'indice reconstitue"):
        rebal_y = REBAL[st.session_state[f"{prefix}_rebal"]]
        brut = reconstruct_index(res["closes"], w, rebalance=REBAL_PANDAS[rebal_y])
        serie = reconstruct_index(res["closes"], w, rebalance=REBAL_PANDAS[rebal_y],
                                  decrement_points=dec_pts, decrement_rate=dec_rate)
        st.line_chart(pd.DataFrame({"Sans decrement": brut, "Avec decrement": serie}))
        if dec_pts or dec_rate:
            drag = (brut.iloc[-1] - serie.iloc[-1]) / brut.iloc[-1]
            st.caption(f"Cout cumule du decrement : {drag*100:.1f}% du niveau final.")
        st.caption("Reconstitution ex post a partir des composants actuels. Ce n'est PAS "
                   "la performance d'un indice reellement publie : les constituants "
                   "d'aujourd'hui ont ete selectionnes en connaissant le passe.")

    return {"mode": "basket", "ready": True, "name": "Panier reconstruit",
            "ticker": ", ".join(res["tickers"]),
            "weights": w, "vols": res["vols"], "corr": res["corr"], "divs": res["divs"],
            "tickers": res["tickers"],
            "rebalance_years": REBAL[st.session_state[f"{prefix}_rebal"]],
            "vol": bvol, "div": 0.0,
            "dec_pts": float(dec_pts), "dec_rate": float(dec_rate)}


# ==================================================================
# Fabrique de simulateur
# ==================================================================
def build_simulator(spec: dict, spot: float, r: float, seed=None, drift: float | None = None):
    """
    Transforme un `spec` en simulateur. ProductEngine ne voit aucune difference.

    drift=None  -> mesure risque-neutre (valorisation)
    drift=0.07  -> mesure reelle (probabilites montrees au client)

    Dans les deux cas `r` reste le taux d'ACTUALISATION : le drift ne le remplace pas.
    """
    if not spec.get("ready", True):
        raise ValueError("Sous-jacent non calibre.")

    if spec.get("mode") == "basket":
        return BasketSimulator(
            spot=spot, r=r, drift=drift,
            weights=spec["weights"], sigmas=spec["vols"], corr=spec["corr"],
            div_yields=spec["divs"], total_return=True,
            rebalance_years=spec["rebalance_years"],
            decrement_points=spec["dec_pts"], decrement_rate=spec["dec_rate"],
            steps_per_year=52, seed=seed)

    # Mono-actif avec decrement en points : BasketSimulator a un constituant,
    # car le decrement en points est path-dependent et demande une grille fine.
    if spec.get("dec_pts"):
        return BasketSimulator(
            spot=spot, r=r, drift=drift,
            weights=[1.0], sigmas=[spec["vol"]], corr=[[1.0]],
            total_return=True, rebalance_years=None,
            decrement_points=spec["dec_pts"], steps_per_year=52, seed=seed)

    return MarketSimulator(spot=spot, r=r, sigma=spec["vol"], div_yield=spec["div"],
                           seed=seed, drift=drift)
