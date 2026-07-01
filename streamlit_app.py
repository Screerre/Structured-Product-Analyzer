import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from core.product import StructuredProduct, Underlying, Coupon, Autocall, CapitalProtection
from core.calendar import CalendarBuilder, FREQ_LABEL, FREQ_STEP
from simulation.montecarlo import MarketSimulator
from simulation.product_engine import ProductEngine
from analytics.probabilities import ProbabilityAnalyzer
from market.data_provider import (PRESET_TICKERS, fetch_market_data, correlation)

st.set_page_config(page_title="Autocall Analyzer", layout="wide")
st.title("Autocall Analyzer — single stock")
st.caption("Outil d'aide a la decision pour CGP. Aucune donnee client.")

LOOKBACKS = {"90 jours": 90, "250 jours (1 an)": 365, "500 jours (2 ans)": 730, "1250 jours (5 ans)": 1825}

# Palette institutionnelle
GREEN = "#2E6356"; GOLD = "#977B3C"; RED = "#A83838"; INK = "#15191D"; GREY = "#9AA0A6"


# ------------------------------------------------------------------
# Helper : champ numerique a 5 decimales (point 1)
# ------------------------------------------------------------------
def num5(container, label, value, step=0.00001, key=None, disabled=False):
    """number_input standardise a 5 decimales pour les taux / niveaux / barrieres."""
    return container.number_input(label, value=float(value), step=step,
                                  format="%.5f", key=key, disabled=disabled)


def product_form(prefix: str, defaults: dict):
    """Formulaire produit. Retourne (product, vol, div, r, spot, name)."""
    st.markdown(f"#### {defaults['title']}")
    preset = st.selectbox("Sous-jacent", list(PRESET_TICKERS.keys()), key=f"{prefix}_preset")
    ticker = st.text_input("Ticker", value=PRESET_TICKERS[preset], key=f"{prefix}_ticker")
    name = preset if preset != "Personnalise" else (ticker or "Sous-jacent")

    vk, dk = f"{prefix}_vol", f"{prefix}_div"
    if vk not in st.session_state: st.session_state[vk] = defaults["vol"]
    if dk not in st.session_state: st.session_state[dk] = 0.03

    # Point 2 : plus de case "Methode". Vol = EWMA (RiskMetrics), la plus reactive.
    win = st.selectbox("Fenetre de calibration vol", list(LOOKBACKS.keys()), index=1, key=f"{prefix}_win")
    st.caption("Volatilite calibree en EWMA (lambda 0,94, standard RiskMetrics).")
    if st.button("Recuperer vol + dividende", key=f"{prefix}_fetch"):
        if ticker.strip():
            with st.spinner("Telechargement..."):
                v, d, info = fetch_market_data(ticker.strip(), LOOKBACKS[win])
            if v is not None:
                st.session_state[vk] = round(v, 5)
                if d is not None: st.session_state[dk] = round(d, 5)
                st.success(info)
            else:
                st.warning(info)
        else:
            st.warning("Renseigne un ticker.")

    c1, c2 = st.columns(2)
    vol = num5(c1, "Volatilite", st.session_state[vk], key=f"{prefix}_voln")
    div = num5(c2, "Dividende", st.session_state[dk], key=f"{prefix}_divn")
    c3, c4 = st.columns(2)
    spot = num5(c3, "Niveau initial", 100.0, step=1.0, key=f"{prefix}_spot")
    r = num5(c4, "Taux sans risque", 0.025, key=f"{prefix}_r")
    c5, c6 = st.columns(2)
    maturity = c5.number_input("Maturite (ans)", value=float(defaults["maturity"]), step=0.25,
                               format="%.2f", key=f"{prefix}_mat")
    frequency = c6.selectbox("Constatation", list(FREQ_LABEL.keys()), index=3,
                             format_func=lambda k: FREQ_LABEL[k], key=f"{prefix}_freq")
    start = st.date_input("Date de lancement", value=date.today(), key=f"{prefix}_start")

    st.caption("Coupon")
    c7, c8 = st.columns(2)
    # Point 3 : coupon ANNUEL. Le moteur repartit ensuite par constatation.
    crate_annual = num5(c7, "Coupon annuel", defaults["coupon"], key=f"{prefix}_cr")
    cnat = c8.selectbox("Nature", ["conditional", "guaranteed"],
                        format_func=lambda x: {"conditional": "Conditionnel", "guaranteed": "Garanti"}[x],
                        key=f"{prefix}_cnat")
    step = FREQ_STEP[frequency]
    crate_period = crate_annual * step
    st.caption(f"Soit {crate_period*100:.3f}% verse a chaque constatation "
               f"({FREQ_LABEL[frequency].lower()}, {round(1/step)}/an).")
    c9, c10 = st.columns(2)
    cbar = num5(c9, "Barr. coupon", 0.70, step=0.01, key=f"{prefix}_cbar", disabled=(cnat == "guaranteed"))
    cmem = c10.checkbox("Memoire", value=True, disabled=(cnat == "guaranteed"), key=f"{prefix}_cmem")

    st.caption("Rappel & protection")
    c11, c12 = st.columns(2)
    abar = num5(c11, "Barr. rappel", 1.00, step=0.01, key=f"{prefix}_abar")
    fcall = c12.number_input("1re constat. (an)", value=1.0, step=0.25, format="%.2f", key=f"{prefix}_fc")
    c13, c14 = st.columns(2)
    usd = c13.checkbox("Step-down", value=defaults["stepdown"], key=f"{prefix}_usd")
    sd = num5(c14, "Step-down pts/an", 0.05, step=0.01, key=f"{prefix}_sd", disabled=not usd)
    kbar = num5(st, "Barr. protection capital", 0.60, step=0.01, key=f"{prefix}_kbar")

    product = StructuredProduct(
        name=f"Autocall {name}", maturity_years=float(maturity),
        underlying=Underlying(name, ticker, spot), frequency=frequency, start_date=start.isoformat(),
        coupon=Coupon(crate_period, cbar, cmem, cnat),
        autocall=Autocall(True, abar, fcall, sd if usd else 0.0),
        capital=CapitalProtection(kbar, True),
    )
    return product, vol, div, r, spot, name


def run_product(product, vol, div, r, spot, n_sims):
    cal = CalendarBuilder().build(product)
    sim = MarketSimulator(spot=spot, r=r, sigma=vol, div_yield=div, seed=None)
    res = ProductEngine(product, sim).run(cal, n_simulations=n_sims)
    return cal, res, ProbabilityAnalyzer().analyze(res)


# ------------------------------------------------------------------
# Graphiques Monte Carlo (point 5)
# ------------------------------------------------------------------
def mc_figure(cal, res, prod, spot):
    times = np.asarray(cal.times)
    t_axis = np.concatenate([[0.0], times])
    paths = res["paths"]
    allp = np.concatenate([np.full((len(paths), 1), spot), paths], axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2),
                                        gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # --- Panel 1 : faisceau de trajectoires + percentiles + barrieres ---
    samp = paths[np.random.choice(len(paths), size=min(300, len(paths)), replace=False)]
    for row in samp:
        ax1.plot(t_axis, np.concatenate([[spot], row]), color=GREEN, alpha=0.05, lw=0.7)
    ax1.plot([], [], color=GREEN, alpha=0.5, lw=1, label="Trajectoires simulees")
    for q, col, l in [(5, RED, "Percentile 5%"), (50, INK, "Mediane"), (95, GOLD, "Percentile 95%")]:
        ax1.plot(t_axis, np.percentile(allp, q, axis=0), color=col, lw=2, label=l)
    ax1.axhline(spot * prod.autocall.barrier, ls="--", color=GOLD, lw=1.2, label="Barriere de rappel")
    ax1.axhline(spot * prod.capital.barrier, ls="--", color=RED, lw=1.2, label="Barriere de capital")
    ax1.axhline(spot, ls=":", color=GREY, lw=1, label="Niveau initial")
    ax1.set_title("Diffusion Monte Carlo du sous-jacent", fontsize=10, color=INK)
    ax1.set_xlabel("Annees"); ax1.set_ylabel("Niveau (base 100)")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.15)

    # --- Panel 2 : distribution des payoffs terminaux ---
    pf = res["payoff"]
    ax2.hist(pf, bins=50, color=GREEN, alpha=0.65, edgecolor="white", lw=0.3)
    ax2.axvline(1.0, color=GREY, ls=":", lw=1.2, label="Capital (100%)")
    ax2.axvline(np.median(pf), color=INK, ls="-", lw=1.6, label=f"Mediane {np.median(pf):.2f}")
    ax2.axvline(pf.mean(), color=GOLD, ls="-", lw=1.6, label=f"Moyenne {pf.mean():.2f}")
    ax2.axvline(np.percentile(pf, 5), color=RED, ls="--", lw=1.4, label=f"P5 {np.percentile(pf,5):.2f}")
    ax2.set_title("Distribution des payoffs", fontsize=10, color=INK)
    ax2.set_xlabel("Payoff (x nominal)"); ax2.set_ylabel("Frequence")
    ax2.legend(fontsize=7, framealpha=0.9); ax2.grid(alpha=0.15, axis="y")

    # --- Panel 3 : proba de rappel cumulee par constatation ---
    cum = np.cumsum(res["prob_recall"]) * 100
    ax3.fill_between(times, 0, cum, color=GREEN, alpha=0.15)
    ax3.plot(times, cum, color=GREEN, lw=2, marker="o", ms=3, label="Rappel cumule")
    ax3.set_ylim(0, 100)
    ax3.set_title("Proba de rappel cumulee", fontsize=10, color=INK)
    ax3.set_xlabel("Annees"); ax3.set_ylabel("%")
    ax3.legend(fontsize=7, framealpha=0.9); ax3.grid(alpha=0.15)

    fig.tight_layout()
    return fig


# ============================================================
# ONGLETS
# ============================================================
tab_one, tab_cmp, tab_corr = st.tabs(["Analyse d'un produit", "Comparer 2 produits", "Correlation"])

# ----- ONGLET 1 -----
with tab_one:
    n_sims = st.select_slider("Trajectoires", options=[5000, 10000, 20000, 40000], value=20000, key="single_ns")
    prod, vol, div, r, spot, name = product_form("S", dict(title="Produit", vol=0.20, maturity=8.0,
                                                           coupon=0.06, stepdown=True))
    if st.button("Lancer l'analyse", type="primary", key="run_single"):
        cal, res, stats = run_product(prod, vol, div, r, spot, int(n_sims))
        an = ProbabilityAnalyzer()
        s1, s2, s3 = st.tabs(["Synthese", "Probas par date", "Graphiques"])
        with s1:
            c = st.columns(4)
            c[0].metric("Proba de rappel", f"{stats['prob_autocall']*100:.1f}%")
            c[1].metric("Proba de perte", f"{stats['prob_loss']*100:.1f}%")
            c[2].metric("Duree de vie", f"{stats['expected_life']:.2f} ans")
            c[3].metric("Valeur actualisee", f"{stats['fair_value']*100:.1f}%")
            c2 = st.columns(4)
            c2[0].metric("Va au terme", f"{stats['prob_maturity']*100:.1f}%")
            c2[1].metric("Coupons moyens", f"{stats['avg_coupon']:.3f}")
            c2[2].metric("Rendt annualise", f"{stats['ann_return_mean']*100:+.2f}%")
            c2[3].metric("Pire 5% payoff", f"{stats['p5_payoff']:.3f}")
        with s2:
            st.markdown("**Lecture des probabilites** (toutes inconditionnelles sauf indication) :")
            st.caption(
                "Survie a la date = proba d'atteindre la constatation sans rappel anterieur. "
                "Rappel a cette date = proba d'etre rappele pile a cette date. "
                "Rappel sachant vivant = proba conditionnelle de rappel sachant qu'on n'a pas encore ete rappele. "
                "Rappel cumule = proba d'avoir ete rappele a cette date ou avant.")
            df = an.per_date_table(res, start_date=prod.start_date)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         height=min(560, 40 + 28 * len(df)))
            st.download_button("CSV", df.to_csv(index=False).encode("utf-8"), "probas.csv", "text/csv")
        with s3:
            st.pyplot(mc_figure(cal, res, prod, spot))
        st.info("Vol historique, plate, sans smile. Barrieres extremes sous-estimees. Outil CGP interne.")

# ----- ONGLET 2 -----
with tab_cmp:
    st.markdown("Configure deux produits et compare-les cote a cote.")
    n_sims_c = st.select_slider("Trajectoires", options=[5000, 10000, 20000], value=10000, key="cmp_ns")
    colA, colB = st.columns(2)
    with colA:
        pA, vA, dA, rA, spA, nA = product_form("A", dict(title="Produit A", vol=0.20, maturity=8.0,
                                                         coupon=0.06, stepdown=True))
    with colB:
        pB, vB, dB, rB, spB, nB = product_form("B", dict(title="Produit B", vol=0.22, maturity=6.0,
                                                         coupon=0.05, stepdown=False))
    if st.button("Comparer", type="primary", key="run_cmp"):
        calA, resA, stA = run_product(pA, vA, dA, rA, spA, int(n_sims_c))
        calB, resB, stB = run_product(pB, vB, dB, rB, spB, int(n_sims_c))
        rows = [
            ("Proba de rappel", f"{stA['prob_autocall']*100:.1f}%", f"{stB['prob_autocall']*100:.1f}%"),
            ("Proba de perte", f"{stA['prob_loss']*100:.1f}%", f"{stB['prob_loss']*100:.1f}%"),
            ("Va au terme", f"{stA['prob_maturity']*100:.1f}%", f"{stB['prob_maturity']*100:.1f}%"),
            ("Duree de vie esperee", f"{stA['expected_life']:.2f} ans", f"{stB['expected_life']:.2f} ans"),
            ("Rendt annualise", f"{stA['ann_return_mean']*100:+.2f}%", f"{stB['ann_return_mean']*100:+.2f}%"),
            ("Valeur actualisee", f"{stA['fair_value']*100:.1f}%", f"{stB['fair_value']*100:.1f}%"),
            ("Pire 5% (payoff)", f"{stA['p5_payoff']:.3f}", f"{stB['p5_payoff']:.3f}"),
            ("Coupons moyens", f"{stA['avg_coupon']:.3f}", f"{stB['avg_coupon']:.3f}"),
        ]
        dfc = pd.DataFrame(rows, columns=["Critere", f"A — {pA.name}", f"B — {pB.name}"])
        st.dataframe(dfc, use_container_width=True, hide_index=True)
        st.caption("Pas de gagnant unique : le plus remunerateur en esperance n'est pas le mieux protege. "
                   "A arbitrer selon le profil du client.")

        # Graphiques comparatifs (point 6) : distribution + rappel cumule, legendes completes
        figc, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4))
        axL.hist(resA["payoff"], bins=45, alpha=0.55, color=GREEN, label=f"A · {pA.name}", edgecolor="white", lw=0.3)
        axL.hist(resB["payoff"], bins=45, alpha=0.55, color=GOLD, label=f"B · {pB.name}", edgecolor="white", lw=0.3)
        axL.axvline(1.0, color=RED, ls="--", lw=1.2, label="Capital (100%)")
        axL.axvline(resA["payoff"].mean(), color=GREEN, ls=":", lw=1.4, label=f"Moy. A {resA['payoff'].mean():.2f}")
        axL.axvline(resB["payoff"].mean(), color=GOLD, ls=":", lw=1.4, label=f"Moy. B {resB['payoff'].mean():.2f}")
        axL.set_title("Distribution des payoffs", fontsize=10, color=INK)
        axL.set_xlabel("Payoff (x nominal)"); axL.set_ylabel("Frequence")
        axL.legend(fontsize=7, framealpha=0.9); axL.grid(alpha=0.15, axis="y")

        tA = np.asarray(calA.times); tB = np.asarray(calB.times)
        axR.plot(tA, np.cumsum(resA["prob_recall"]) * 100, color=GREEN, lw=2, marker="o", ms=3, label=f"A · {pA.name}")
        axR.plot(tB, np.cumsum(resB["prob_recall"]) * 100, color=GOLD, lw=2, marker="s", ms=3, label=f"B · {pB.name}")
        axR.set_ylim(0, 100)
        axR.set_title("Proba de rappel cumulee", fontsize=10, color=INK)
        axR.set_xlabel("Annees"); axR.set_ylabel("%")
        axR.legend(fontsize=8, framealpha=0.9); axR.grid(alpha=0.15)
        figc.tight_layout()
        st.pyplot(figc)

# ----- ONGLET 3 : CORRELATION (point 7, refonte) -----
with tab_corr:
    st.markdown("Correlation **historique** des rendements quotidiens entre deux sous-jacents.")
    st.caption("Mesure realisee sur le passe. Ce n'est pas la correlation implicite d'un desk, "
               "mais un ordre de grandeur utile pour juger un worst-of.")
    cc1, cc2, cc3 = st.columns(3)
    p1 = cc1.selectbox("Sous-jacent 1", list(PRESET_TICKERS.keys()), key="corr_p1")
    t1 = cc1.text_input("Ticker 1", value=PRESET_TICKERS[p1], key="corr_t1")
    p2 = cc2.selectbox("Sous-jacent 2", list(PRESET_TICKERS.keys()), index=1, key="corr_p2")
    t2 = cc2.text_input("Ticker 2", value=PRESET_TICKERS[p2], key="corr_t2")
    win_c = cc3.selectbox("Fenetre", list(LOOKBACKS.keys()), index=1, key="corr_win")

    if st.button("Calculer la correlation", type="primary", key="run_corr"):
        if t1.strip() and t2.strip():
            with st.spinner("Telechargement des deux historiques..."):
                corr, info, rr = correlation(t1.strip(), t2.strip(), LOOKBACKS[win_c])
            if corr is not None:
                a = rr.iloc[:, 0].to_numpy(); b = rr.iloc[:, 1].to_numpy()
                # Statistiques de regression  b = alpha + beta * a
                beta, alpha = np.polyfit(a, b, 1)
                r2 = corr ** 2
                vol_a = float(np.std(a) * np.sqrt(252))
                vol_b = float(np.std(b) * np.sqrt(252))
                cov_ann = float(np.cov(a, b)[0, 1] * 252)

                m = st.columns(4)
                m[0].metric("Correlation (Pearson)", f"{corr:+.3f}")
                m[1].metric("R²", f"{r2:.3f}")
                m[2].metric("Beta (2 sur 1)", f"{beta:+.3f}")
                m[3].metric("Alpha annualise", f"{alpha*252*100:+.2f}%")
                m2 = st.columns(4)
                m2[0].metric(f"Vol annualisee {t1}", f"{vol_a*100:.1f}%")
                m2[1].metric(f"Vol annualisee {t2}", f"{vol_b*100:.1f}%")
                m2[2].metric("Covariance annualisee", f"{cov_ann:.4f}")
                m2[3].metric("Jours communs", f"{len(rr)}")
                st.success(info)

                lvl = "faible" if abs(corr) < 0.3 else ("moderee" if abs(corr) < 0.7 else "forte")
                st.caption(
                    f"Correlation {lvl}. Pour un worst-of, plus la correlation est faible, plus le coupon "
                    f"offert est eleve (le client vend de la correlation). R² = {r2*100:.1f}% de la variance "
                    f"de {t2} expliquee lineairement par {t1}.")

                figc, (axS, axR2) = plt.subplots(1, 2, figsize=(13, 5),
                                                 gridspec_kw={"width_ratios": [1, 1]})
                # Nuage + droite de regression
                axS.scatter(a * 100, b * 100, s=8, alpha=0.35, color=GREEN)
                xs = np.linspace(a.min(), a.max(), 100)
                axS.plot(xs * 100, (alpha + beta * xs) * 100, color=RED, lw=1.8,
                         label=f"y = {beta:.2f}x {'+' if alpha>=0 else '-'} {abs(alpha)*100:.3f}%")
                axS.axhline(0, color=GREY, lw=0.6); axS.axvline(0, color=GREY, lw=0.6)
                axS.set_xlabel(f"Rendt quotidien {t1} (%)"); axS.set_ylabel(f"Rendt quotidien {t2} (%)")
                axS.set_title(f"Nuage + regression — r = {corr:+.2f}", fontsize=10, color=INK)
                axS.legend(fontsize=8, framealpha=0.9); axS.grid(alpha=0.15)

                # Correlation glissante 60 jours
                roll = rr.iloc[:, 0].rolling(60).corr(rr.iloc[:, 1]).dropna()
                axR2.plot(range(len(roll)), roll.to_numpy(), color=GREEN, lw=1.4)
                axR2.axhline(corr, color=RED, ls="--", lw=1.2, label=f"Moyenne periode {corr:+.2f}")
                axR2.set_ylim(-1, 1)
                axR2.set_xlabel("Jours (fenetre glissante 60j)"); axR2.set_ylabel("Correlation")
                axR2.set_title("Correlation glissante 60 jours", fontsize=10, color=INK)
                axR2.legend(fontsize=8, framealpha=0.9); axR2.grid(alpha=0.15)
                figc.tight_layout()
                st.pyplot(figc)
            else:
                st.warning(info)
        else:
            st.warning("Renseigne les deux tickers.")