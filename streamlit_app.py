import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from core.product import StructuredProduct, Underlying, Coupon, Autocall, CapitalProtection
from core.calendar import CalendarBuilder, FREQ_LABEL, FREQ_STEP
from simulation.product_engine import ProductEngine
from analytics.probabilities import ProbabilityAnalyzer
from market.data_provider import PRESET_TICKERS, correlation
from ui_underlying import underlying_form, build_simulator, LOOKBACKS

st.set_page_config(page_title="Autocall Analyzer", layout="wide")
st.title("Autocall Analyzer")
st.caption("Outil d'aide a la decision pour CGP. Aucune donnee client.")

PRESET_KEYS = list(PRESET_TICKERS.keys())

NATURES = ["conditional", "accrued", "guaranteed"]
NATURE_LABEL = {
    "conditional": "Phoenix (coupon conditionnel en cours de vie)",
    "accrued": "Athena (coupon capitalise, verse a la sortie)",
    "guaranteed": "Garanti",
}

# Palette institutionnelle
GREEN = "#2E6356"; GOLD = "#977B3C"; RED = "#A83838"; INK = "#15191D"; GREY = "#9AA0A6"


# ==================================================================
# Helpers
# ==================================================================
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def num5(container, label, key, default, step=0.00001, disabled=False):
    """
    number_input a 5 decimales pilote UNIQUEMENT par session_state.

    Des qu'un widget a un `key`, Streamlit ignore l'argument `value=` aux reruns
    suivants : session_state fait autorite. On initialise donc la cle en amont et
    toute mise a jour externe ecrit directement dedans, depuis un callback.
    """
    init_state(key, float(default))
    return container.number_input(label, step=step, format="%.5f", key=key, disabled=disabled)


def show_fig(fig):
    """Affiche puis libere la figure : sans close, matplotlib fuit a chaque rerun."""
    st.pyplot(fig)
    plt.close(fig)


def measure_selector(prefix: str):
    """
    Choix de la mesure de probabilite.

    Le drift risque-neutre est correct pour valoriser, mais les probabilites qui
    en decoulent incorporent la prime de risque actions : elles sont pessimistes
    par construction. Sur un 8 ans, l'ecart avec la mesure reelle atteint
    facilement plusieurs points de probabilite de perte.
    """
    init_state(f"{prefix}_measure", "Risque-neutre (valorisation)")
    m = st.radio("Mesure de probabilite",
                 ["Risque-neutre (valorisation)", "Reelle (scenario client)"],
                 horizontal=True, key=f"{prefix}_measure")
    if m.startswith("Reelle"):
        init_state(f"{prefix}_mu", 7.0)
        mu = st.number_input("Rendement attendu du sous-jacent (%/an)", step=0.5,
                             format="%.2f", key=f"{prefix}_mu",
                             help="Taux sans risque + prime de risque actions. "
                                  "4 a 6 points au-dessus du taux sans risque est un "
                                  "ordre de grandeur usuel pour un indice actions large.")
        return float(mu) / 100.0
    return None


# ==================================================================
# Formulaire produit
# ==================================================================
def product_form(prefix: str, defaults: dict):
    """Formulaire produit. Retourne (product, spec, r, spot, name)."""
    st.markdown(f"#### {defaults['title']}")

    spec = underlying_form(prefix, default_vol=defaults["vol"])
    name = spec.get("name") or "Sous-jacent"

    st.divider()
    c3, c4 = st.columns(2)
    spot = num5(c3, "Niveau initial", f"{prefix}_spot", 100.0, step=1.0)
    r = num5(c4, "Taux sans risque", f"{prefix}_r", 0.025)
    c5, c6 = st.columns(2)
    init_state(f"{prefix}_mat", float(defaults["maturity"]))
    maturity = c5.number_input("Maturite (ans)", step=0.25, format="%.2f", key=f"{prefix}_mat")
    init_state(f"{prefix}_freq", list(FREQ_LABEL.keys())[3])
    frequency = c6.selectbox("Constatation", list(FREQ_LABEL.keys()),
                             format_func=lambda k: FREQ_LABEL[k], key=f"{prefix}_freq")
    start = st.date_input("Date de lancement", value=date.today(), key=f"{prefix}_start")

    st.caption("Coupon")
    c7, c8 = st.columns(2)
    crate_annual = num5(c7, "Coupon annuel", f"{prefix}_cr", defaults["coupon"])
    init_state(f"{prefix}_cnat", "conditional")
    cnat = c8.selectbox("Nature", NATURES, format_func=lambda x: NATURE_LABEL[x],
                        key=f"{prefix}_cnat")
    step = FREQ_STEP[frequency]
    crate_period = crate_annual * step

    if cnat == "accrued":
        n_per = max(int(round(float(maturity) / step)), 1)
        st.caption(f"Athena : rien n'est verse en cours de vie. A la sortie, le porteur "
                   f"recoit 100% + {crate_period*100:.3f}% par constatation ecoulee. "
                   f"Gain maximal au terme : {(1 + crate_period*n_per)*100:.1f}%. "
                   f"En cas de perte au terme, aucun coupon n'est verse.")
    else:
        st.caption(f"Soit {crate_period*100:.3f}% verse a chaque constatation "
                   f"({FREQ_LABEL[frequency].lower()}, {round(1/step)}/an).")

    c9, c10 = st.columns(2)
    cbar = num5(c9, "Barr. coupon", f"{prefix}_cbar", 0.70, step=0.01,
                disabled=(cnat != "conditional"))
    cmem = c10.checkbox("Memoire", value=True, disabled=(cnat != "conditional"),
                        key=f"{prefix}_cmem")

    st.caption("Rappel & protection")
    c11, c12 = st.columns(2)
    abar = num5(c11, "Barr. rappel", f"{prefix}_abar", 1.00, step=0.01)
    init_state(f"{prefix}_fc", 1.0)
    fcall = c12.number_input("1re constat. (an)", step=0.25, format="%.2f", key=f"{prefix}_fc")
    c13, c14 = st.columns(2)
    usd = c13.checkbox("Step-down", value=defaults["stepdown"], key=f"{prefix}_usd")
    sd = num5(c14, "Step-down pts/an", f"{prefix}_sd", 0.05, step=0.01, disabled=not usd)
    kbar = num5(st, "Barr. protection capital", f"{prefix}_kbar", 0.60, step=0.01)

    product = StructuredProduct(
        name=f"Autocall {name}", maturity_years=float(maturity),
        underlying=Underlying(name, spec.get("ticker", ""), spot),
        frequency=frequency, start_date=start.isoformat(),
        coupon=Coupon(crate_period, cbar, cmem, cnat),
        autocall=Autocall(True, abar, fcall, sd if usd else 0.0),
        capital=CapitalProtection(kbar, True),
    )
    return product, spec, r, spot, name


def run_product(product, spec, r, spot, n_sims, seed=None, drift=None):
    cal = CalendarBuilder().build(product)
    sim = build_simulator(spec, spot=spot, r=r, seed=seed, drift=drift)
    res = ProductEngine(product, sim).run(cal, n_simulations=n_sims)
    return cal, res, ProbabilityAnalyzer().analyze(res)


def mc_error(p: float, n: int) -> float:
    """Demi-largeur de l'intervalle de confiance a 95% d'une proportion simulee."""
    return 1.96 * float(np.sqrt(max(p * (1 - p), 0.0) / n))


# ==================================================================
# Graphiques Monte Carlo
# ==================================================================
def mc_figure(cal, res, prod, spot):
    times = np.asarray(cal.times)
    t_axis = np.concatenate([[0.0], times])
    paths = res["paths"]
    allp = np.concatenate([np.full((len(paths), 1), spot), paths], axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2),
                                        gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # --- Panel 1 : faisceau + percentiles + barrieres ---
    samp = paths[np.random.choice(len(paths), size=min(300, len(paths)), replace=False)]
    for row in samp:
        ax1.plot(t_axis, np.concatenate([[spot], row]), color=GREEN, alpha=0.05, lw=0.7)
    ax1.plot([], [], color=GREEN, alpha=0.5, lw=1, label="Trajectoires simulees")
    for q, col, l in [(5, RED, "Percentile 5%"), (50, INK, "Mediane"), (95, GOLD, "Percentile 95%")]:
        ax1.plot(t_axis, np.percentile(allp, q, axis=0), color=col, lw=2, label=l)

    # Barriere de rappel EN ESCALIER : avec un step-down elle decroit, une
    # horizontale donnait un visuel faux des que la case etait cochee.
    bp = res.get("barrier_path")
    if bp is not None and np.any(~np.isnan(bp)):
        ok = ~np.isnan(bp)
        ax1.step(times[ok], bp[ok] * spot, where="post", color=GOLD, ls="--", lw=1.5,
                 label="Barriere de rappel")
    else:
        ax1.axhline(spot * prod.autocall.barrier, ls="--", color=GOLD, lw=1.2,
                    label="Barriere de rappel")
    ax1.axhline(spot * prod.capital.barrier, ls="--", color=RED, lw=1.2,
                label="Barriere de capital")
    ax1.axhline(spot, ls=":", color=GREY, lw=1, label="Niveau initial")
    ax1.set_title("Diffusion Monte Carlo du sous-jacent", fontsize=10, color=INK)
    ax1.set_xlabel("Annees"); ax1.set_ylabel("Niveau (base 100)")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.15)

    # --- Panel 2 : payoffs VENTILES PAR ISSUE ---
    # La distribution est fortement multimodale (une masse par date de rappel).
    # Un histogramme unique melange trois situations que le client doit distinguer.
    pf = np.asarray(res["payoff"], dtype=float)
    stt = np.array([str(s) for s in res["status"]])
    groups = [("autocall", "Rappele", GREEN), ("maturity", "Terme sans perte", GOLD),
              ("loss", "Perte en capital", RED)]
    data, labs, cols = [], [], []
    for k, lab, col in groups:
        sel = stt == k
        if sel.any():
            data.append(pf[sel]); labs.append(f"{lab} ({sel.mean()*100:.0f}%)"); cols.append(col)
    if data:
        ax2.hist(data, bins=40, stacked=True, color=cols, label=labs,
                 edgecolor="white", lw=0.3)
    ax2.axvline(1.0, color=GREY, ls=":", lw=1.2, label="Capital (100%)")
    ax2.axvline(np.percentile(pf, 5), color=INK, ls="--", lw=1.4,
                label=f"P5 {np.percentile(pf,5):.2f}")
    ax2.set_title("Payoffs nominaux par issue", fontsize=10, color=INK)
    ax2.set_xlabel("Payoff (x nominal)"); ax2.set_ylabel("Frequence")
    ax2.legend(fontsize=7, framealpha=0.9); ax2.grid(alpha=0.15, axis="y")

    # --- Panel 3 : rappel cumule + survie ---
    cum = np.cumsum(res["prob_recall"]) * 100
    ax3.fill_between(times, 0, cum, color=GREEN, alpha=0.15)
    ax3.plot(times, cum, color=GREEN, lw=2, marker="o", ms=3, label="Rappel cumule")
    ax3.plot(times, res["prob_alive_before"] * 100, color=GREY, lw=1.4, ls="--",
             label="Encore vivant")
    ax3.set_ylim(0, 100)
    ax3.set_title("Rappel cumule et survie", fontsize=10, color=INK)
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
    cns, cseed = st.columns([3, 1])
    n_sims = cns.select_slider("Trajectoires", options=[5000, 10000, 20000, 40000],
                               value=20000, key="single_ns")
    fix_seed = cseed.checkbox("Graine fixe", value=True, key="single_seed_on",
                              help="Resultats reproductibles, necessaire pour tracer "
                                   "un devoir de conseil.")
    drift = measure_selector("single")

    prod, spec, r, spot, name = product_form("S", dict(title="Produit", vol=0.20, maturity=8.0,
                                                       coupon=0.06, stepdown=True))

    if not spec.get("ready", True):
        st.warning("Calibre d'abord le sous-jacent.")
    elif st.button("Lancer l'analyse", type="primary", key="run_single"):
        with st.spinner("Simulation..."):
            cal, res, stats = run_product(prod, spec, r, spot, int(n_sims),
                                          seed=42 if fix_seed else None, drift=drift)
        an = ProbabilityAnalyzer()
        n = int(n_sims)

        if drift is None:
            st.warning("Mesure risque-neutre : la valeur actualisee est correcte, mais les "
                       "probabilites affichees sont pessimistes car elles integrent la prime "
                       "de risque actions. Ne pas les presenter telles quelles a un client.")
        else:
            st.warning(f"Mesure reelle (rendement attendu {drift*100:.2f}%/an) : les "
                       "probabilites sont exploitables, mais la valeur actualisee n'a plus "
                       "de sens dans ce mode. Elle est masquee.")

        s1, s2, s3 = st.tabs(["Synthese", "Probas par date", "Graphiques"])
        with s1:
            c = st.columns(4)
            pa = stats['prob_autocall']; pl = stats['prob_loss']
            c[0].metric("Proba de rappel", f"{pa*100:.1f}%",
                        f"± {mc_error(pa, n)*100:.2f} pt", delta_color="off")
            c[1].metric("Proba de perte", f"{pl*100:.1f}%",
                        f"± {mc_error(pl, n)*100:.2f} pt", delta_color="off")
            c[2].metric("Duree de vie", f"{stats['expected_life']:.2f} ans")
            if drift is None:
                se_fv = 1.96 * float(np.std(res["pv"]) / np.sqrt(n))
                c[3].metric("Valeur actualisee", f"{stats['fair_value']*100:.1f}%",
                            f"± {se_fv*100:.2f} pt", delta_color="off")
            else:
                c[3].metric("Valeur actualisee", "n/a")
            c2 = st.columns(4)
            pm = stats['prob_maturity']
            c2[0].metric("Va au terme", f"{pm*100:.1f}%",
                         f"± {mc_error(pm, n)*100:.2f} pt", delta_color="off")
            c2[1].metric("Coupons moyens", f"{stats['avg_coupon']:.3f}")
            c2[2].metric("Rendt annualise", f"{stats['ann_return_mean']*100:+.2f}%")
            c2[3].metric("Pire 5% payoff", f"{stats['p5_payoff']:.3f}")
            st.caption("Les marges sont des intervalles de confiance a 95% dus au seul bruit "
                       "Monte Carlo. Elles ne disent rien de l'erreur de modele, qui est "
                       "bien plus grande.")
        with s2:
            st.markdown("**Lecture des probabilites** (toutes inconditionnelles sauf indication) :")
            st.caption(
                "Survie a la date = proba d'atteindre la constatation sans rappel anterieur. "
                "Rappel a cette date = proba d'etre rappele pile a cette date. "
                "Rappel sachant vivant = proba conditionnelle de rappel sachant qu'on n'a pas "
                "encore ete rappele. Rappel cumule = proba d'avoir ete rappele a cette date ou avant.")
            df = an.per_date_table(res, start_date=prod.start_date)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         height=min(560, 40 + 28 * len(df)))
            st.download_button("CSV", df.to_csv(index=False).encode("utf-8"),
                               "probas.csv", "text/csv")
        with s3:
            show_fig(mc_figure(cal, res, prod, spot))
        st.info("Volatilite historique, plate, sans smile. Les barrieres profondes sont donc "
                "sous-estimees : un put a 60% se traite en pratique a une vol implicite bien "
                "superieure a la vol ATM. Outil CGP interne, ne remplace pas une valorisation "
                "emetteur.")

# ----- ONGLET 2 -----
with tab_cmp:
    st.markdown("Configure deux produits et compare-les cote a cote.")
    n_sims_c = st.select_slider("Trajectoires", options=[5000, 10000, 20000],
                                value=10000, key="cmp_ns")
    drift_c = measure_selector("cmp")

    colA, colB = st.columns(2)
    with colA:
        pA, specA, rA, spA, nA = product_form("A", dict(title="Produit A", vol=0.20,
                                                        maturity=8.0, coupon=0.06, stepdown=True))
    with colB:
        pB, specB, rB, spB, nB = product_form("B", dict(title="Produit B", vol=0.22,
                                                        maturity=6.0, coupon=0.05, stepdown=False))

    if not (specA.get("ready", True) and specB.get("ready", True)):
        st.warning("Calibre les deux sous-jacents.")
    elif st.button("Comparer", type="primary", key="run_cmp"):
        with st.spinner("Simulation..."):
            # Meme graine pour A et B : l'ecart mesure vient des parametres, pas du bruit.
            calA, resA, stA = run_product(pA, specA, rA, spA, int(n_sims_c), seed=42, drift=drift_c)
            calB, resB, stB = run_product(pB, specB, rB, spB, int(n_sims_c), seed=42, drift=drift_c)

        n = int(n_sims_c)
        rows = [
            ("Proba de rappel", f"{stA['prob_autocall']*100:.1f}%", f"{stB['prob_autocall']*100:.1f}%"),
            ("Proba de perte", f"{stA['prob_loss']*100:.1f}%", f"{stB['prob_loss']*100:.1f}%"),
            ("Va au terme", f"{stA['prob_maturity']*100:.1f}%", f"{stB['prob_maturity']*100:.1f}%"),
            ("Duree de vie esperee", f"{stA['expected_life']:.2f} ans", f"{stB['expected_life']:.2f} ans"),
            ("Rendt annualise", f"{stA['ann_return_mean']*100:+.2f}%", f"{stB['ann_return_mean']*100:+.2f}%"),
            ("Pire 5% (payoff)", f"{stA['p5_payoff']:.3f}", f"{stB['p5_payoff']:.3f}"),
            ("Coupons moyens", f"{stA['avg_coupon']:.3f}", f"{stB['avg_coupon']:.3f}"),
        ]
        if drift_c is None:
            rows.append(("Valeur actualisee", f"{stA['fair_value']*100:.1f}%",
                         f"{stB['fair_value']*100:.1f}%"))
        dfc = pd.DataFrame(rows, columns=["Critere", f"A - {pA.name}", f"B - {pB.name}"])
        st.dataframe(dfc, use_container_width=True, hide_index=True)

        marge = mc_error(0.5, n) * 100
        st.caption(f"Pas de gagnant unique : le plus remunerateur en esperance n'est pas le "
                   f"mieux protege. A {n} trajectoires, un ecart de probabilite inferieur a "
                   f"environ {marge:.1f} point n'est pas significatif.")

        figc, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4))
        axL.hist(resA["payoff"], bins=45, alpha=0.55, color=GREEN, label=f"A - {pA.name}",
                 edgecolor="white", lw=0.3)
        axL.hist(resB["payoff"], bins=45, alpha=0.55, color=GOLD, label=f"B - {pB.name}",
                 edgecolor="white", lw=0.3)
        axL.axvline(1.0, color=RED, ls="--", lw=1.2, label="Capital (100%)")
        axL.set_title("Distribution des payoffs", fontsize=10, color=INK)
        axL.set_xlabel("Payoff (x nominal)"); axL.set_ylabel("Frequence")
        axL.legend(fontsize=7, framealpha=0.9); axL.grid(alpha=0.15, axis="y")

        tA = np.asarray(calA.times); tB = np.asarray(calB.times)
        axR.plot(tA, np.cumsum(resA["prob_recall"]) * 100, color=GREEN, lw=2, marker="o",
                 ms=3, label=f"A - {pA.name}")
        axR.plot(tB, np.cumsum(resB["prob_recall"]) * 100, color=GOLD, lw=2, marker="s",
                 ms=3, label=f"B - {pB.name}")
        axR.set_ylim(0, 100)
        axR.set_title("Proba de rappel cumulee", fontsize=10, color=INK)
        axR.set_xlabel("Annees"); axR.set_ylabel("%")
        axR.legend(fontsize=8, framealpha=0.9); axR.grid(alpha=0.15)
        figc.tight_layout()
        show_fig(figc)

# ----- ONGLET 3 : CORRELATION -----
with tab_corr:
    st.markdown("Correlation **historique** des rendements quotidiens entre deux sous-jacents.")
    st.caption("Mesure realisee sur le passe. Ce n'est pas la correlation implicite d'un desk, "
               "mais un ordre de grandeur utile pour juger un worst-of.")

    def _on_corr_preset(slot: str):
        p = st.session_state[f"corr_p{slot}"]
        if p != "Personnalise":
            st.session_state[f"corr_t{slot}"] = PRESET_TICKERS[p]

    init_state("corr_p1", PRESET_KEYS[0])
    init_state("corr_t1", PRESET_TICKERS[st.session_state["corr_p1"]])
    init_state("corr_p2", PRESET_KEYS[1])
    init_state("corr_t2", PRESET_TICKERS[st.session_state["corr_p2"]])

    cc1, cc2, cc3 = st.columns(3)
    cc1.selectbox("Sous-jacent 1", PRESET_KEYS, key="corr_p1",
                  on_change=_on_corr_preset, args=("1",))
    t1 = cc1.text_input("Ticker 1", key="corr_t1")
    cc2.selectbox("Sous-jacent 2", PRESET_KEYS, key="corr_p2",
                  on_change=_on_corr_preset, args=("2",))
    t2 = cc2.text_input("Ticker 2", key="corr_t2")
    win_c = cc3.selectbox("Fenetre", list(LOOKBACKS.keys()), index=1, key="corr_win")

    if st.button("Calculer la correlation", type="primary", key="run_corr"):
        if t1.strip() and t2.strip():
            with st.spinner("Telechargement des deux historiques..."):
                corr, info, rr = correlation(t1.strip(), t2.strip(), LOOKBACKS[win_c])
            if corr is not None:
                a = rr.iloc[:, 0].to_numpy(); b = rr.iloc[:, 1].to_numpy()
                beta, alpha = np.polyfit(a, b, 1)
                r2 = corr ** 2
                vol_a = float(np.std(a) * np.sqrt(252))
                vol_b = float(np.std(b) * np.sqrt(252))
                cov_ann = float(np.cov(a, b)[0, 1] * 252)

                m = st.columns(4)
                m[0].metric("Correlation (Pearson)", f"{corr:+.3f}")
                m[1].metric("R2", f"{r2:.3f}")
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
                    f"Correlation {lvl}. Pour un worst-of, plus la correlation est faible, plus "
                    f"le coupon offert est eleve (le client vend de la correlation). "
                    f"R2 = {r2*100:.1f}% de la variance de {t2} expliquee lineairement par {t1}. "
                    f"La correlation glissante ci-dessous montre a quel point ce chiffre unique "
                    f"est instable, et elle monte vers 1 dans les phases de stress.")

                figr, (axS, axR2) = plt.subplots(1, 2, figsize=(13, 5))
                axS.scatter(a * 100, b * 100, s=8, alpha=0.35, color=GREEN)
                xs = np.linspace(a.min(), a.max(), 100)
                axS.plot(xs * 100, (alpha + beta * xs) * 100, color=RED, lw=1.8,
                         label=f"y = {beta:.2f}x {'+' if alpha>=0 else '-'} {abs(alpha)*100:.3f}%")
                axS.axhline(0, color=GREY, lw=0.6); axS.axvline(0, color=GREY, lw=0.6)
                axS.set_xlabel(f"Rendt quotidien {t1} (%)"); axS.set_ylabel(f"Rendt quotidien {t2} (%)")
                axS.set_title(f"Nuage + regression - r = {corr:+.2f}", fontsize=10, color=INK)
                axS.legend(fontsize=8, framealpha=0.9); axS.grid(alpha=0.15)

                roll = rr.iloc[:, 0].rolling(60).corr(rr.iloc[:, 1]).dropna()
                axR2.plot(range(len(roll)), roll.to_numpy(), color=GREEN, lw=1.4)
                axR2.axhline(corr, color=RED, ls="--", lw=1.2, label=f"Moyenne periode {corr:+.2f}")
                axR2.set_ylim(-1, 1)
                axR2.set_xlabel("Jours (fenetre glissante 60j)"); axR2.set_ylabel("Correlation")
                axR2.set_title("Correlation glissante 60 jours", fontsize=10, color=INK)
                axR2.legend(fontsize=8, framealpha=0.9); axR2.grid(alpha=0.15)
                figr.tight_layout()
                show_fig(figr)
            else:
                st.warning(info)
        else:
            st.warning("Renseigne les deux tickers.")
