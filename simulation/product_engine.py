import numpy as np
from core.calendar import Calendar
from core.product import StructuredProduct


class ProductEngine:
    """
    Autocall vectorise. Probas date par date + actualisation des flux.

    Le simulateur est injecte et doit exposer :
        .spot, .r, .simulate_paths(times, n_sims) -> ndarray (n_sims, len(times))
    MarketSimulator et BasketSimulator respectent ce contrat.

    TROIS NATURES DE COUPON
    -----------------------
    "guaranteed"  Coupon verse a chaque constatation, sans condition.

    "conditional" PHOENIX. Coupon verse a chaque constatation si le sous-jacent
                  est au-dessus de la barriere coupon. Avec effet memoire, les
                  coupons manques sont rattrapes a la premiere constatation
                  au-dessus de la barriere. Le coupon est encaisse en cours de
                  vie et reste acquis meme si le produit finit en perte.

    "accrued"     ATHENA. Rien n'est verse en cours de vie. Le coupon est
                  capitalise et n'est verse qu'a la SORTIE : rappel anticipe, ou
                  terme si le sous-jacent est au-dessus de la barriere de capital.
                  En cas de perte au terme, AUCUN coupon n'est verse.

                  Ne pas simuler un Athena avec un Phoenix a barriere quasi nulle :
                  les trajectoires perdantes conserveraient les coupons encaisses
                  en route, ce qui transforme le pire scenario en gain. Sur un
                  8 ans a 11%, une trajectoire finissant a 40% ressortirait a 1,28
                  au lieu de 0,40.

    LE DECREMENT N'EST PLUS TRAITE ICI. Il est applique dans le simulateur, sur
    une grille fine et en points (voir simulation/basket.py).
    """

    def __init__(self, product: StructuredProduct, simulator):
        self.product = product
        self.simulator = simulator

    def run(self, calendar: Calendar, n_simulations: int = 20000):
        p = self.product
        times = calendar.times
        prices = self.simulator.simulate_paths(times, n_simulations)
        ratio = prices / self.simulator.spot
        r = self.simulator.r          # taux d'ACTUALISATION, distinct du drift

        n = n_simulations
        maturity = float(p.maturity_years)
        coup = p.coupon; auto = p.autocall; cap = p.capital
        first_call = auto.first_call_year if auto else 0.0
        accrued = bool(coup) and getattr(coup, "nature", "") == "accrued"

        status = np.empty(n, dtype=object)
        payoff = np.zeros(n)         # flux nominaux (non actualises)
        pv = np.zeros(n)             # flux ACTUALISES
        call_time = np.full(n, maturity)
        coupons = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        memory_count = np.zeros(n, dtype=int)

        n_dates = len(calendar.events)
        prob_recall = np.zeros(n_dates)
        prob_below_coupon = np.zeros(n_dates)
        prob_below_capital = np.zeros(n_dates)
        prob_alive_before = np.zeros(n_dates)
        barrier_path = np.full(n_dates, np.nan)

        for k, ev in enumerate(calendar.events):
            t = ev.time_year
            w = ratio[:, k]
            disc = np.exp(-r * t)
            n_periods = k + 1                       # constatations ecoulees
            acc = coup.rate * n_periods if accrued else 0.0

            prob_alive_before[k] = alive.mean()
            if coup: prob_below_coupon[k] = float(np.mean(w < coup.barrier))
            if cap:  prob_below_capital[k] = float(np.mean(w < cap.barrier))

            # ---------------- COUPON en cours de vie ----------------
            if coup and not accrued:
                if coup.nature == "guaranteed":
                    coupons[alive] += coup.rate
                    pv[alive] += coup.rate * disc
                else:
                    hit = alive & (w >= coup.barrier)
                    if coup.memory:
                        add = coup.rate * (1 + memory_count)
                        coupons[hit] += add[hit]
                        pv[hit] += add[hit] * disc
                        memory_count[hit] = 0
                        memory_count[alive & (w < coup.barrier)] += 1
                    else:
                        coupons[hit] += coup.rate
                        pv[hit] += coup.rate * disc
            # En mode "accrued" on ne verse rien ici : le coupon est capitalise
            # et sera regle en une fois a la sortie, actualise a la bonne date.

            # ---------------- AUTOCALL ----------------
            if ev.is_autocall:
                bar = max(auto.barrier - auto.step_down * (t - first_call), 0.0)
                barrier_path[k] = bar
                called = alive & (w >= bar)
                prob_recall[k] = called.mean()
                status[called] = "autocall"
                call_time[called] = t

                if accrued:
                    coupons[called] = acc
                    payoff[called] = 1.0 + acc
                    pv[called] += (1.0 + acc) * disc
                else:
                    payoff[called] = 1.0 + coupons[called]
                    pv[called] += 1.0 * disc

                alive[called] = False

            # ---------------- MATURITE ----------------
            if ev.is_maturity:
                wT = ratio[:, k]; surv = alive
                if cap:
                    protected = surv & (wT >= cap.barrier)
                    lost = surv & (wT < cap.barrier)
                else:
                    protected = surv; lost = np.zeros(n, dtype=bool)

                status[protected] = "maturity"
                status[lost] = "loss"

                if accrued:
                    coupons[protected] = acc
                    payoff[protected] = 1.0 + acc
                    pv[protected] += (1.0 + acc) * disc
                    # Perte au terme : le coupon capitalise n'est jamais verse.
                    coupons[lost] = 0.0
                    payoff[lost] = wT[lost]
                    pv[lost] += wT[lost] * disc
                else:
                    payoff[protected] = 1.0 + coupons[protected]
                    pv[protected] += 1.0 * disc
                    payoff[lost] = wT[lost] + coupons[lost]
                    pv[lost] += wT[lost] * disc

                alive[:] = False

        return {
            "status": status, "payoff": payoff, "pv": pv, "time": call_time, "coupon": coupons,
            "paths": prices, "ratio": ratio, "events": calendar.events,
            "prob_recall": prob_recall, "prob_below_coupon": prob_below_coupon,
            "prob_below_capital": prob_below_capital, "prob_alive_before": prob_alive_before,
            "barrier_path": barrier_path, "n_sims": n,
        }
