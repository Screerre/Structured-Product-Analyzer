import numpy as np
from core.calendar import Calendar
from core.product import StructuredProduct


class ProductEngine:
    """
    Autocall single stock, vectorise. Probas date par date + actualisation des flux.

    Le simulateur est injecte et doit exposer :
        .spot, .r, .simulate_paths(times, n_sims) -> ndarray (n_sims, len(times))
    MarketSimulator et BasketSimulator respectent ce contrat, le moteur est donc
    indifferent au fait que le sous-jacent soit un ticker unique ou un panier
    reconstitue.

    LE DECREMENT N'EST PLUS TRAITE ICI. Il l'etait ainsi :

        ratio = ratio * ((1.0 - p.decrement) ** t_arr)[None, :]

    Deux defauts. C'etait un decrement en POURCENTAGE alors que le marche francais
    est majoritairement en POINTS fixes, ce qui n'a pas le meme profil de risque :
    50 points coutent 5% a un indice a 1000 et 7,1% a un indice a 700, donc le
    prelevement se durcit precisement dans les scenarios ou la barriere de capital
    est menacee. Et surtout il s'ajoutait au dividende q deja retranche dans le
    drift du GBM, alors que dans un indice decrement le prelevement synthetique
    REMPLACE le dividende. Le drag etait donc compte deux fois.

    Le decrement est desormais applique dans le simulateur, sur une grille fine
    (il est path-dependent) et avec la bonne convention. Voir simulation/basket.py.
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

        status = np.empty(n, dtype=object)
        payoff = np.zeros(n)         # flux nominaux (non actualises)
        pv = np.zeros(n)             # flux ACTUALISES (present value)
        call_time = np.full(n, maturity)
        coupons = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        memory_count = np.zeros(n, dtype=int)

        n_dates = len(calendar.events)
        prob_recall = np.zeros(n_dates)
        prob_below_coupon = np.zeros(n_dates)
        prob_below_capital = np.zeros(n_dates)
        prob_alive_before = np.zeros(n_dates)
        barrier_path = np.full(n_dates, np.nan)   # barriere de rappel effective par date

        for k, ev in enumerate(calendar.events):
            t = ev.time_year
            w = ratio[:, k]
            disc = np.exp(-r * t)

            prob_alive_before[k] = alive.mean()
            if coup: prob_below_coupon[k] = float(np.mean(w < coup.barrier))
            if cap:  prob_below_capital[k] = float(np.mean(w < cap.barrier))

            # COUPON
            if coup:
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

            # AUTOCALL
            if ev.is_autocall:
                bar = max(auto.barrier - auto.step_down * (t - first_call), 0.0)
                barrier_path[k] = bar
                called = alive & (w >= bar)
                prob_recall[k] = called.mean()
                status[called] = "autocall"
                payoff[called] = 1.0 + coupons[called]
                pv[called] += 1.0 * disc
                call_time[called] = t
                alive[called] = False

            # MATURITE
            if ev.is_maturity:
                wT = ratio[:, k]; surv = alive
                if cap:
                    protected = surv & (wT >= cap.barrier)
                    lost = surv & (wT < cap.barrier)
                else:
                    protected = surv; lost = np.zeros(n, dtype=bool)
                status[protected] = "maturity"; payoff[protected] = 1.0 + coupons[protected]
                pv[protected] += 1.0 * disc
                status[lost] = "loss"; payoff[lost] = wT[lost] + coupons[lost]
                pv[lost] += wT[lost] * disc
                alive[:] = False

        return {
            "status": status, "payoff": payoff, "pv": pv, "time": call_time, "coupon": coupons,
            "paths": prices, "ratio": ratio, "events": calendar.events,
            "prob_recall": prob_recall, "prob_below_coupon": prob_below_coupon,
            "prob_below_capital": prob_below_capital, "prob_alive_before": prob_alive_before,
            "barrier_path": barrier_path, "n_sims": n,
        }
