import numpy as np
import pandas as pd
from datetime import date, timedelta


class ProbabilityAnalyzer:
    def analyze(self, results: dict):
        status = results["status"]; payoff = results["payoff"]; time = results["time"]
        coupon = results["coupon"]
        ann = payoff ** (1.0 / time) - 1.0
        return {
            "prob_autocall": float(np.mean(status == "autocall")),
            "prob_loss": float(np.mean(status == "loss")),
            "prob_maturity": float(np.mean(status == "maturity")),
            "avg_payoff": float(payoff.mean()),
            "median_payoff": float(np.median(payoff)),
            "p5_payoff": float(np.percentile(payoff, 5)),
            "avg_coupon": float(coupon.mean()),
            "expected_life": float(time.mean()),
            "ann_return_mean": float(ann.mean()),
            "fair_value": float(results["pv"].mean()),
        }

    def _fmt_echeance(self, ty: float) -> str:
        """Annee fractionnaire -> libelle lisible (ex: '2 ans 6 mois')."""
        months = int(round(ty * 12))
        y, m = divmod(months, 12)
        parts = []
        if y: parts.append(f"{y} an" + ("s" if y > 1 else ""))
        if m: parts.append(f"{m} mois")
        return " ".join(parts) if parts else "0 mois"

    def per_date_table(self, results: dict, start_date: str = "") -> pd.DataFrame:
        """
        Tableau date par date, colonnes explicites :
        - Survie a la date      : proba d'ATTEINDRE cette constatation sans rappel anterieur
        - Rappel a cette date   : proba INCONDITIONNELLE d'etre rappele PILE a cette date
        - Rappel sachant vivant : proba CONDITIONNELLE de rappel a cette date sachant non encore rappele
        - Rappel cumule         : proba d'avoir ete rappele a cette date ou avant
        - Sous barr. coupon     : proba que le sous-jacent soit sous la barriere de coupon
        - Sous barr. capital    : proba que le sous-jacent soit sous la barriere de perte en capital
        Toutes les probas (sauf 'sachant vivant') sont inconditionnelles : mesurees sur l'ensemble
        des trajectoires au depart, donc directement additionnables.
        """
        events = results["events"]
        rec   = results["prob_recall"]            # inconditionnelle, pile a la date
        bc    = results["prob_below_coupon"]
        bk    = results["prob_below_capital"]
        alive = results["prob_alive_before"]      # proba d'arriver vivant a la date

        d0 = None
        if start_date:
            try:
                d0 = date.fromisoformat(start_date)
            except Exception:
                d0 = None

        rows = []
        cum = 0.0
        for k, ev in enumerate(events):
            cum += rec[k]
            cond = (rec[k] / alive[k]) if alive[k] > 1e-9 else 0.0   # rappel sachant vivant
            row = {
                "N°": ev.index,
                "Echeance": self._fmt_echeance(ev.time_year),
            }
            if d0 is not None:
                row["Date"] = (d0 + timedelta(days=round(ev.time_year * 365.25))).isoformat()
            row.update({
                "Survie a la date":      f"{alive[k]*100:.1f}%",
                "Rappel a cette date":   f"{rec[k]*100:.1f}%",
                "Rappel sachant vivant": f"{cond*100:.1f}%",
                "Rappel cumule":         f"{cum*100:.1f}%",
                "Sous barr. coupon":     f"{bc[k]*100:.1f}%",
                "Sous barr. capital":    f"{bk[k]*100:.1f}%",
            })
            rows.append(row)
        return pd.DataFrame(rows)