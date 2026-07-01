from dataclasses import dataclass
from typing import List
from core.product import StructuredProduct

FREQ_STEP = {"monthly": 1/12, "quarterly": 0.25, "semiannual": 0.5, "annual": 1.0}
FREQ_LABEL = {"monthly":"Mensuelle","quarterly":"Trimestrielle","semiannual":"Semestrielle","annual":"Annuelle"}


@dataclass
class Event:
    index: int            # numero de constatation (1..N)
    time_year: float      # instant en annees
    is_autocall: bool     # rappel observable a cette date ?
    is_maturity: bool


@dataclass
class Calendar:
    events: List[Event]

    @property
    def times(self):
        return [e.time_year for e in self.events]


class CalendarBuilder:
    def build(self, product: StructuredProduct) -> Calendar:
        maturity = float(product.maturity_years)
        step = FREQ_STEP[product.frequency]
        first_call = product.autocall.first_call_year if product.autocall else 0.0

        # grille reguliere jusqu'a maturite incluse
        times = []
        t = step
        while t < maturity - 1e-9:
            times.append(round(t, 6))
            t += step
        times.append(maturity)

        events = []
        for i, t in enumerate(times, start=1):
            is_mat = abs(t - maturity) < 1e-9
            is_auto = bool(product.autocall and product.autocall.enabled and t >= first_call and not is_mat)
            events.append(Event(i, t, is_auto, is_mat))
        return Calendar(events)