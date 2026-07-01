from dataclasses import dataclass, field
from typing import Optional, Dict, Literal


# =========================
# Sous-jacent unique
# =========================
@dataclass
class Underlying:
    name: str                       # nom lisible (ex: "Euro Stoxx 50")
    ticker: str = ""                # ticker yfinance (ex: "^STOXX50E")
    initial_price: float = 100.0    # niveau de reference (base 100)


# =========================
# Coupon
# =========================
@dataclass
class Coupon:
    rate: float                                   # par constatation
    barrier: float = 0.70                         # ignoree si garanti
    memory: bool = False                          # INTERRUPTEUR memoire
    nature: Literal["conditional", "guaranteed"] = "conditional"  # INTERRUPTEUR nature


# =========================
# Autocall
# =========================
@dataclass
class Autocall:
    enabled: bool = True
    barrier: float = 1.00                         # % du niveau initial
    first_call_year: float = 1.0                  # INTERRUPTEUR periode de non-rappel
    step_down: float = 0.0                        # INTERRUPTEUR step-down (pts/an)


# =========================
# Protection capital (europeenne)
# =========================
@dataclass
class CapitalProtection:
    barrier: float = 0.60
    european: bool = True


# =========================
# Produit structure (SINGLE STOCK)
# =========================
@dataclass
class StructuredProduct:
    name: str = "autocall_single"
    maturity_years: float = 5.0
    underlying: Underlying = None
    frequency: Literal["monthly","quarterly","semiannual","annual"] = "annual"  # constatation
    start_date: str = ""                          # date de lancement (info / calendrier)
    coupon: Optional[Coupon] = None
    autocall: Optional[Autocall] = None
    capital: Optional[CapitalProtection] = None
    decrement: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "underlying": self.underlying.name if self.underlying else None,
            "ticker": self.underlying.ticker if self.underlying else None,
            "maturity_years": self.maturity_years,
            "frequency": self.frequency,
            "start_date": self.start_date,
            "coupon_rate": self.coupon.rate if self.coupon else None,
            "coupon_nature": self.coupon.nature if self.coupon else None,
            "coupon_memory": self.coupon.memory if self.coupon else None,
            "coupon_barrier": self.coupon.barrier if self.coupon else None,
            "autocall_barrier": self.autocall.barrier if self.autocall else None,
            "step_down": self.autocall.step_down if self.autocall else None,
            "first_call": self.autocall.first_call_year if self.autocall else None,
            "capital_barrier": self.capital.barrier if self.capital else None,
            "decrement": self.decrement,
        }