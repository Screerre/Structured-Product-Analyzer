from core.product import StructuredProduct, Basket, Underlying, Coupon, Autocall, CapitalProtection
from core.calendar import CalendarBuilder
from simulation.montecarlo import MarketSimulator
from simulation.product_engine import ProductEngine
from analytics.probabilities import ProbabilityAnalyzer


def build_sample_product():

    airbus = Underlying("AIRBUS", "equity", 100)
    lvmh = Underlying("LVMH", "equity", 100)

    basket = Basket([airbus, lvmh], type="worst_of")

    return StructuredProduct(
        name="Produit Test",
        maturity_years=5,
        basket=basket,

        coupon=Coupon(0.05, "annual", 0.70, True),

        autocall=Autocall(True, 1.00, "annual"),

        capital=CapitalProtection(0.60, True),

        decrement=0.0
    )


def main():

    product = build_sample_product()

    calendar = CalendarBuilder().build(product)

    simulator = MarketSimulator(
        assets={"AIRBUS": 100, "LVMH": 100},
        r=0.02,
        sigma=0.2,
        corr=0.3
    )

    engine = ProductEngine(product, simulator)

    results = engine.run(calendar, n_simulations=1000)

    analyzer = ProbabilityAnalyzer()
    stats = analyzer.analyze(results)

    print("\nRESULTATS")
    print(stats)

    # 🔥 DEBUG IMPORTANT : UNE SEULE LIGNE
    path = simulator.simulate_path([0, 1, 2, 3, 4, 5])
    print("\nDEBUG PRIX AIRBUS :", path["prices"]["AIRBUS"])


if __name__ == "__main__":
    main()