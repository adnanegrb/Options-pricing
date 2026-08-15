import pytest
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes
from options_pricing.models.monte_carlo import MonteCarlo
from options_pricing.models.binomial_tree import BinomialTree
from options_pricing.greeks.analytical import AnalyticalGreeks
from options_pricing.greeks.numerical import NumericalGreeks
from options_pricing.exotics.barrier import BarrierOption


@pytest.fixture
def atm_call():
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")


@pytest.fixture
def atm_put():
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")


class TestBlackScholes:
    def test_put_call_parity(self, atm_call, atm_put):
        
        call_price = BlackScholes(atm_call).price()
        put_price = BlackScholes(atm_put).price()
        p = atm_call
        import math
        expected = p.S * math.exp(-p.q * p.T) - p.K * math.exp(-p.r * p.T)
        assert (call_price - put_price) == pytest.approx(expected, abs=1e-8)

    def test_deep_itm_call_converges_to_intrinsic(self):
        
        p = OptionParams(S=200, K=100, T=1.0, r=0.05, sigma=0.01, option_type="call")
        price = BlackScholes(p).price()
        import math
        intrinsic = p.S - p.K * math.exp(-p.r * p.T)
        assert price == pytest.approx(intrinsic, rel=1e-2)

    def test_price_is_positive(self, atm_call):
        assert BlackScholes(atm_call).price() > 0


class TestMonteCarlo:
    def test_converges_to_black_scholes(self, atm_call):
        bs_price = BlackScholes(atm_call).price()
        mc_price = MonteCarlo(atm_call, n_paths=200_000).price()
        
        assert mc_price == pytest.approx(bs_price, rel=0.01)

    def test_converges_for_put(self, atm_put):
        bs_price = BlackScholes(atm_put).price()
        mc_price = MonteCarlo(atm_put, n_paths=200_000).price()
        assert mc_price == pytest.approx(bs_price, rel=0.01)

    def test_confidence_interval_contains_bs_price(self, atm_call):
        bs_price = BlackScholes(atm_call).price()
        estimate, lower, upper = MonteCarlo(atm_call, n_paths=200_000).price_with_confidence_interval()
        assert lower <= bs_price <= upper


class TestBinomialTree:
    def test_converges_to_black_scholes_as_steps_grow(self, atm_call):
        bs_price = BlackScholes(atm_call).price()
        binomial_price = BinomialTree(atm_call, n_steps=1000).price()
        assert binomial_price == pytest.approx(bs_price, rel=1e-3)

    def test_converges_for_put(self, atm_put):
        bs_price = BlackScholes(atm_put).price()
        binomial_price = BinomialTree(atm_put, n_steps=1000).price()
        assert binomial_price == pytest.approx(bs_price, rel=1e-3)

    def test_american_call_no_dividends_equals_european(self, atm_call):
        
        european = BinomialTree(atm_call, n_steps=500, american=False).price()
        american = BinomialTree(atm_call, n_steps=500, american=True).price()
        assert american == pytest.approx(european, rel=1e-6)

    def test_american_put_worth_more_than_european(self):
        
        p = OptionParams(S=100, K=120, T=1.0, r=0.05, sigma=0.2, option_type="put")
        european = BinomialTree(p, n_steps=500, american=False).price()
        american = BinomialTree(p, n_steps=500, american=True).price()
        assert american >= european - 1e-6


class TestGreeks:
    def test_analytical_matches_numerical(self, atm_call):
        analytical = AnalyticalGreeks(atm_call).all_greeks()
        numerical = NumericalGreeks(atm_call).all_greeks()
        for greek_name in analytical:
            assert analytical[greek_name] == pytest.approx(numerical[greek_name], abs=1e-2), greek_name

    def test_call_delta_between_0_and_1(self, atm_call):
        delta = AnalyticalGreeks(atm_call).delta()
        assert 0 < delta < 1

    def test_put_delta_between_minus1_and_0(self, atm_put):
        delta = AnalyticalGreeks(atm_put).delta()
        assert -1 < delta < 0

    def test_gamma_is_positive(self, atm_call, atm_put):
        
        assert AnalyticalGreeks(atm_call).gamma() > 0
        assert AnalyticalGreeks(atm_put).gamma() > 0

    def test_vega_is_positive(self, atm_call, atm_put):
        
        assert AnalyticalGreeks(atm_call).vega() > 0
        assert AnalyticalGreeks(atm_put).vega() > 0

    def test_call_and_put_gamma_are_equal(self, atm_call, atm_put):
        
        call_gamma = AnalyticalGreeks(atm_call).gamma()
        put_gamma = AnalyticalGreeks(atm_put).gamma()
        assert call_gamma == pytest.approx(put_gamma, rel=1e-6)


class TestBarrierOption:
    def test_in_out_parity(self):
        
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        vanilla = BlackScholes(p).price()

        out_price = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up").price("analytical")
        in_price = BarrierOption(p, barrier=120, barrier_type="knock_in", direction="up").price("analytical")

        assert (out_price + in_price) == pytest.approx(vanilla, abs=1e-8)

    def test_analytical_matches_mc(self):
        
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        barrier = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up")

        analytical_price = barrier.price("analytical")
        mc_price = barrier.price("mc", n_paths=80_000, n_steps=1000)

        assert mc_price == pytest.approx(analytical_price, abs=0.15)

    def test_down_and_out_analytical_matches_mc(self):
        
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        barrier = BarrierOption(p, barrier=80, barrier_type="knock_out", direction="down")

        analytical_price = barrier.price("analytical")
        mc_price = barrier.price("mc", n_paths=80_000, n_steps=500)

        assert mc_price == pytest.approx(analytical_price, abs=0.1)

    def test_barrier_out_worth_less_than_vanilla(self):
        
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        vanilla = BlackScholes(p).price()
        out_price = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up").price("analytical")
        assert out_price < vanilla

    def test_rejects_invalid_up_barrier(self):
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        with pytest.raises(ValueError):
            BarrierOption(p, barrier=90, barrier_type="knock_out", direction="up")
