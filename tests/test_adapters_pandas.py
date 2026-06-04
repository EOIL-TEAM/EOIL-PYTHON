"""Tests for eoil.adapters.pandas — EOILPortfolioOptimizer."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="pandas not installed")

from eoil.adapters.pandas import (  # noqa: E402
    EOILPortfolioOptimizer,
    _build_objective,
    _portfolio_stats,
    _softmax,
)
from eoil.models import StreamResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_returns(n_assets: int = 3, n_periods: int = 100) -> "pd.DataFrame":
    data = RNG.normal(0.0005, 0.01, size=(n_periods, n_assets))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


def _make_stream_result(
    x_best: list[float] | None = None,
    f_best: float = -1.5,
    n: int = 3,
) -> StreamResult:
    return StreamResult(
        x_best=x_best if x_best is not None else [0.0] * n,
        f_best=f_best,
        converged=False,
        total_steps=100,
        escapes=2,
        compute_units_actual=10,
        eoil_charged="1.0",
    )


def _make_client(stream_result: StreamResult) -> MagicMock:
    client = MagicMock()
    client.stream_optimize.return_value = stream_result
    return client


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_sums_to_one(self):
        z = np.array([1.0, 2.0, 3.0])
        w = _softmax(z)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_all_positive(self):
        z = np.array([-5.0, 0.0, 5.0])
        w = _softmax(z)
        assert np.all(w > 0)

    def test_uniform_input_gives_equal_weights(self):
        z = np.zeros(4)
        w = _softmax(z)
        assert np.allclose(w, 0.25)

    def test_large_values_numerically_stable(self):
        z = np.array([1000.0, 1001.0, 999.0])
        w = _softmax(z)
        assert np.all(np.isfinite(w))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_output_shape(self):
        z = np.ones(5)
        w = _softmax(z)
        assert w.shape == (5,)


# ---------------------------------------------------------------------------
# _portfolio_stats
# ---------------------------------------------------------------------------


class TestPortfolioStats:
    def setup_method(self):
        self.returns = _make_returns(n_assets=3, n_periods=252).values

    def test_returns_three_values(self):
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        result = _portfolio_stats(w, self.returns, periods_per_year=252)
        assert len(result) == 3

    def test_values_are_finite(self):
        w = np.array([0.5, 0.3, 0.2])
        ann_ret, ann_vol, sharpe = _portfolio_stats(w, self.returns, 252)
        assert math.isfinite(ann_ret)
        assert math.isfinite(ann_vol)
        assert math.isfinite(sharpe)

    def test_vol_is_positive(self):
        w = np.array([0.5, 0.3, 0.2])
        _, ann_vol, _ = _portfolio_stats(w, self.returns, 252)
        assert ann_vol > 0

    def test_different_periods_per_year(self):
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        _, vol_252, _ = _portfolio_stats(w, self.returns, 252)
        _, vol_52, _ = _portfolio_stats(w, self.returns, 52)
        # Weekly annualisation gives lower vol than daily
        assert vol_52 < vol_252


# ---------------------------------------------------------------------------
# _build_objective
# ---------------------------------------------------------------------------


class TestBuildObjective:
    def setup_method(self):
        self.returns = _make_returns(n_assets=3, n_periods=252).values
        self.z = np.array([0.0, 0.0, 0.0])

    def test_sharpe_returns_float(self):
        fn = _build_objective("sharpe", self.returns, 252, 0.0)
        result = fn(self.z)
        assert isinstance(result, float)

    def test_sharpe_is_negated(self):
        """Sharpe objective is negated — EOIL minimises."""
        fn = _build_objective("sharpe", self.returns, 252, 0.0)
        result = fn(self.z)
        w = _softmax(self.z)
        _, _, raw_sharpe = _portfolio_stats(w, self.returns, 252)
        assert result == pytest.approx(-raw_sharpe)

    def test_min_variance_returns_float(self):
        fn = _build_objective("min_variance", self.returns, 252, 0.0)
        result = fn(self.z)
        assert isinstance(result, float)
        assert result > 0

    def test_min_variance_is_variance(self):
        fn = _build_objective("min_variance", self.returns, 252, 0.0)
        result = fn(self.z)
        w = _softmax(self.z)
        _, ann_vol, _ = _portfolio_stats(w, self.returns, 252)
        assert result == pytest.approx(ann_vol ** 2)

    def test_max_return_is_negated(self):
        fn = _build_objective("max_return", self.returns, 252, 0.0)
        result = fn(self.z)
        w = _softmax(self.z)
        ann_ret, _, _ = _portfolio_stats(w, self.returns, 252)
        assert result == pytest.approx(-ann_ret)

    def test_risk_free_rate_affects_sharpe(self):
        fn0 = _build_objective("sharpe", self.returns, 252, 0.0)
        fn5 = _build_objective("sharpe", self.returns, 252, 0.05)
        # higher rfr → lower (more negative) sharpe → higher objective value for
        # a positive-return portfolio or less negative; just check they differ
        assert fn0(self.z) != fn5(self.z)


# ---------------------------------------------------------------------------
# EOILPortfolioOptimizer — __init__ validation
# ---------------------------------------------------------------------------


class TestEOILPortfolioOptimizerInit:
    def test_n_assets_set(self):
        returns = _make_returns(n_assets=5)
        opt = EOILPortfolioOptimizer(returns, client=MagicMock())
        assert opt.n_assets_ == 5

    def test_weights_none_before_fit(self):
        returns = _make_returns()
        opt = EOILPortfolioOptimizer(returns, client=MagicMock())
        assert opt.weights_ is None

    def test_not_a_dataframe_raises(self):
        with pytest.raises(TypeError, match="pd.DataFrame"):
            EOILPortfolioOptimizer(np.zeros((10, 3)), client=MagicMock())

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            EOILPortfolioOptimizer(pd.DataFrame(), client=MagicMock())

    def test_single_asset_raises(self):
        returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
        with pytest.raises(ValueError, match="at least 2"):
            EOILPortfolioOptimizer(returns, client=MagicMock())

    def test_nan_in_returns_raises(self):
        returns = _make_returns()
        returns.iloc[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            EOILPortfolioOptimizer(returns, client=MagicMock())

    def test_invalid_objective_raises(self):
        returns = _make_returns()
        with pytest.raises(ValueError, match="objective must be one of"):
            EOILPortfolioOptimizer(returns, client=MagicMock(), objective="bad")

    def test_invalid_bounds_raises(self):
        returns = _make_returns()
        with pytest.raises(ValueError, match="bounds"):
            EOILPortfolioOptimizer(returns, client=MagicMock(), bounds=(1.0, -1.0))

    def test_too_many_assets_raises(self):
        # 1001 assets — over server cap
        big = pd.DataFrame(
            RNG.normal(0, 0.01, size=(50, 1001)),
            columns=[f"A{i}" for i in range(1001)],
        )
        with pytest.raises(ValueError, match="1000"):
            EOILPortfolioOptimizer(big, client=MagicMock())


# ---------------------------------------------------------------------------
# EOILPortfolioOptimizer — fit() stream call
# ---------------------------------------------------------------------------


class TestFitStreamCall:
    def _run_fit(self, n=3, **kwargs):
        returns = _make_returns(n_assets=n)
        sr = _make_stream_result(n=n)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client, budget_steps=50, **kwargs)
        opt.fit()
        return client.stream_optimize.call_args, opt, sr

    def test_dimension_forwarded(self):
        call_args, opt, _ = self._run_fit(n=4)
        assert call_args.kwargs["dimension"] == 4

    def test_budget_steps_forwarded(self):
        call_args, _, _ = self._run_fit()
        assert call_args.kwargs["budget_steps"] == 50

    def test_gradient_false(self):
        call_args, _, _ = self._run_fit()
        assert call_args.kwargs["gradient"] is False

    def test_bounds_forwarded(self):
        call_args, _, _ = self._run_fit(bounds=(-3.0, 3.0))
        assert call_args.kwargs["bounds"] == (-3.0, 3.0)

    def test_verify_ssl_forwarded(self):
        call_args, _, _ = self._run_fit(verify_ssl=False)
        assert call_args.kwargs["verify_ssl"] is False

    def test_on_step_forwarded(self):
        cb = MagicMock()
        call_args, _, _ = self._run_fit(on_step=cb)
        assert call_args.kwargs["on_step"] is cb

    def test_x0_uniform_logits(self):
        call_args, opt, _ = self._run_fit(n=3)
        x0 = call_args.kwargs["x0"]
        assert x0 == [0.0, 0.0, 0.0]

    def test_called_once(self):
        returns = _make_returns(n_assets=3)
        sr = _make_stream_result(n=3)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()
        assert client.stream_optimize.call_count == 1


# ---------------------------------------------------------------------------
# EOILPortfolioOptimizer — fit() output validation
# ---------------------------------------------------------------------------


class TestFitOutputs:
    def _fit(self, x_best=None, n=3):
        returns = _make_returns(n_assets=n)
        sr = _make_stream_result(x_best=x_best, n=n)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()
        return opt, sr, returns

    def test_weights_is_series(self):
        opt, _, _ = self._fit()
        assert isinstance(opt.weights_, pd.Series)

    def test_weights_sum_to_one(self):
        opt, _, _ = self._fit()
        assert abs(opt.weights_.sum() - 1.0) < 1e-10

    def test_weights_all_non_negative(self):
        opt, _, _ = self._fit()
        assert (opt.weights_ >= 0).all()

    def test_weights_index_matches_columns(self):
        returns = _make_returns(n_assets=4)
        sr = _make_stream_result(n=4)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()
        assert list(opt.weights_.index) == list(returns.columns)

    def test_stream_result_stored(self):
        opt, sr, _ = self._fit()
        assert opt.stream_result_ is sr

    def test_portfolio_return_finite(self):
        opt, _, _ = self._fit()
        assert math.isfinite(opt.portfolio_return_)

    def test_portfolio_volatility_positive(self):
        opt, _, _ = self._fit()
        assert opt.portfolio_volatility_ > 0

    def test_sharpe_ratio_finite(self):
        opt, _, _ = self._fit()
        assert math.isfinite(opt.sharpe_ratio_)

    def test_fit_returns_self(self):
        returns = _make_returns()
        sr = _make_stream_result()
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        result = opt.fit()
        assert result is opt

    def test_nan_x_best_raises(self):
        returns = _make_returns(n_assets=3)
        sr = _make_stream_result(x_best=[float("nan"), 0.0, 0.0])
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        with pytest.raises(ValueError, match="non-finite"):
            opt.fit()

    def test_inf_x_best_raises(self):
        returns = _make_returns(n_assets=3)
        sr = _make_stream_result(x_best=[float("inf"), 0.0, 0.0])
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        with pytest.raises(ValueError, match="non-finite"):
            opt.fit()


# ---------------------------------------------------------------------------
# EOILPortfolioOptimizer — objective variants
# ---------------------------------------------------------------------------


class TestObjectiveVariants:
    def _fit_with_objective(self, objective, x_best=None):
        returns = _make_returns(n_assets=3)
        n = 3
        sr = _make_stream_result(x_best=x_best or [0.0] * n, n=n)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client, objective=objective)
        opt.fit()
        return opt

    def test_sharpe_mode(self):
        opt = self._fit_with_objective("sharpe")
        assert opt.sharpe_ratio_ is not None

    def test_min_variance_mode(self):
        opt = self._fit_with_objective("min_variance")
        assert opt.portfolio_volatility_ is not None
        assert opt.portfolio_volatility_ > 0

    def test_max_return_mode(self):
        opt = self._fit_with_objective("max_return")
        assert opt.portfolio_return_ is not None

    def test_risk_free_rate_stored(self):
        returns = _make_returns()
        opt = EOILPortfolioOptimizer(
            returns, client=MagicMock(), risk_free_rate=0.03
        )
        assert opt._risk_free_rate == 0.03


# ---------------------------------------------------------------------------
# EOILPortfolioOptimizer — wrapped fn NaN guard
# ---------------------------------------------------------------------------


class TestWrappedFnNaNGuard:
    def test_nan_in_x_raises_from_server(self):
        """The wrapped fn passed to stream_optimize should raise on NaN x."""
        returns = _make_returns(n_assets=3)
        captured_fn = None

        def fake_stream_optimize(**kwargs):
            nonlocal captured_fn
            captured_fn = kwargs["fn"]
            return _make_stream_result(n=3)

        client = MagicMock()
        client.stream_optimize.side_effect = fake_stream_optimize
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()

        with pytest.raises(ValueError, match="non-finite"):
            captured_fn([float("nan"), 0.0, 0.0])

    def test_valid_x_returns_float(self):
        returns = _make_returns(n_assets=3)
        captured_fn = None

        def fake_stream_optimize(**kwargs):
            nonlocal captured_fn
            captured_fn = kwargs["fn"]
            return _make_stream_result(n=3)

        client = MagicMock()
        client.stream_optimize.side_effect = fake_stream_optimize
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()

        result = captured_fn([0.5, -0.5, 1.0])
        assert isinstance(result, float)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_weights_name(self):
        returns = _make_returns(n_assets=3)
        sr = _make_stream_result(n=3)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()
        assert opt.weights_.name == "weight"

    def test_non_uniform_x_best_gives_non_equal_weights(self):
        """Non-zero logits should yield non-uniform weights."""
        returns = _make_returns(n_assets=3)
        sr = _make_stream_result(x_best=[2.0, 0.0, -2.0], n=3)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()
        weights = opt.weights_.values
        assert not np.allclose(weights, weights[0])  # not all equal

    def test_custom_asset_names_in_index(self):
        returns = pd.DataFrame(
            RNG.normal(0, 0.01, size=(50, 3)),
            columns=["AAPL", "MSFT", "GOOG"],
        )
        sr = _make_stream_result(n=3)
        client = _make_client(sr)
        opt = EOILPortfolioOptimizer(returns, client=client)
        opt.fit()
        assert list(opt.weights_.index) == ["AAPL", "MSFT", "GOOG"]

    def test_periods_per_year_changes_annualisation(self):
        returns = _make_returns(n_assets=3)
        sr = _make_stream_result(n=3)

        client1 = _make_client(sr)
        opt1 = EOILPortfolioOptimizer(
            returns, client=client1, periods_per_year=252
        )
        opt1.fit()

        client2 = _make_client(sr)
        opt2 = EOILPortfolioOptimizer(
            returns, client=client2, periods_per_year=52
        )
        opt2.fit()

        # Different periods_per_year should yield different annualised stats
        assert opt1.portfolio_volatility_ != pytest.approx(opt2.portfolio_volatility_)
