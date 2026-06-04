"""Tests for eoil.adapters.scipy — minimize()."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from eoil.adapters.scipy import OptimizeResult, minimize, _parse_bounds, _wrap_fn
from eoil.models import StreamResult


# ---------------------------------------------------------------------------
# Helpers — build a fake StreamResult
# ---------------------------------------------------------------------------

def _make_stream_result(
    x_best=None,
    f_best=0.0,
    converged=True,
    total_steps=100,
    eoil_charged="2.5",
) -> StreamResult:
    return StreamResult(
        x_best=x_best or [1.0, 1.0],
        f_best=f_best,
        converged=converged,
        total_steps=total_steps,
        escapes=0,
        compute_units_actual=10,
        eoil_charged=eoil_charged,
    )


def _make_client(stream_result: StreamResult) -> MagicMock:
    client = MagicMock()
    client.stream_optimize.return_value = stream_result
    return client


# ---------------------------------------------------------------------------
# _parse_bounds
# ---------------------------------------------------------------------------

class TestParseBounds:
    def test_none_returns_default(self):
        assert _parse_bounds(None, 2) == (-5.0, 5.0)

    def test_scalar_pair(self):
        assert _parse_bounds((-3.0, 3.0), 2) == (-3.0, 3.0)

    def test_per_dimension_uniform(self):
        # No warning when all bounds are the same
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lo, hi = _parse_bounds([(-2.0, 2.0), (-2.0, 2.0)], 2)
        assert lo == -2.0
        assert hi == 2.0

    def test_per_dimension_non_uniform_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lo, hi = _parse_bounds([(-1.0, 1.0), (-3.0, 3.0)], 2)
        assert lo == -3.0
        assert hi == 3.0
        assert len(w) == 1
        assert "collapsed" in str(w[0].message).lower()

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported bounds format"):
            _parse_bounds("invalid", 2)

    def test_scipy_bounds_object(self):
        pytest.importorskip("scipy")
        from scipy.optimize import Bounds
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lo, hi = _parse_bounds(Bounds(lb=-4.0, ub=4.0), 2)
        assert lo == -4.0
        assert hi == 4.0

    def test_scipy_bounds_per_dim_collapses(self):
        pytest.importorskip("scipy")
        from scipy.optimize import Bounds
        import numpy as np
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lo, hi = _parse_bounds(Bounds(lb=[-1.0, -3.0], ub=[1.0, 3.0]), 2)
        assert lo == -3.0
        assert hi == 3.0
        assert len(w) == 1


# ---------------------------------------------------------------------------
# _wrap_fn
# ---------------------------------------------------------------------------

class TestWrapFn:
    def test_jac_none_returns_auto(self):
        fn = lambda x: 1.0  # noqa: E731
        wrapped, mode = _wrap_fn(fn, None)
        assert wrapped is fn
        assert mode == "auto"

    def test_jac_false_returns_auto(self):
        fn = lambda x: 1.0  # noqa: E731
        wrapped, mode = _wrap_fn(fn, False)
        assert wrapped is fn
        assert mode == "auto"

    def test_jac_true_returns_true(self):
        fn = lambda x: (1.0, [0.0])  # noqa: E731
        wrapped, mode = _wrap_fn(fn, True)
        assert wrapped is fn
        assert mode is True

    def test_jac_callable_wraps(self):
        fn = lambda x: sum(xi**2 for xi in x)  # noqa: E731
        jac = lambda x: [2 * xi for xi in x]  # noqa: E731
        wrapped, mode = _wrap_fn(fn, jac)
        result = wrapped([1.0, 2.0])
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx([2.0, 4.0])
        assert mode is True

    def test_jac_invalid_raises(self):
        with pytest.raises(ValueError, match="jac must be"):
            _wrap_fn(lambda x: 1.0, 42)


# ---------------------------------------------------------------------------
# minimize() — unit tests (mock client)
# ---------------------------------------------------------------------------

class TestMinimize:
    def test_basic_converged(self):
        sr = _make_stream_result(x_best=[1.0, 1.0], f_best=0.0, converged=True, total_steps=200)
        client = _make_client(sr)

        result = minimize(lambda x: 0.0, x0=[0.0, 0.0], client=client)

        assert isinstance(result, OptimizeResult)
        assert result.x == [1.0, 1.0]
        assert result.fun == 0.0
        assert result.success is True
        assert result.nit == 200
        assert result.nfev == 200
        assert result.status == 0
        assert "converged" in result.message.lower()

    def test_not_converged_status(self):
        sr = _make_stream_result(converged=False, total_steps=500)
        client = _make_client(sr)

        result = minimize(lambda x: 1.0, x0=[0.0], client=client)

        assert result.success is False
        assert result.status == 1
        assert "did not converge" in result.message.lower()

    def test_eoil_charged_propagated(self):
        sr = _make_stream_result(eoil_charged="7.25")
        client = _make_client(sr)
        result = minimize(lambda x: 0.0, x0=[0.0, 0.0], client=client)
        assert result.eoil_charged == "7.25"

    def test_raw_is_stream_result(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        result = minimize(lambda x: 0.0, x0=[0.0, 0.0], client=client)
        assert result.raw is sr

    def test_dimension_derived_from_x0(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        minimize(lambda x: 0.0, x0=[0.0, 0.0, 0.0, 0.0], client=client)
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["dimension"] == 4

    def test_x0_forwarded(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        minimize(lambda x: 0.0, x0=[1.5, -2.5], client=client)
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["x0"] == [1.5, -2.5]

    def test_budget_steps_forwarded(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        minimize(lambda x: 0.0, x0=[0.0], client=client, budget_steps=1000)
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["budget_steps"] == 1000

    def test_bounds_forwarded_scalar_pair(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        minimize(lambda x: 0.0, x0=[0.0, 0.0], client=client, bounds=(-3.0, 3.0))
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["bounds"] == (-3.0, 3.0)

    def test_verify_ssl_forwarded(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        minimize(lambda x: 0.0, x0=[0.0], client=client, verify_ssl=False)
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["verify_ssl"] is False

    def test_jac_true_gradient_mode(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        fn = lambda x: (sum(xi**2 for xi in x), [2 * xi for xi in x])  # noqa: E731
        minimize(fn, x0=[0.0, 0.0], client=client, jac=True)
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["gradient"] is True

    def test_jac_callable_combined_fn(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        fn = lambda x: sum(xi**2 for xi in x)  # noqa: E731
        jac = lambda x: [2 * xi for xi in x]  # noqa: E731
        minimize(fn, x0=[0.0, 0.0], client=client, jac=jac)
        call_kwargs = client.stream_optimize.call_args.kwargs
        # gradient=True because _wrap_fn combined them
        assert call_kwargs["gradient"] is True
        # Verify the wrapped fn returns (f, grad)
        wrapped_fn = call_kwargs["fn"]
        f, g = wrapped_fn([1.0, 2.0])
        assert f == pytest.approx(5.0)
        assert g == pytest.approx([2.0, 4.0])

    def test_empty_x0_raises(self):
        client = _make_client(_make_stream_result())
        with pytest.raises(ValueError, match="x0 must not be empty"):
            minimize(lambda x: 0.0, x0=[], client=client)

    def test_scipy_compat_ignored_kwargs_accepted(self):
        """method/hess/hessp/constraints/tol/callback/args are all ignored gracefully."""
        sr = _make_stream_result()
        client = _make_client(sr)
        result = minimize(
            lambda x: 0.0,
            x0=[0.0],
            client=client,
            method="L-BFGS-B",
            hess=None,
            hessp=None,
            constraints=[],
            tol=1e-6,
            callback=lambda x: None,
            args=(),
        )
        assert result.success is True

    def test_on_step_forwarded(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        cb = lambda step, x, f: None  # noqa: E731
        minimize(lambda x: 0.0, x0=[0.0], client=client, on_step=cb)
        call_kwargs = client.stream_optimize.call_args.kwargs
        assert call_kwargs["on_step"] is cb

    def test_repr(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        result = minimize(lambda x: 0.0, x0=[0.0], client=client)
        r = repr(result)
        assert "OptimizeResult" in r
        assert "success=" in r
