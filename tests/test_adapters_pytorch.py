"""Tests for eoil.adapters.pytorch — EOILOptimizer.

Run with::

    pytest packages/eoil/tests/test_adapters_pytorch.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.nn as nn  # noqa: E402

from eoil.adapters.pytorch import (  # noqa: E402
    EOILOptimizer,
    _collect_params,
    _flatten,
    _get_shapes_sizes,
    _unflatten,
    _write_params,
)
from eoil.models import StreamResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_result(
    x_best: list[float] | None = None,
    f_best: float = 0.05,
    converged: bool = False,
    total_steps: int = 50,
    eoil_charged: str = "1.0",
    n: int = 1,
) -> StreamResult:
    return StreamResult(
        x_best=x_best if x_best is not None else [0.1] * n,
        f_best=f_best,
        converged=converged,
        total_steps=total_steps,
        escapes=0,
        compute_units_actual=5,
        eoil_charged=eoil_charged,
    )


def _make_client(stream_result: StreamResult) -> MagicMock:
    client = MagicMock()
    client.stream_optimize.return_value = stream_result
    return client


def _tiny_model() -> nn.Linear:
    """nn.Linear(2, 1) — weight (2,) + bias (1,) = 3 params."""
    torch.manual_seed(0)
    return nn.Linear(2, 1, bias=True)


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# _collect_params
# ---------------------------------------------------------------------------


class TestCollectParams:
    def test_list_of_tensors(self):
        p1 = torch.tensor([1.0, 2.0])
        p2 = torch.tensor([3.0])
        result = _collect_params([p1, p2])
        assert result == [p1, p2]

    def test_generator_is_materialised(self):
        tensors = [torch.zeros(2), torch.zeros(3)]
        result = _collect_params(t for t in tensors)
        assert len(result) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one parameter"):
            _collect_params([])


# ---------------------------------------------------------------------------
# _flatten / _unflatten / _write_params
# ---------------------------------------------------------------------------


class TestFlattenUnflatten:
    def test_flatten_single_tensor(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        result = _flatten([p])
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_flatten_multi_tensor(self):
        p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = nn.Parameter(torch.tensor([3.0]))
        result = _flatten([p1, p2])
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_flatten_2d_tensor(self):
        p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        result = _flatten([p])
        assert result == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_flatten_scalar_tensor(self):
        p = nn.Parameter(torch.tensor(5.0))
        result = _flatten([p])
        assert result == pytest.approx([5.0])

    def test_roundtrip_single(self):
        p = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        shapes, sizes = _get_shapes_sizes([p])
        x = [1.0, 2.0, 3.0]
        _unflatten(x, [p], shapes, sizes, bounds=(-10.0, 10.0))
        assert _flatten([p]) == pytest.approx([1.0, 2.0, 3.0])

    def test_roundtrip_multi_tensor(self):
        p1 = nn.Parameter(torch.zeros(2))
        p2 = nn.Parameter(torch.zeros(3))
        params = [p1, p2]
        shapes, sizes = _get_shapes_sizes(params)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        _unflatten(x, params, shapes, sizes, bounds=(-10.0, 10.0))
        assert _flatten(params) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_unflatten_clips_to_bounds(self):
        p = nn.Parameter(torch.zeros(3))
        shapes, sizes = _get_shapes_sizes([p])
        x = [-5.0, 0.0, 5.0]
        _unflatten(x, [p], shapes, sizes, bounds=(-2.0, 2.0))
        result = _flatten([p])
        assert result == pytest.approx([-2.0, 0.0, 2.0])

    def test_unflatten_preserves_shape(self):
        p = nn.Parameter(torch.zeros(2, 3))
        shapes, sizes = _get_shapes_sizes([p])
        x = list(range(6))
        _unflatten(x, [p], shapes, sizes, bounds=(-100.0, 100.0))
        assert p.data.shape == (2, 3)

    def test_write_params_no_clip(self):
        """_write_params should NOT clip — used for exact snapshot restoration."""
        p = nn.Parameter(torch.zeros(2))
        shapes, sizes = _get_shapes_sizes([p])
        x = [-99.0, 99.0]
        _write_params(x, [p], shapes, sizes)
        result = _flatten([p])
        assert result == pytest.approx([-99.0, 99.0])


# ---------------------------------------------------------------------------
# EOILOptimizer — __init__
# ---------------------------------------------------------------------------


class TestEOILOptimizerInit:
    def test_param_count_correct(self):
        model = _tiny_model()
        opt = EOILOptimizer(model.parameters(), client=MagicMock(), bounds=(-1.0, 1.0))
        assert opt.param_count_ == _param_count(model)  # 3

    def test_last_loss_none_before_step(self):
        model = _tiny_model()
        opt = EOILOptimizer(model.parameters(), client=MagicMock())
        assert opt.last_loss_ is None

    def test_stream_result_none_before_step(self):
        model = _tiny_model()
        opt = EOILOptimizer(model.parameters(), client=MagicMock())
        assert opt.stream_result_ is None

    def test_invalid_gradient_mode_raises(self):
        model = _tiny_model()
        with pytest.raises(ValueError, match="gradient must be"):
            EOILOptimizer(model.parameters(), client=MagicMock(), gradient="bad")

    def test_invalid_bounds_raises(self):
        model = _tiny_model()
        with pytest.raises(ValueError, match="bounds"):
            EOILOptimizer(model.parameters(), client=MagicMock(), bounds=(1.0, -1.0))

    def test_bounds_equal_raises(self):
        model = _tiny_model()
        with pytest.raises(ValueError, match="bounds"):
            EOILOptimizer(model.parameters(), client=MagicMock(), bounds=(0.0, 0.0))

    def test_dimension_over_1000_raises(self):
        # Build a model with > 1000 params
        model = nn.Linear(50, 50)  # 50*50 + 50 = 2550 params
        with pytest.raises(ValueError, match="1000"):
            EOILOptimizer(model.parameters(), client=MagicMock())

    def test_fd_large_dim_warns(self):
        # Linear(10, 6) = 60 + 6 = 66 params — above _FD_DIM_WARN=50
        model = nn.Linear(10, 6)
        with pytest.warns(UserWarning, match="gradient='fd'"):
            EOILOptimizer(
                model.parameters(), client=MagicMock(), gradient="fd"
            )

    def test_autograd_large_dim_no_warn(self):
        # Same model; "autograd" should not warn
        model = nn.Linear(10, 6)
        with warnings.catch_warnings():
            import warnings as _warnings
            _warnings.simplefilter("error")
            EOILOptimizer(
                model.parameters(),
                client=MagicMock(),
                gradient="autograd",
            )


# ---------------------------------------------------------------------------
# EOILOptimizer — step() stream call parameters
# ---------------------------------------------------------------------------


class TestEOILOptimizerStepStreamCall:
    """Verify the arguments forwarded to client.stream_optimize."""

    def _run_step(self, model, gradient="autograd", bounds=(-5.0, 5.0), **kwargs):
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n, n=n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()),
            client=client,
            bounds=bounds,
            budget_steps=100,
            gradient=gradient,
            **kwargs,
        )

        def closure():
            x = model(torch.zeros(1, model.in_features))
            return x.sum()

        opt.step(closure)
        return client.stream_optimize.call_args

    def test_dimension_forwarded(self):
        model = _tiny_model()
        call_args = self._run_step(model)
        assert call_args.kwargs["dimension"] == _param_count(model)

    def test_budget_steps_forwarded(self):
        model = _tiny_model()
        call_args = self._run_step(model)
        assert call_args.kwargs["budget_steps"] == 100

    def test_bounds_forwarded(self):
        model = _tiny_model()
        call_args = self._run_step(model, bounds=(-3.0, 3.0))
        assert call_args.kwargs["bounds"] == (-3.0, 3.0)

    def test_gradient_true_for_autograd(self):
        model = _tiny_model()
        call_args = self._run_step(model, gradient="autograd")
        assert call_args.kwargs["gradient"] is True

    def test_gradient_false_for_fd(self):
        model = _tiny_model()
        call_args = self._run_step(model, gradient="fd")
        assert call_args.kwargs["gradient"] is False

    def test_verify_ssl_forwarded(self):
        model = _tiny_model()
        call_args = self._run_step(model, verify_ssl=False)
        assert call_args.kwargs["verify_ssl"] is False

    def test_on_step_forwarded(self):
        on_step_cb = MagicMock()
        model = _tiny_model()
        call_args = self._run_step(model, on_step=on_step_cb)
        assert call_args.kwargs["on_step"] is on_step_cb

    def test_eval_timeout_forwarded(self):
        model = _tiny_model()
        call_args = self._run_step(model, eval_timeout_s=30.0)
        assert call_args.kwargs["eval_timeout_s"] == 30.0

    def test_x0_present_when_n_le_300(self):
        model = _tiny_model()  # 3 params
        call_args = self._run_step(model)
        assert call_args.kwargs["x0"] is not None
        assert len(call_args.kwargs["x0"]) == _param_count(model)

    def test_x0_none_when_n_gt_300(self):
        # nn.Linear(20, 16) = 320 + 16 = 336 params → above 300
        model = nn.Linear(20, 16)
        n = _param_count(model)
        assert n > 300
        sr = _make_stream_result(x_best=[0.0] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(list(model.parameters()), client=client, bounds=(-1.0, 1.0))
        with pytest.warns(UserWarning, match="x0 warm-start omitted"):
            opt.step(lambda: model(torch.zeros(1, 20)).sum())
        assert client.stream_optimize.call_args.kwargs["x0"] is None


# ---------------------------------------------------------------------------
# EOILOptimizer — step() warm-start
# ---------------------------------------------------------------------------


class TestEOILOptimizerWarmStart:
    def test_second_call_uses_x_best_as_x0(self):
        model = _tiny_model()
        n = _param_count(model)
        x_best_first = [0.5] * n
        sr1 = _make_stream_result(x_best=x_best_first)
        sr2 = _make_stream_result(x_best=[0.3] * n)
        client = MagicMock()
        client.stream_optimize.side_effect = [sr1, sr2]
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )

        def closure():
            return model(torch.zeros(1, 2)).sum()

        opt.step(closure)
        opt.step(closure)

        second_x0 = client.stream_optimize.call_args_list[1].kwargs["x0"]
        assert second_x0 == pytest.approx(x_best_first)


# ---------------------------------------------------------------------------
# EOILOptimizer — step() write-back and attributes
# ---------------------------------------------------------------------------


class TestEOILOptimizerWriteback:
    def test_params_updated_to_x_best(self):
        model = _tiny_model()
        n = _param_count(model)
        target = [0.25] * n
        sr = _make_stream_result(x_best=target)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        opt.step(lambda: model(torch.zeros(1, 2)).sum())
        current = _flatten(list(model.parameters()))
        assert current == pytest.approx(target)

    def test_last_loss_set(self):
        model = _tiny_model()
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n, f_best=0.42)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        opt.step(lambda: model(torch.zeros(1, 2)).sum())
        assert opt.last_loss_ == pytest.approx(0.42)

    def test_stream_result_set(self):
        model = _tiny_model()
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        opt.step(lambda: model(torch.zeros(1, 2)).sum())
        assert opt.stream_result_ is sr

    def test_step_returns_f_best(self):
        model = _tiny_model()
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n, f_best=1.23)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        loss = opt.step(lambda: model(torch.zeros(1, 2)).sum())
        assert loss == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# EOILOptimizer — step() exception recovery
# ---------------------------------------------------------------------------


class TestEOILOptimizerExceptionRecovery:
    def test_params_restored_on_stream_error(self):
        model = _tiny_model()
        initial_params = _flatten(list(model.parameters()))
        client = MagicMock()
        client.stream_optimize.side_effect = RuntimeError("server error")
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        with pytest.raises(RuntimeError):
            opt.step(lambda: model(torch.zeros(1, 2)).sum())
        restored = _flatten(list(model.parameters()))
        assert restored == pytest.approx(initial_params)

    def test_params_restored_on_nan_x_best(self):
        model = _tiny_model()
        initial_params = _flatten(list(model.parameters()))
        n = _param_count(model)
        sr = _make_stream_result(x_best=[float("nan")] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        with pytest.raises(ValueError, match="non-finite"):
            opt.step(lambda: model(torch.zeros(1, 2)).sum())
        restored = _flatten(list(model.parameters()))
        assert restored == pytest.approx(initial_params)

    def test_last_loss_unchanged_on_error(self):
        model = _tiny_model()
        client = MagicMock()
        client.stream_optimize.side_effect = RuntimeError("boom")
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        with pytest.raises(RuntimeError):
            opt.step(lambda: model(torch.zeros(1, 2)).sum())
        assert opt.last_loss_ is None


# ---------------------------------------------------------------------------
# EOILOptimizer — step() objective function (autograd mode)
# ---------------------------------------------------------------------------


class TestAutoGradObjective:
    """Inspect the captured fn argument and call it directly."""

    def _capture_fn(self, model, closure):
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        opt.step(closure)
        return client.stream_optimize.call_args.kwargs["fn"], model, opt

    def test_autograd_objective_returns_float_and_grad(self):
        model = _tiny_model()
        fn, model, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum()
        )
        x_test = [0.1] * opt.param_count_
        result = fn(x_test)
        assert isinstance(result, tuple)
        f, grad = result
        assert isinstance(f, float)
        assert math.isfinite(f)
        assert len(grad) == opt.param_count_
        assert all(isinstance(g, float) for g in grad)

    def test_autograd_backward_called(self):
        model = _tiny_model()
        fn, model, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum()
        )
        x_test = [0.1] * opt.param_count_
        fn(x_test)
        for p in model.parameters():
            assert p.grad is not None

    def test_autograd_nan_in_x_raises(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum()
        )
        x_nan = [float("nan")] * opt.param_count_
        with pytest.raises(ValueError, match="non-finite"):
            fn(x_nan)

    def test_autograd_inf_in_x_raises(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum()
        )
        x_inf = [float("inf")] + [0.0] * (opt.param_count_ - 1)
        with pytest.raises(ValueError, match="non-finite"):
            fn(x_inf)

    def test_autograd_non_tensor_closure_raises_type_error(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(model, lambda: 1.0)  # returns float
        with pytest.raises(TypeError, match="torch.Tensor"):
            fn([0.1] * opt.param_count_)

    def test_autograd_non_finite_loss_raises(self):
        model = _tiny_model()

        def bad_closure():
            return torch.tensor(float("nan"))

        fn, _, opt = self._capture_fn(model, bad_closure)
        with pytest.raises(ValueError, match="non-finite loss"):
            fn([0.1] * opt.param_count_)

    def test_autograd_clips_x_to_bounds(self):
        """x values passed to objective should be clipped; params stay in bounds."""
        model = nn.Linear(1, 1, bias=False)
        fn, model, opt = self._capture_fn(
            model, lambda: model(torch.ones(1, 1)).sum()
        )
        # Pass value far outside bounds; after clip, param should be at hi=5.0
        fn([999.0])
        assert model.weight.data.item() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# EOILOptimizer — step() objective function (fd mode)
# ---------------------------------------------------------------------------


class TestFDObjective:
    def _capture_fn(self, model, closure):
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()),
            client=client,
            bounds=(-5.0, 5.0),
            gradient="fd",
        )
        opt.step(closure)
        return client.stream_optimize.call_args.kwargs["fn"], model, opt

    def test_fd_objective_returns_float(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum().item()
        )
        x_test = [0.1] * opt.param_count_
        result = fn(x_test)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_fd_accepts_tensor_return(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum()
        )
        result = fn([0.1] * opt.param_count_)
        assert isinstance(result, float)

    def test_fd_nan_in_x_raises(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(
            model, lambda: model(torch.zeros(1, 2)).sum()
        )
        with pytest.raises(ValueError, match="non-finite"):
            fn([float("nan")] * opt.param_count_)

    def test_fd_non_finite_loss_raises(self):
        model = _tiny_model()
        fn, _, opt = self._capture_fn(model, lambda: float("inf"))
        with pytest.raises(ValueError, match="non-finite loss"):
            fn([0.1] * opt.param_count_)


# ---------------------------------------------------------------------------
# EOILOptimizer — bounds warning
# ---------------------------------------------------------------------------


class TestBoundsWarning:
    def test_warns_when_params_outside_bounds(self):
        model = nn.Linear(1, 1, bias=False)
        # Manually set the weight outside the bounds
        with torch.no_grad():
            model.weight.fill_(10.0)
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-1.0, 1.0)
        )
        with pytest.warns(UserWarning, match="outside the declared bounds"):
            opt.step(lambda: model(torch.ones(1, 1)).sum())


# ---------------------------------------------------------------------------
# EOILOptimizer — zero_grad
# ---------------------------------------------------------------------------


class TestZeroGrad:
    def test_zero_grad_returns_none(self):
        model = _tiny_model()
        opt = EOILOptimizer(model.parameters(), client=MagicMock())
        assert opt.zero_grad() is None

    def test_zero_grad_callable(self):
        model = _tiny_model()
        opt = EOILOptimizer(model.parameters(), client=MagicMock())
        opt.zero_grad()  # should not raise


# ---------------------------------------------------------------------------
# Integration — end-to-end with a tiny model and mock client
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_params_change_after_step(self):
        """After step(), model parameters should differ from initial values."""
        torch.manual_seed(42)
        model = nn.Linear(2, 1)
        initial_params = _flatten(list(model.parameters()))

        # x_best is different from initial params
        n = _param_count(model)
        x_best = [v + 0.5 for v in initial_params]
        sr = _make_stream_result(x_best=x_best)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-10.0, 10.0)
        )

        X = torch.randn(4, 2)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()

        opt.step(lambda: criterion(model(X), y))

        final_params = _flatten(list(model.parameters()))
        assert final_params != pytest.approx(initial_params)

    def test_full_step_attributes_after_run(self):
        """last_loss_ and stream_result_ are set correctly after step()."""
        model = _tiny_model()
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.1] * n, f_best=0.007)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        loss = opt.step(lambda: model(torch.zeros(1, 2)).sum())

        assert loss == pytest.approx(0.007)
        assert opt.last_loss_ == pytest.approx(0.007)
        assert opt.stream_result_ is sr
        assert opt.param_count_ == n

    def test_step_called_once(self):
        """stream_optimize is called exactly once per step()."""
        model = _tiny_model()
        n = _param_count(model)
        sr = _make_stream_result(x_best=[0.0] * n)
        client = _make_client(sr)
        opt = EOILOptimizer(
            list(model.parameters()), client=client, bounds=(-5.0, 5.0)
        )
        opt.step(lambda: model(torch.zeros(1, 2)).sum())
        assert client.stream_optimize.call_count == 1


import warnings  # noqa: E402  (used in TestEOILOptimizerInit)
