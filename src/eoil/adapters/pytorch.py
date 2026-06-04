"""pytorch adapter for EOIL — EOILOptimizer as a drop-in for torch.optim.

Usage::

    import torch
    import torch.nn as nn
    import eoil
    from eoil.adapters.pytorch import EOILOptimizer

    client = eoil.Client(api_key="eoil_sk_...")

    model = nn.Linear(10, 1)
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    criterion = nn.MSELoss()

    optimizer = EOILOptimizer(
        model.parameters(),
        client=client,
        bounds=(-2.0, 2.0),
        budget_steps=300,
        gradient="autograd",   # uses torch autograd — fastest
    )

    def closure():
        return criterion(model(X), y)

    optimizer.step(closure)

    print(f"best loss: {optimizer.last_loss_:.4f}")
    print(optimizer.stream_result_)

Gradient modes
--------------
``"autograd"`` (default)
    Calls ``loss.backward()`` after each evaluation.  The closure *must*
    return a ``torch.Tensor`` scalar with a live computation graph.
    Fastest and most accurate.

``"fd"``
    Uses finite-difference gradients computed server-side.  Works for
    any closure (differentiable or not), but is expensive: each EOIL
    step costs ``2 × dimension + 1`` closure calls.  A warning is raised
    when ``dimension > 50``.

Notes
-----
* EOIL enforces a hard limit of 1000 parameters — ``ValueError`` is
  raised in ``__init__`` if the model is larger.
* When ``dimension > 300`` the ``x0`` warm-start is silently omitted
  to avoid URL-length constraints.
* Server-side bounds are only used for escape calibration; the adapter
  clips every ``x_next`` / ``x_best`` to the declared bounds.
* If an exception is raised during ``step()``, the model parameters are
  restored to their pre-step values.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Optional, Union

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]

__all__ = ["EOILOptimizer"]

# URL-length limit: omit x0 warm-start above this dimension
_X0_URL_LIMIT: int = 300
# warn about FD closure-call explosion above this dimension
_FD_DIM_WARN: int = 50


# ---------------------------------------------------------------------------
# Internal helpers (exposed so tests can import them directly)
# ---------------------------------------------------------------------------


def _collect_params(params_iter) -> list:
    """Materialise a parameter iterator into a stable list of tensors."""
    params = list(params_iter)
    if not params:
        raise ValueError("EOILOptimizer requires at least one parameter.")
    return params


def _get_shapes_sizes(params: list) -> tuple[list, list]:
    shapes = [tuple(p.data.shape) for p in params]
    sizes = [p.data.numel() for p in params]
    return shapes, sizes


def _flatten(params: list) -> list[float]:
    """Flatten all parameter tensors into a single ``list[float]``."""
    result: list[float] = []
    for p in params:
        result.extend(p.data.detach().cpu().reshape(-1).tolist())
    return result


def _unflatten(
    x: list[float],
    params: list,
    shapes: list,
    sizes: list,
    bounds: tuple[float, float],
) -> None:
    """Write *x* back into parameter ``.data`` in-place, clipping to *bounds*."""
    offset = 0
    lo, hi = bounds
    for p, shape, size in zip(params, shapes, sizes):
        chunk = [max(lo, min(hi, v)) for v in x[offset : offset + size]]
        offset += size
        with torch.no_grad():
            p.data.copy_(
                torch.tensor(
                    chunk, dtype=p.data.dtype, device=p.data.device
                ).reshape(shape)
            )


def _write_params(
    x: list[float],
    params: list,
    shapes: list,
    sizes: list,
) -> None:
    """Write *x* back into parameter ``.data`` without bounds clipping.

    Used exclusively for exception-recovery restoration so that the original
    (potentially out-of-bounds) parameter values are preserved exactly.
    """
    offset = 0
    for p, shape, size in zip(params, shapes, sizes):
        chunk = x[offset : offset + size]
        offset += size
        with torch.no_grad():
            p.data.copy_(
                torch.tensor(
                    chunk, dtype=p.data.dtype, device=p.data.device
                ).reshape(shape)
            )


# ---------------------------------------------------------------------------
# EOILOptimizer
# ---------------------------------------------------------------------------


class EOILOptimizer:
    """Use EOIL global optimisation to tune the parameters of a PyTorch model.

    Parameters
    ----------
    params:
        An iterable of ``torch.Tensor`` parameters (e.g.
        ``model.parameters()``).  The iterator is materialised once at
        construction time.
    client:
        An authenticated ``eoil.Client`` instance.
    bounds:
        ``(lo, hi)`` box bounds applied to every parameter.  Values from the
        EOIL server are clipped to this range before being written back into
        the model.
    budget_steps:
        Number of EOIL optimisation steps per ``step()`` call.
    gradient:
        ``"autograd"`` (default) — uses ``loss.backward()`` for exact
        gradients; requires a differentiable closure.
        ``"fd"`` — finite-difference gradients; works for any closure.
    on_step:
        Optional callable invoked after each EOIL step with the current
        best ``(x, f)``; forwarded to ``client.stream_optimize``.
    eval_timeout_s:
        Per-closure evaluation timeout in seconds passed to the server.
    verify_ssl:
        Whether to verify the TLS certificate of the EOIL WebSocket endpoint.
    """

    def __init__(
        self,
        params,
        *,
        client: Any,
        bounds: tuple[float, float] = (-1.0, 1.0),
        budget_steps: int = 200,
        gradient: str = "autograd",
        on_step: Optional[Callable] = None,
        eval_timeout_s: float = 80.0,
        verify_ssl: bool = True,
    ) -> None:
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for EOILOptimizer. "
                "Install it with: pip install 'eoil[torch]'"
            )

        if gradient not in ("autograd", "fd"):
            raise ValueError(
                f"gradient must be 'autograd' or 'fd', got {gradient!r}"
            )

        if bounds[1] <= bounds[0]:
            raise ValueError(
                f"bounds[1] ({bounds[1]}) must be greater than bounds[0] ({bounds[0]})"
            )

        self._params = _collect_params(params)
        self._shapes, self._sizes = _get_shapes_sizes(self._params)
        self._client = client
        self._bounds = bounds
        self._budget_steps = budget_steps
        self._gradient_mode = gradient
        self._on_step = on_step
        self._eval_timeout_s = eval_timeout_s
        self._verify_ssl = verify_ssl
        self._last_x: Optional[list[float]] = None

        # Public read-only metadata
        self.param_count_: int = sum(self._sizes)

        # Set after step(); None before first call
        self.last_loss_: Optional[float] = None
        self.stream_result_: Any = None

        if self.param_count_ > 1000:
            raise ValueError(
                f"EOILOptimizer: model has {self.param_count_} parameters, "
                "which exceeds the EOIL server hard limit of 1000. "
                "Consider optimising only the final layer, or using a smaller model."
            )

        if self.param_count_ > _FD_DIM_WARN and gradient == "fd":
            warnings.warn(
                f"EOILOptimizer: gradient='fd' with {self.param_count_} parameters "
                f"costs {2 * self.param_count_ + 1} closure calls per EOIL step. "
                f"At budget_steps={budget_steps} that is approximately "
                f"{budget_steps * (2 * self.param_count_ + 1):,} total closure calls. "
                "Use gradient='autograd' for much better performance.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, closure: Callable) -> float:
        """Run one full EOIL session and write the optimised parameters back.

        Parameters
        ----------
        closure:
            A zero-argument callable that evaluates the model loss.
            Must return a ``torch.Tensor`` scalar when
            ``gradient='autograd'``; may return a plain ``float`` when
            ``gradient='fd'``.

        Returns
        -------
        float
            The best loss value found (``stream_result_.f_best``).

        Raises
        ------
        ValueError
            If the server returns non-finite values in ``x_next`` or
            ``x_best``, or if the closure returns a non-finite loss.
        TypeError
            If ``gradient='autograd'`` but the closure does not return a
            ``torch.Tensor``.
        """
        # Snapshot current params — used to restore on exception
        x0_snapshot = _flatten(self._params)

        # Warn if current params are already outside bounds
        lo, hi = self._bounds
        if any(v < lo or v > hi for v in x0_snapshot):
            warnings.warn(
                f"EOILOptimizer: some model parameters are outside the declared "
                f"bounds ({lo}, {hi}). Values will be clipped before each "
                "evaluation. Consider re-initialising the model within bounds.",
                UserWarning,
                stacklevel=2,
            )

        # x0 warm-start — reuse last x_best when possible
        x0_for_session = (
            self._last_x if self._last_x is not None else x0_snapshot
        )
        if len(x0_for_session) > _X0_URL_LIMIT:
            warnings.warn(
                f"EOILOptimizer: dimension={len(x0_for_session)} > {_X0_URL_LIMIT}. "
                "x0 warm-start omitted: parameter vector too large to pass as initial point. "
                "The server will use a random starting point.",
                UserWarning,
                stacklevel=2,
            )
            x0_kwarg = None
        else:
            x0_kwarg = x0_for_session

        # Capture locals for closure scope
        params = self._params
        bounds = self._bounds
        shapes = self._shapes
        sizes = self._sizes

        # ------------------------------------------------------------------
        # Build objective function
        # ------------------------------------------------------------------

        if self._gradient_mode == "autograd":

            def _objective(x_raw: Any) -> tuple[float, list[float]]:
                x = list(x_raw) if not isinstance(x_raw, list) else x_raw
                if not all(math.isfinite(v) for v in x):
                    raise ValueError(
                        "EOILOptimizer: received non-finite (NaN/Inf) values "
                        "from the EOIL server in x_next."
                    )
                _unflatten(x, params, shapes, sizes, bounds)
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                with torch.enable_grad():
                    loss = closure()
                if not torch.is_tensor(loss):
                    raise TypeError(
                        "EOILOptimizer: closure() must return a torch.Tensor "
                        f"when gradient='autograd'. Got {type(loss).__name__!r}. "
                        "Use gradient='fd' for non-differentiable closures."
                    )
                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    raise ValueError(
                        f"EOILOptimizer: closure() returned non-finite loss: "
                        f"{loss_val}. Check for numerical instabilities."
                    )
                loss.backward()
                grad_flat: list[float] = []
                for p in params:
                    if p.grad is not None:
                        grad_flat.extend(
                            p.grad.data.detach().cpu().reshape(-1).tolist()
                        )
                    else:
                        grad_flat.extend([0.0] * p.data.numel())
                return loss_val, grad_flat

            stream_gradient: Union[bool, str] = True

        else:  # gradient == "fd"

            def _objective(x_raw: Any) -> float:  # type: ignore[misc]
                x = list(x_raw) if not isinstance(x_raw, list) else x_raw
                if not all(math.isfinite(v) for v in x):
                    raise ValueError(
                        "EOILOptimizer: received non-finite (NaN/Inf) values "
                        "from the EOIL server in x_next."
                    )
                _unflatten(x, params, shapes, sizes, bounds)
                with torch.no_grad():
                    loss = closure()
                loss_val = (
                    loss.item() if torch.is_tensor(loss) else float(loss)
                )
                if not math.isfinite(loss_val):
                    raise ValueError(
                        f"EOILOptimizer: closure() returned non-finite loss: "
                        f"{loss_val}."
                    )
                return loss_val

            stream_gradient = False

        # ------------------------------------------------------------------
        # Call EOIL, restore params on failure
        # ------------------------------------------------------------------

        try:
            result = self._client.stream_optimize(
                fn=_objective,
                dimension=self.param_count_,
                budget_steps=self._budget_steps,
                bounds=self._bounds,
                x0=x0_kwarg,
                gradient=stream_gradient,
                on_step=self._on_step,
                eval_timeout_s=self._eval_timeout_s,
                verify_ssl=self._verify_ssl,
            )
        except Exception:
            _write_params(x0_snapshot, params, shapes, sizes)
            raise

        # Guard against NaN/Inf in x_best from server
        if not all(math.isfinite(v) for v in result.x_best):
            _write_params(x0_snapshot, params, shapes, sizes)
            raise ValueError(
                "EOILOptimizer: x_best returned from EOIL contains non-finite "
                "values. Model parameters restored to pre-step values."
            )

        # Write optimised params back (bounds-clipped)
        _unflatten(result.x_best, params, shapes, sizes, bounds)
        self._last_x = result.x_best
        self.last_loss_ = result.f_best
        self.stream_result_ = result

        return result.f_best

    def zero_grad(self) -> None:
        """No-op — provided for API compatibility with ``torch.optim.Optimizer``."""
