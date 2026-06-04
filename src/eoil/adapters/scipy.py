"""scipy adapter for EOIL — drop-in replacement for scipy.optimize.minimize.

Usage::

    from eoil.adapters.scipy import minimize
    import eoil

    client = eoil.Client(api_key="eoil_sk_...")

    def my_fn(x):
        f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        grad = [
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2),
        ]
        return f, grad

    result = minimize(my_fn, x0=[0.0, 0.0], client=client)
    print(result.x, result.fun, result.success)

The function signature mirrors ``scipy.optimize.minimize`` for the
``fun``, ``x0``, ``jac``, and ``bounds`` arguments.  Options that have
no EOIL equivalent (``method``, ``hess``, ``hessp``, ``constraints``,
``tol``) are silently ignored so that callers can swap in this adapter
with minimal code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Optional scipy import (for type hints only at module level)
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import Bounds as _ScipyBounds  # type: ignore[import]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _ScipyBounds = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    import eoil as _eoil_pkg

__all__ = ["minimize", "OptimizeResult"]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class OptimizeResult:
    """Scipy-compatible result object returned by :func:`minimize`.

    All standard ``scipy.optimize.OptimizeResult`` attributes are populated
    so that callers can switch between scipy and this adapter transparently.

    Attributes
    ----------
    x:
        Best solution vector (``list[float]``).
    fun:
        Best objective value (``float``).
    success:
        ``True`` if the optimizer converged.
    nit:
        Number of function evaluations consumed.
    message:
        Human-readable status description.
    eoil_charged:
        Compute units charged by EOIL (``str``), e.g. ``"3.5"``.
    raw:
        The underlying :class:`eoil.StreamResult` for full access to
        EOIL-specific fields.
    """

    def __init__(
        self,
        *,
        x: list[float],
        fun: float,
        success: bool,
        nit: int,
        message: str,
        eoil_charged: str,
        raw: Any,
    ) -> None:
        self.x = x
        self.fun = fun
        self.success = success
        self.nit = nit
        self.nfev = nit  # scipy alias
        self.message = message
        self.eoil_charged = eoil_charged
        self.raw = raw
        # scipy sets status=0 for success, 1 for failure
        self.status = 0 if success else 1

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(x={self.x}, fun={self.fun!r}, success={self.success}, "
            f"nit={self.nit}, message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# Bounds helpers
# ---------------------------------------------------------------------------

def _parse_bounds(
    bounds: Any,
    dimension: int,
) -> tuple[float, float]:
    """Convert scipy-style bounds to a scalar EOIL ``(lo, hi)`` pair.

    EOIL applies box bounds uniformly across all dimensions.  If the caller
    supplies per-dimension bounds, the tightest enclosing box is used and a
    warning is emitted so the caller is aware of the approximation.

    Accepted formats
    ----------------
    - ``None``  → ``(-5.0, 5.0)`` default
    - ``(lo, hi)``  scalar pair
    - ``[(lo_0, hi_0), ..., (lo_n, hi_n)]``  per-dimension list
    - ``scipy.optimize.Bounds(lb, ub)``  scipy Bounds object
    """
    import warnings

    if bounds is None:
        return (-5.0, 5.0)

    # scipy.optimize.Bounds object
    if _HAS_SCIPY and isinstance(bounds, _ScipyBounds):
        lb = list(bounds.lb) if hasattr(bounds.lb, "__iter__") else [bounds.lb] * dimension
        ub = list(bounds.ub) if hasattr(bounds.ub, "__iter__") else [bounds.ub] * dimension
        per_dim = list(zip(lb, ub))
        return _collapse_bounds(per_dim, dimension)

    # Sequence of (lo, hi) pairs
    if isinstance(bounds, (list, tuple)) and len(bounds) > 0:
        first = bounds[0]
        if isinstance(first, (list, tuple)):
            # per-dimension
            return _collapse_bounds(list(bounds), dimension)
        # scalar pair
        if len(bounds) == 2 and isinstance(first, (int, float)):
            return (float(bounds[0]), float(bounds[1]))

    raise ValueError(
        f"Unsupported bounds format: {bounds!r}. "
        "Pass a (lo, hi) tuple, a list of (lo, hi) per dimension, "
        "or a scipy.optimize.Bounds object."
    )


def _collapse_bounds(
    per_dim: list[tuple[float, float]],
    dimension: int,
) -> tuple[float, float]:
    """Collapse per-dimension bounds to the tightest enclosing box."""
    import warnings

    lo_vals = [float(lo) for lo, _ in per_dim]
    hi_vals = [float(hi) for _, hi in per_dim]
    lo = min(lo_vals)
    hi = max(hi_vals)

    # Check whether the collapse is lossy
    if len(set(lo_vals)) > 1 or len(set(hi_vals)) > 1:
        warnings.warn(
            "EOIL applies box bounds uniformly across all dimensions. "
            f"Per-dimension bounds have been collapsed to ({lo}, {hi}). "
            "Consider pre-scaling your variables so a uniform box is accurate.",
            UserWarning,
            stacklevel=4,
        )

    return (lo, hi)


# ---------------------------------------------------------------------------
# Objective wrapper: handle jac=True / jac=callable / no jac
# ---------------------------------------------------------------------------

def _wrap_fn(
    fun: Callable,
    jac: Union[bool, Callable, None],
) -> tuple[Callable, str]:
    """Return (wrapped_fn, gradient_mode) for use with stream_optimize.

    scipy ``jac`` semantics:
    - ``False`` or ``None``  → fn returns scalar only → FD
    - ``True``               → fn returns (f, grad) tuple
    - callable               → separate gradient function

    Returns
    -------
    wrapped_fn:
        Callable that returns either ``float`` or ``(float, list[float])``.
    gradient_mode:
        One of ``"auto"``, ``True``, ``False`` for stream_optimize's
        ``gradient`` parameter.
    """
    if jac is None or jac is False:
        # Pure scalar fn; let stream_optimize use finite differences
        return fun, "auto"

    if jac is True:
        # fn already returns (f, grad)
        return fun, True  # type: ignore[return-value]

    if callable(jac):
        # Separate gradient callable — wrap into a single function
        def _combined(x: Any) -> tuple[float, Any]:
            return float(fun(x)), jac(x)  # type: ignore[operator]

        return _combined, True  # type: ignore[return-value]

    raise ValueError(
        f"jac must be True, False, None, or a callable; got {jac!r}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def minimize(
    fun: Callable,
    x0: Sequence[float],
    *,
    client: "eoil.Client",  # type: ignore[name-defined]
    jac: Union[bool, Callable, None] = None,
    bounds: Any = None,
    budget_steps: int = 500,
    options: Optional[dict[str, Any]] = None,
    on_step: Optional[Callable] = None,
    verify_ssl: bool = True,
    # Ignored scipy-compat kwargs — accepted to allow transparent swap-in
    method: Any = None,  # noqa: ARG001
    hess: Any = None,  # noqa: ARG001
    hessp: Any = None,  # noqa: ARG001
    constraints: Any = None,  # noqa: ARG001
    tol: Any = None,  # noqa: ARG001
    callback: Any = None,  # noqa: ARG001
    args: Any = None,  # noqa: ARG001
) -> OptimizeResult:
    """Minimise a scalar function using the EOIL optimizer.

    This is a near drop-in replacement for ``scipy.optimize.minimize``.
    The ``method``, ``hess``, ``hessp``, ``constraints``, ``tol``,
    ``callback``, and ``args`` parameters are accepted but ignored.

    Parameters
    ----------
    fun:
        Objective function.  Called as ``fun(x)`` where ``x`` is a
        ``list[float]`` or ``np.ndarray``.  May return a scalar ``float``
        **or** a ``(float, list[float])`` tuple when ``jac=True``.
    x0:
        Initial guess.  Length determines the problem dimension.
    client:
        An authenticated :class:`eoil.Client` instance.  **Required.**
    jac:
        Gradient specification.  Mirrors ``scipy.optimize.minimize``:

        - ``None`` / ``False``  — use finite differences (default)
        - ``True``              — ``fun`` returns ``(f, grad)``
        - callable             — separate ``jac(x)`` function

    bounds:
        Box constraints.  Accepted formats:

        - ``(lo, hi)``  — uniform box applied to all dimensions
        - ``[(lo_0, hi_0), ...]``  — per-dimension; collapsed to uniform box
        - ``scipy.optimize.Bounds`` — converted to uniform box

        Defaults to ``(-5.0, 5.0)`` if omitted.
    budget_steps:
        Approximate number of function evaluations.  Default ``500``.
    options:
        Dict of extra options.  Currently unused; reserved for future tuning.
    on_step:
        Optional callback ``on_step(step, x, f_best)`` called each evaluation.
    verify_ssl:
        Set ``False`` to disable SSL certificate verification (useful for
        staging environments).

    Returns
    -------
    OptimizeResult
        A scipy-compatible result with ``.x``, ``.fun``, ``.success``,
        ``.nit``, ``.message``, and EOIL-specific ``.eoil_charged`` / ``.raw``.

    Examples
    --------
    ::

        from eoil.adapters.scipy import minimize
        import eoil

        client = eoil.Client(api_key="eoil_sk_...")

        def rosenbrock(x):
            return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

        result = minimize(rosenbrock, x0=[0.0, 0.0], client=client, bounds=(-5.0, 5.0))
        print(result.x)       # [1.0, 1.0]
        print(result.success) # True
    """
    x0_list = list(x0)
    dimension = len(x0_list)
    if dimension == 0:
        raise ValueError("x0 must not be empty.")

    parsed_bounds = _parse_bounds(bounds, dimension)
    wrapped_fn, gradient_mode = _wrap_fn(fun, jac)

    stream_result = client.stream_optimize(
        fn=wrapped_fn,
        dimension=dimension,
        budget_steps=budget_steps,
        bounds=parsed_bounds,
        x0=x0_list,
        gradient=gradient_mode,
        on_step=on_step,
        verify_ssl=verify_ssl,
    )

    message = "Optimization converged." if stream_result.converged else "Optimization did not converge."

    return OptimizeResult(
        x=stream_result.x_best,
        fun=stream_result.f_best,
        success=stream_result.converged,
        nit=stream_result.total_steps,
        message=message,
        eoil_charged=stream_result.eoil_charged,
        raw=stream_result,
    )
