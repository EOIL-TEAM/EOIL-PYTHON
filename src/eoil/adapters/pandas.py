"""pandas adapter for EOIL — EOILPortfolioOptimizer.

Optimise a portfolio of assets using EOIL global optimisation.

Usage::

    import pandas as pd
    import eoil
    from eoil.adapters.pandas import EOILPortfolioOptimizer

    client = eoil.Client(api_key="eoil_sk_...")

    # returns: DataFrame with rows=time periods, columns=asset names
    returns = pd.DataFrame({
        "AAPL": [0.01, -0.005, 0.008, ...],
        "MSFT": [0.007, 0.003, -0.002, ...],
        "GOOG": [-0.003, 0.012, 0.005, ...],
    })

    opt = EOILPortfolioOptimizer(
        returns,
        client=client,
        objective="sharpe",   # 'sharpe' | 'min_variance' | 'max_return'
        budget_steps=300,
    )
    opt.fit()

    print(opt.weights_)          # pd.Series — asset → weight, sums to 1
    print(opt.sharpe_ratio_)     # float
    print(opt.portfolio_return_) # float — annualised
    print(opt.portfolio_volatility_) # float — annualised

Reparameterisation
------------------
EOIL optimises unconstrained logits ``z`` of shape ``(n_assets,)``.
Portfolio weights are derived via ``softmax(z)``, ensuring:

* all weights are strictly positive
* weights sum exactly to 1
* EOIL sees a smooth, unconstrained search space with no simplex projection

The dimension of the EOIL problem equals the number of assets.

Notes
-----
* Returns are assumed to be **daily** and are annualised with a 252-trading-day
  factor.  Pass ``periods_per_year`` to change this.
* ``gradient=False`` is used throughout — portfolio metrics are computed via
  numpy/pandas and have no analytic gradient available to the adapter.
* Raises ``ValueError`` if ``returns`` contains NaN values.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Optional

try:
    import numpy as np
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

__all__ = ["EOILPortfolioOptimizer"]

_VALID_OBJECTIVES = ("sharpe", "min_variance", "max_return")

# ── helpers (exposed for unit tests) ─────────────────────────────────────────


def _softmax(z: "np.ndarray") -> "np.ndarray":
    """Numerically stable softmax."""
    e = np.exp(z - np.max(z))
    return e / e.sum()


def _portfolio_stats(
    weights: "np.ndarray",
    returns_matrix: "np.ndarray",
    periods_per_year: float,
) -> tuple[float, float, float]:
    """Return (annualised_return, annualised_vol, sharpe) for given weights."""
    port_returns = returns_matrix @ weights          # (T,)
    ann_ret = float(port_returns.mean() * periods_per_year)
    ann_vol = float(port_returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    return ann_ret, ann_vol, sharpe


def _build_objective(
    objective: str,
    returns_matrix: "np.ndarray",
    periods_per_year: float,
    risk_free_rate: float,
) -> Callable[["np.ndarray"], float]:
    """Return a scalar objective function over the logit vector z."""

    def _fn(z: "np.ndarray") -> float:
        w = _softmax(z)
        ann_ret, ann_vol, _ = _portfolio_stats(w, returns_matrix, periods_per_year)
        if objective == "sharpe":
            sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0
            return -sharpe                  # EOIL minimises
        elif objective == "min_variance":
            return ann_vol ** 2
        else:  # max_return
            return -ann_ret

    return _fn


# ── EOILPortfolioOptimizer ────────────────────────────────────────────────────


class EOILPortfolioOptimizer:
    """Use EOIL global optimisation to find optimal portfolio weights.

    Parameters
    ----------
    returns:
        A ``pd.DataFrame`` where rows are time periods and columns are asset
        names.  Values should be simple period returns (e.g. 0.01 for +1%).
        Must contain no NaN values.
    client:
        An authenticated ``eoil.Client`` instance.
    objective:
        Optimisation target.  One of:

        * ``"sharpe"`` (default) — maximise Sharpe ratio
        * ``"min_variance"`` — minimise annualised variance
        * ``"max_return"`` — maximise annualised return
    budget_steps:
        Number of EOIL optimisation steps.
    risk_free_rate:
        Annualised risk-free rate used in the Sharpe ratio denominator.
        Ignored for ``min_variance`` and ``max_return``.
    periods_per_year:
        Number of return periods per year, used for annualisation.
        Default ``252`` (daily returns).
    bounds:
        ``(lo, hi)`` box bounds on the unconstrained logit vector.  The
        logit space is smooth and bounded; ``(-5.0, 5.0)`` covers softmax
        weights from ~0.003 to ~0.997 for most portfolio sizes.
    on_step:
        Optional callable forwarded to ``client.stream_optimize``.
    verify_ssl:
        Whether to verify the TLS certificate of the EOIL WebSocket endpoint.
    """

    def __init__(
        self,
        returns: "pd.DataFrame",
        *,
        client: Any,
        objective: str = "sharpe",
        budget_steps: int = 300,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0,
        bounds: tuple[float, float] = (-5.0, 5.0),
        on_step: Optional[Callable] = None,
        verify_ssl: bool = True,
    ) -> None:
        if not _HAS_PANDAS:  # pragma: no cover
            raise ImportError(
                "pandas and numpy are required for EOILPortfolioOptimizer. "
                "Install with: pip install 'eoil[pandas]'"
            )

        if not isinstance(returns, pd.DataFrame):
            raise TypeError(
                f"returns must be a pd.DataFrame, got {type(returns).__name__!r}"
            )

        if returns.empty:
            raise ValueError("returns DataFrame must not be empty.")

        if returns.shape[1] < 2:
            raise ValueError(
                f"returns must have at least 2 assets (columns), "
                f"got {returns.shape[1]}."
            )

        if returns.isnull().any().any():
            raise ValueError(
                "returns DataFrame contains NaN values. "
                "Forward-fill or drop missing data before fitting."
            )

        if objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_VALID_OBJECTIVES}, "
                f"got {objective!r}."
            )

        if bounds[1] <= bounds[0]:
            raise ValueError(
                f"bounds[1] ({bounds[1]}) must be greater than bounds[0] ({bounds[0]})"
            )

        n_assets = returns.shape[1]
        if n_assets > 1000:
            raise ValueError(
                f"EOILPortfolioOptimizer: {n_assets} assets exceeds the "
                "EOIL server hard limit of 1000."
            )

        self._returns_matrix = returns.values.astype(float)
        self._asset_names = list(returns.columns)
        self._client = client
        self._objective = objective
        self._budget_steps = budget_steps
        self._risk_free_rate = risk_free_rate
        self._periods_per_year = periods_per_year
        self._bounds = bounds
        self._on_step = on_step
        self._verify_ssl = verify_ssl

        self.n_assets_: int = n_assets

        # Set after fit(); None before
        self.weights_: Optional["pd.Series"] = None
        self.stream_result_: Any = None
        self.portfolio_return_: Optional[float] = None
        self.portfolio_volatility_: Optional[float] = None
        self.sharpe_ratio_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> "EOILPortfolioOptimizer":
        """Run the EOIL optimisation session and compute optimal weights.

        Returns
        -------
        self
            Returns the fitted optimizer for chaining.

        Raises
        ------
        ValueError
            If the server returns non-finite values in ``x_best``.
        """
        returns_matrix = self._returns_matrix
        periods_per_year = self._periods_per_year

        obj_fn = _build_objective(
            self._objective,
            returns_matrix,
            periods_per_year,
            self._risk_free_rate,
        )

        def _wrapped(x_raw: Any) -> float:
            z = np.array(list(x_raw) if not isinstance(x_raw, list) else x_raw, dtype=float)
            if not np.all(np.isfinite(z)):
                raise ValueError(
                    "EOILPortfolioOptimizer: received non-finite (NaN/Inf) "
                    "values from the EOIL server."
                )
            return obj_fn(z)

        # x0 warm-start: uniform logits → equal weights
        x0 = [0.0] * self.n_assets_

        result = self._client.stream_optimize(
            fn=_wrapped,
            dimension=self.n_assets_,
            budget_steps=self._budget_steps,
            bounds=self._bounds,
            x0=x0,
            gradient=False,
            on_step=self._on_step,
            verify_ssl=self._verify_ssl,
        )

        if not all(math.isfinite(v) for v in result.x_best):
            raise ValueError(
                "EOILPortfolioOptimizer: x_best from EOIL contains "
                "non-finite values."
            )

        lo, hi = self._bounds
        z_best = np.clip(np.array(result.x_best, dtype=float), lo, hi)
        weights = _softmax(z_best)

        ann_ret, ann_vol, sharpe = _portfolio_stats(
            weights, returns_matrix, periods_per_year
        )
        adjusted_sharpe = (ann_ret - self._risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0

        self.weights_ = pd.Series(weights, index=self._asset_names, name="weight")
        self.stream_result_ = result
        self.portfolio_return_ = ann_ret
        self.portfolio_volatility_ = ann_vol
        self.sharpe_ratio_ = adjusted_sharpe

        return self
