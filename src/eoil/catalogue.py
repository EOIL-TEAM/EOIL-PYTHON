"""Catalogue client — named methods for all built-in EOIL landscapes."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

from .models import OptimizationResult


def _to_nested_list(data: Any) -> List[List[float]]:
    """Coerce numpy array or nested list to a plain nested list of floats."""
    try:
        import numpy as np  # type: ignore
        if isinstance(data, np.ndarray):
            return data.tolist()
    except ImportError:
        pass
    if isinstance(data, (list, tuple)):
        return [list(row) for row in data]
    raise TypeError(f"Expected a 2-D array or nested list, got {type(data).__name__}")


def _to_flat_list(data: Any) -> List[float]:
    """Coerce 1-D numpy array or list to a plain list of floats."""
    try:
        import numpy as np  # type: ignore
        if isinstance(data, np.ndarray):
            return data.tolist()
    except ImportError:
        pass
    return [float(v) for v in data]


class CatalogueClient:
    """
    Named-method interface to all built-in EOIL landscapes.

    Accessed via ``client.catalogue``. Do not instantiate directly.

    Example
    -------
    ::

        import eoil

        with eoil.Client(api_key="eoil_sk_...") as client:
            # Benchmark
            result = client.catalogue.rastrigin(dimension=10, budget_steps=2000)
            print(result.f_best)

            # Portfolio
            result = client.catalogue.portfolio_sharpe(
                returns=my_returns_matrix,  # shape (T, N)
                dimension=N,
                budget_steps=5000,
            )
            print(result.x)  # optimal portfolio weights
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Portfolio landscapes
    # ------------------------------------------------------------------

    def portfolio_sharpe(
        self,
        returns: Any,
        *,
        dimension: Optional[int] = None,
        budget_steps: int = 2000,
        compute_units: int = 100,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Maximise portfolio Sharpe ratio.

        Parameters
        ----------
        returns:
            Asset returns matrix, shape ``(T, N)`` — T time periods, N assets.
            Accepts a nested list or numpy array.
        dimension:
            Number of assets (N). Inferred from ``returns`` if not provided.
        budget_steps:
            Maximum function evaluations. Default 2000.
        """
        returns_list = _to_nested_list(returns)
        n_assets = len(returns_list[0]) if returns_list else 0
        dim = dimension if dimension is not None else n_assets
        if dim < 2:
            raise ValueError("portfolio_sharpe requires at least 2 assets (dimension >= 2).")
        if not returns_list or len(returns_list[0]) != dim:
            raise ValueError(
                f"returns matrix must have {dim} columns (one per asset), "
                f"got {len(returns_list[0]) if returns_list else 0}."
            )
        return self._client._post_job(
            objective_type="portfolio_sharpe",
            dimension=dim,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            parameters={"returns": returns_list},
            heuristics_override=heuristics_override,
        )

    def portfolio_minvol(
        self,
        cov: Any,
        *,
        dimension: Optional[int] = None,
        budget_steps: int = 2000,
        compute_units: int = 100,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Minimise portfolio variance (minimum volatility portfolio).

        Parameters
        ----------
        cov:
            Asset covariance matrix, shape ``(N, N)``.
            Accepts a nested list or numpy array.
        dimension:
            Number of assets (N). Inferred from ``cov`` if not provided.
        budget_steps:
            Maximum function evaluations. Default 2000.
        """
        cov_list = _to_nested_list(cov)
        n_assets = len(cov_list)
        dim = dimension if dimension is not None else n_assets
        if dim < 2:
            raise ValueError("portfolio_minvol requires at least 2 assets (dimension >= 2).")
        if len(cov_list) != dim or any(len(row) != dim for row in cov_list):
            raise ValueError(
                f"cov must be a ({dim}, {dim}) square matrix."
            )
        return self._client._post_job(
            objective_type="portfolio_minvol",
            dimension=dim,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            parameters={"cov": cov_list},
            heuristics_override=heuristics_override,
        )

    # ------------------------------------------------------------------
    # Benchmark landscapes
    # ------------------------------------------------------------------

    def rastrigin(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = (-5.12, 5.12),
        budget_steps: int = 2000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Rastrigin benchmark — highly multimodal, global min at origin."""
        return self._client._post_job(
            objective_type="rastrigin",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def sphere(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = None,
        budget_steps: int = 1000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Sphere benchmark — simple convex, global min at origin."""
        return self._client._post_job(
            objective_type="sphere",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def rosenbrock(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = (-5.0, 10.0),
        budget_steps: int = 3000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Rosenbrock benchmark — narrow curved valley, global min at (1,...,1)."""
        return self._client._post_job(
            objective_type="rosenbrock",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def ackley(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = (-32.768, 32.768),
        budget_steps: int = 2000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Ackley benchmark — multimodal with exponential landscape."""
        return self._client._post_job(
            objective_type="ackley",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def levy(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = (-10.0, 10.0),
        budget_steps: int = 2000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Levy benchmark — multimodal, global min at (1,...,1)."""
        return self._client._post_job(
            objective_type="levy",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def griewank(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = (-600.0, 600.0),
        budget_steps: int = 2000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Griewank benchmark — regular distributed local minima."""
        return self._client._post_job(
            objective_type="griewank",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def quadratic(
        self,
        *,
        dimension: int = 10,
        bounds: Optional[tuple] = None,
        budget_steps: int = 1000,
        compute_units: int = 100,
        x0: Optional[List[float]] = None,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Coupled quadratic benchmark — ill-conditioned quadratic bowl."""
        return self._client._post_job(
            objective_type="quadratic",
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )
