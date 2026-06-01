"""EOIL API client."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from .exceptions import (
    AuthError,
    EoilError,
    InsufficientBalanceError,
    OptimizerError,
    RateLimitError,
)
from .models import (
    OBJECTIVE_TYPES,
    JobStatus,
    OptimizationResult,
)
from .catalogue import CatalogueClient

_DEFAULT_BASE_URL = "https://api.eoil.ltd"
_DEFAULT_TIMEOUT = 120.0  # seconds

# Translates public-facing preset keys to the internal wire-format names.
_PRESET_KEY_MAP: dict[str, str] = {
    "restarts": "num_restarts",
    "patience": "basin_escape_threshold",
    "depth": "sorf_layers",
}


class Client:
    """
    EOIL API client.

    Parameters
    ----------
    api_key:
        Your EOIL API key (``eoil_sk_...``). Required.
    base_url:
        Override the API base URL. Defaults to ``https://api.eoil.ltd``.
    timeout:
        HTTP timeout in seconds. Defaults to 120.

    Example
    -------
    ::

        import eoil

        client = eoil.Client(api_key="eoil_sk_...")
        result = client.optimize(
            objective_type="rastrigin",
            dimension=10,
            bounds=(-5.12, 5.12),
        )
        print(result.x_best, result.f_best)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise AuthError("api_key must not be empty.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"eoil-python/0.2.0a2",
            },
            timeout=timeout,
        )
        self._catalogue = CatalogueClient(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def catalogue(self) -> "CatalogueClient":
        """Access the catalogue of named landscape methods."""
        return self._catalogue

    def optimize(
        self,
        *,
        objective_type: str = "sphere",
        dimension: int = 2,
        bounds: Optional[Union[Tuple[float, float], List[float]]] = None,
        x0: Optional[List[float]] = None,
        budget_steps: int = 1000,
        compute_units: int = 100,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Submit an optimisation job and wait for the result (inline mode).

        Parameters
        ----------
        objective_type:
            Built-in landscape name. One of: ``sphere``, ``rastrigin``,
            ``rosenbrock``, ``quadratic``, ``ackley``, ``levy``, ``griewank``.
        dimension:
            Problem dimensionality (1–1000).
        bounds:
            Box bounds ``(lower, upper)`` applied uniformly to all dimensions.
        x0:
            Initial guess vector. Length must equal ``dimension``.
        budget_steps:
            Maximum function evaluations (100–100 000).
        compute_units:
            Estimated compute units for billing. Default 100.
        max_eoil:
            Spend cap in EOIL credits, e.g. ``"5.0"``. Job is rejected if
            estimated cost exceeds this.
        idempotency_key:
            Optional client-generated key for safe retries. Must be unique
            per logical request.

        Returns
        -------
        OptimizationResult
        """
        if objective_type not in OBJECTIVE_TYPES:
            raise ValueError(
                f"Unknown objective_type '{objective_type}'. "
                f"Must be one of: {', '.join(sorted(OBJECTIVE_TYPES))}"
            )
        if not 1 <= dimension <= 1000:
            raise ValueError("dimension must be between 1 and 1000.")
        if not 100 <= budget_steps <= 100_000:
            raise ValueError("budget_steps must be between 100 and 100 000.")

        return self._post_job(
            objective_type=objective_type,
            dimension=dimension,
            bounds=bounds,
            x0=x0,
            budget_steps=budget_steps,
            compute_units=compute_units,
            max_eoil=max_eoil,
            idempotency_key=idempotency_key,
            heuristics_override=heuristics_override,
        )

    def get_job(self, job_id: str) -> OptimizationResult:
        """
        Retrieve a previously submitted job by ID.

        Parameters
        ----------
        job_id:
            The job ID returned by :meth:`optimize` or :meth:`submit_job`.
        """
        response = self._http.get(f"/jobs/{job_id}")
        return self._parse_job_response(response)

    def submit_job(
        self,
        *,
        objective_type: str = "sphere",
        dimension: int = 2,
        bounds: Optional[Union[Tuple[float, float], List[float]]] = None,
        x0: Optional[List[float]] = None,
        budget_steps: int = 1000,
        compute_units: int = 100,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """
        Queue a job for async execution and return its job ID immediately.

        Use :meth:`get_job` to poll for the result.
        """
        if objective_type not in OBJECTIVE_TYPES:
            raise ValueError(
                f"Unknown objective_type '{objective_type}'. "
                f"Must be one of: {', '.join(sorted(OBJECTIVE_TYPES))}"
            )

        job_payload: Dict[str, Any] = {
            "objective_type": objective_type,
            "dimension": dimension,
            "budget_steps": budget_steps,
        }
        if bounds is not None:
            job_payload["bounds"] = list(bounds)
        if x0 is not None:
            job_payload["x0"] = list(x0)

        body: Dict[str, Any] = {
            "job": job_payload,
            "computeUnits": compute_units,
            "mode": "queue",
        }
        if max_eoil is not None:
            body["maxEoil"] = max_eoil

        extra_headers: Dict[str, str] = {}
        if idempotency_key is not None:
            extra_headers["Idempotency-Key"] = idempotency_key
        else:
            extra_headers["Idempotency-Key"] = str(uuid.uuid4())

        response = self._http.post("/jobs", json=body, headers=extra_headers)
        self._raise_for_status(response)
        data = response.json()
        return data["jobId"]

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_job(
        self,
        *,
        objective_type: str,
        dimension: int,
        bounds: Optional[Union[Tuple[float, float], List[float]]] = None,
        x0: Optional[List[float]] = None,
        budget_steps: int = 1000,
        compute_units: int = 100,
        max_eoil: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> "OptimizationResult":
        """Internal: build and POST a job payload, return parsed result."""
        job_payload: Dict[str, Any] = {
            "objective_type": objective_type,
            "dimension": dimension,
            "budget_steps": budget_steps,
        }
        if bounds is not None:
            job_payload["bounds"] = list(bounds)
        if x0 is not None:
            job_payload["x0"] = list(x0)
        if parameters is not None:
            job_payload["parameters"] = parameters
        if heuristics_override is not None:
            job_payload["heuristics_override"] = {
                _PRESET_KEY_MAP.get(k, k): v for k, v in heuristics_override.items()
            }

        body: Dict[str, Any] = {
            "job": job_payload,
            "computeUnits": compute_units,
            "mode": "inline",
        }
        if max_eoil is not None:
            body["maxEoil"] = max_eoil

        extra_headers: Dict[str, str] = {}
        if idempotency_key is not None:
            extra_headers["Idempotency-Key"] = idempotency_key
        else:
            extra_headers["Idempotency-Key"] = str(uuid.uuid4())

        response = self._http.post("/jobs", json=body, headers=extra_headers)
        return self._parse_optimize_response(response)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code in (200, 202):
            return
        try:
            body = response.json()
            error_msg = body.get("error") or body.get("message") or response.text
        except Exception:
            error_msg = response.text

        if response.status_code == 401:
            raise AuthError(f"Unauthorised: {error_msg}", status_code=401)
        if response.status_code == 403:
            raise AuthError(f"Forbidden (check API key scopes): {error_msg}", status_code=403)
        if response.status_code == 402:
            body_data: Dict[str, Any] = {}
            try:
                body_data = response.json()
            except Exception:
                pass
            raise InsufficientBalanceError(
                error_msg,
                balance_eoil=body_data.get("balanceEoil"),
                required_eoil=body_data.get("requiredEoil"),
            )
        if response.status_code == 429:
            raise RateLimitError()
        raise EoilError(f"API error {response.status_code}: {error_msg}", status_code=response.status_code)

    def _parse_optimize_response(self, response: httpx.Response) -> OptimizationResult:
        if response.status_code not in (200, 202):
            self._raise_for_status(response)

        data = response.json()

        if not data.get("ok", True):
            error_msg = data.get("error", "Unknown optimizer error")
            if response.status_code == 402:
                raise InsufficientBalanceError(
                    error_msg,
                    balance_eoil=data.get("balanceEoil"),
                    required_eoil=data.get("requiredEoil"),
                )
            raise OptimizerError(error_msg, status_code=response.status_code)

        result_data = data.get("result") or {}
        charged = data.get("charged") or {}

        return OptimizationResult(
            job_id=data["jobId"],
            request_id=data.get("requestId", ""),
            status=JobStatus(data.get("status", "succeeded")),
            x_best=result_data.get("x_best"),
            f_best=result_data.get("f_best"),
            total_steps=result_data.get("total_steps"),
            converged=result_data.get("converged"),
            escapes=result_data.get("escapes"),
            time_ms=result_data.get("time_ms"),
            compute_units_actual=result_data.get("compute_units_actual"),
            fiat_currency=charged.get("fiatCurrency"),
            fiat_cost=charged.get("fiatCost"),
            eoil_charged=charged.get("eoilCharged"),
        )

    def _parse_job_response(self, response: httpx.Response) -> OptimizationResult:
        self._raise_for_status(response)
        data = response.json()
        result_data = data.get("result") or {}
        charged = data.get("charged") or {}

        return OptimizationResult(
            job_id=data["jobId"],
            request_id=data.get("requestId", ""),
            status=JobStatus(data.get("status", "succeeded")),
            x_best=result_data.get("x_best"),
            f_best=result_data.get("f_best"),
            total_steps=result_data.get("total_steps"),
            converged=result_data.get("converged"),
            escapes=result_data.get("escapes"),
            time_ms=result_data.get("time_ms"),
            compute_units_actual=result_data.get("compute_units_actual"),
            fiat_currency=charged.get("fiatCurrency"),
            fiat_cost=charged.get("fiatCost"),
            eoil_charged=charged.get("eoilCharged"),
        )
