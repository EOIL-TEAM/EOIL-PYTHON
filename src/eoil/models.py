"""Pydantic models for EOIL SDK request/response types."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from .exceptions import (  # noqa: F401 — re-exported for backwards compat
    EoilError,
    AuthError,
    InsufficientBalanceError,
    RateLimitError,
    OptimizerError,
)


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Objective types
# ---------------------------------------------------------------------------

OBJECTIVE_TYPES = frozenset(
    ["sphere", "rastrigin", "rosenbrock", "quadratic", "ackley", "levy", "griewank"]
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class OptimizationResult(BaseModel):
    """Result returned by Client.optimize()."""

    job_id: str = Field(..., description="EOIL job ID")
    request_id: str = Field(..., description="Server-side request ID for support")
    status: JobStatus

    # Solution (present when status == succeeded)
    x_best: Optional[List[float]] = Field(None, description="Best solution vector found")
    f_best: Optional[float] = Field(None, description="Best objective value found")
    total_steps: Optional[int] = Field(None, description="Function evaluations used")
    converged: Optional[bool] = Field(None, description="Whether the solver converged")
    escapes: Optional[int] = Field(None, description="Solver iterations")
    time_ms: Optional[float] = Field(None, description="Wall-clock solve time in ms")
    compute_units_actual: Optional[int] = Field(None, description="Compute units consumed")

    # Billing
    fiat_currency: Optional[str] = None
    fiat_cost: Optional[str] = None
    eoil_charged: Optional[str] = None
    credits_used: Optional[str] = Field(None, description="Alias for eoil_charged")

    model_config = {"use_enum_values": True}

    @property
    def x(self) -> Optional[List[float]]:
        """Alias for x_best."""
        return self.x_best

    @property
    def f(self) -> Optional[float]:
        """Alias for f_best."""
        return self.f_best

    @property
    def evals(self) -> Optional[int]:
        """Alias for total_steps."""
        return self.total_steps


# Backwards-compatibility alias
OptimizeResult = OptimizationResult
