"""Pydantic models for EOIL SDK request/response types."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EoilError(Exception):
    """Base class for all EOIL SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(EoilError):
    """Raised when the API key is missing, invalid, or lacks required scopes."""


class InsufficientBalanceError(EoilError):
    """Raised when the account has insufficient EOIL credit balance."""

    def __init__(self, message: str, balance_eoil: Optional[str] = None, required_eoil: Optional[str] = None) -> None:
        super().__init__(message, status_code=402)
        self.balance_eoil = balance_eoil
        self.required_eoil = required_eoil


class RateLimitError(EoilError):
    """Raised when the rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Retry after a moment.") -> None:
        super().__init__(message, status_code=429)


class OptimizerError(EoilError):
    """Raised when the optimizer service returns a failure."""


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

class OptimizeResult(BaseModel):
    """Result returned by Client.optimize()."""

    job_id: str = Field(..., description="EOIL job ID")
    request_id: str = Field(..., description="Server-side request ID for support")
    status: JobStatus

    # Solution (present when status == succeeded)
    x_best: Optional[List[float]] = Field(None, description="Best solution vector found")
    f_best: Optional[float] = Field(None, description="Best objective value found")
    total_steps: Optional[int] = Field(None, description="Function evaluations used")
    converged: Optional[bool] = Field(None, description="Whether the solver converged")
    escapes: Optional[int] = Field(None, description="Number of basin escapes performed")
    time_ms: Optional[float] = Field(None, description="Wall-clock solve time in ms")
    compute_units_actual: Optional[int] = Field(None, description="Compute units consumed")

    # Billing
    fiat_currency: Optional[str] = None
    fiat_cost: Optional[str] = None
    eoil_charged: Optional[str] = None

    model_config = {"use_enum_values": True}
