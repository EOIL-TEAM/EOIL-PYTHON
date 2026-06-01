"""
EOIL Python SDK

Usage:
    import eoil

    client = eoil.Client(api_key="eoil_sk_...")
    result = client.optimize(
        objective_type="rastrigin",
        dimension=10,
        bounds=(-5.12, 5.12),
        budget_steps=2000,
    )
    print(result.x_best, result.f_best)
"""

from .client import Client
from .models import OptimizeResult, JobStatus, EoilError, AuthError, InsufficientBalanceError, RateLimitError

__version__ = "0.2.0a1"

__all__ = [
    "Client",
    "OptimizeResult",
    "JobStatus",
    "EoilError",
    "AuthError",
    "InsufficientBalanceError",
    "RateLimitError",
]
