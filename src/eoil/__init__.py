from .client import Client
from .catalogue import CatalogueClient
from .models import OptimizationResult, OptimizeResult, JobStatus
from .exceptions import EoilError, AuthError, InsufficientBalanceError, RateLimitError, OptimizerError
from .presets import PRESETS, get_preset

__version__ = "0.2.0a2"

__all__ = [
    "Client",
    "CatalogueClient",
    "OptimizationResult",
    "OptimizeResult",
    "JobStatus",
    "EoilError",
    "AuthError",
    "InsufficientBalanceError",
    "RateLimitError",
    "OptimizerError",
    "PRESETS",
    "get_preset",
]
