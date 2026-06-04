from .client import Client
from .catalogue import CatalogueClient
from .models import OptimizationResult, OptimizeResult, JobStatus, StreamResult
from .exceptions import (
    EoilError,
    AuthError,
    InsufficientBalanceError,
    RateLimitError,
    OptimizerError,
    StreamError,
    SessionExpiredError,
)
from .presets import PRESETS, get_preset
from .autograd import finite_difference_gradient

__version__ = "0.5.0"

__all__ = [
    "Client",
    "CatalogueClient",
    "OptimizationResult",
    "OptimizeResult",
    "StreamResult",
    "JobStatus",
    "EoilError",
    "AuthError",
    "InsufficientBalanceError",
    "RateLimitError",
    "OptimizerError",
    "StreamError",
    "SessionExpiredError",
    "PRESETS",
    "get_preset",
    "finite_difference_gradient",
]
