"""EOIL SDK exception hierarchy."""

from __future__ import annotations

from typing import Optional


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


class StreamError(EoilError):
    """Raised on WebSocket streaming protocol errors or unexpected server-side close codes."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class SessionExpiredError(StreamError):
    """Raised when the optimizer session has expired on the server (close code 4410)."""

    def __init__(self, message: str = "Optimizer session expired.") -> None:
        super().__init__("SESSION_GONE", message)
