"""Bidirectional WebSocket streaming client for client.optimize(fn, ...)."""

from __future__ import annotations

import json
import ssl
import time
import warnings
import concurrent.futures
from typing import Callable, List, Optional, Tuple, Union
from urllib.parse import urlencode

from .exceptions import (
    AuthError,
    EoilError,
    InsufficientBalanceError,
    RateLimitError,
    SessionExpiredError,
    StreamError,
)
from .models import StreamResult

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _coerce_x(raw: List[float]) -> Union["_np.ndarray", List[float]]:  # type: ignore[name-defined]
    if _HAS_NUMPY:
        return _np.array(raw, dtype=float)
    return raw


def _finite_difference_grad(
    fn: Callable,
    x: List[float],
    f0: float,
    h: float,
) -> List[float]:
    """Central finite differences, pure Python — no torch/numpy required."""
    grad = []
    for i in range(len(x)):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        f_plus = float(fn(_coerce_x(x_plus)))
        f_minus = float(fn(_coerce_x(x_minus)))
        grad.append((f_plus - f_minus) / (2.0 * h))
    return grad


class StreamSession:
    """
    Drives the WebSocket eval loop for client.optimize(fn, ...).

    Not intended to be used directly — call via Client.stream_optimize().
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        fn: Callable,
        dimension: int,
        budget_steps: int,
        bounds: Tuple[float, float],
        x0: Optional[List[float]],
        gradient: Union[bool, str],
        fd_step: float,
        on_step: Optional[Callable],
        timeout_s: float,
        eval_timeout_s: float,
        verify_ssl: bool = True,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._fn = fn
        self._dimension = dimension
        self._budget_steps = budget_steps
        self._bounds = bounds
        self._x0 = x0
        self._gradient = gradient
        self._fd_step = fd_step
        self._on_step = on_step
        self._timeout_s = timeout_s
        self._eval_timeout_s = eval_timeout_s
        self._verify_ssl = verify_ssl

        self._fd_warned: bool = False       # one-time FD warning flag

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> StreamResult:
        import websockets.sync.client  # lazy — only when actually streaming
        import websockets.exceptions

        ws_url = self._build_url()

        ssl_context: Optional[ssl.SSLContext] = None
        if not self._verify_ssl and ws_url.startswith("wss://"):
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        try:
            with websockets.sync.client.connect(
                ws_url,
                open_timeout=30,
                additional_headers={"User-Agent": "eoil-python/0.5.0"},
                ssl=ssl_context,
            ) as ws:
                deadline = time.monotonic() + self._timeout_s

                for raw in ws:
                    if time.monotonic() > deadline:
                        ws.send(json.dumps({"type": "cancel"}))
                        raise EoilError(
                            f"Session exceeded timeout_s={self._timeout_s}s wall-clock limit."
                        )

                    frame = json.loads(raw)
                    ftype = frame.get("type")

                    if ftype == "x":
                        self._handle_x(ws, frame)

                    elif ftype == "result":
                        return StreamResult(
                            x_best=frame["x_best"],
                            f_best=frame["f_best"],
                            total_steps=frame.get("total_steps", 0),
                            converged=frame.get("converged", False),
                            escapes=frame.get("escapes", 0),
                            compute_units_actual=frame.get("compute_units_actual"),
                            eoil_charged=frame.get("eoil_charged"),
                        )

                    elif ftype == "error":
                        code = frame.get("code", "UNKNOWN")
                        msg = frame.get("message", "")
                        if code == "SESSION_GONE":
                            raise SessionExpiredError(msg)
                        raise StreamError(code, msg)

                # Connection closed without a result frame
                raise StreamError("CONNECTION_CLOSED", "Server closed the connection without sending a result.")

        except websockets.exceptions.InvalidStatus as exc:
            self._raise_for_upgrade_rejection(exc)
            raise  # unreachable — satisfies type checker

        except websockets.exceptions.ConnectionClosedError as exc:
            code = exc.rcvd.code if exc.rcvd else None
            reason = exc.rcvd.reason if exc.rcvd else ""
            if code == 1011:
                raise StreamError(
                    "INACTIVITY_TIMEOUT",
                    "Server closed the session after 90s of inactivity. "
                    "Ensure your objective function returns within eval_timeout_s.",
                ) from exc
            if code == 4008:
                raise StreamError("PROTOCOL_ERROR", reason) from exc
            if code == 4410:
                raise SessionExpiredError(reason) from exc
            raise StreamError(f"CONNECTION_CLOSED_{code}", reason) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        # Convert http(s) base URL to ws(s)
        base = self._base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://"):]
        else:
            ws_base = base

        params: dict = {
            "key": self._api_key,
            "dimension": self._dimension,
            "budget_steps": self._budget_steps,
            "bounds_lo": self._bounds[0],
            "bounds_hi": self._bounds[1],
        }
        if self._x0 is not None:
            params["x0"] = json.dumps(self._x0)

        return f"{ws_base}/optimizer/stream?{urlencode(params)}"

    def _handle_x(self, ws, frame: dict) -> None:
        x_raw: List[float] = frame["x"]
        x = _coerce_x(x_raw)

        f, grad = self._call_fn_with_timeout(x, x_raw)

        ws.send(json.dumps({"type": "eval", "f": f, "grad": grad}))

        if self._on_step is not None:
            self._on_step(frame["step"], x_raw, frame.get("f_best"))

    def _call_fn_with_timeout(
        self, x: Union["_np.ndarray", List[float]], x_raw: List[float]  # type: ignore[name-defined]
    ) -> Tuple[float, List[float]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._fn, x)
            try:
                result = future.result(timeout=self._eval_timeout_s)
            except concurrent.futures.TimeoutError:
                raise EoilError(
                    f"Objective function exceeded eval_timeout_s={self._eval_timeout_s}s. "
                    "The server closes sessions after 90s of inactivity. "
                    "Increase eval_timeout_s or optimise your function's runtime."
                )

        # Resolve gradient
        if isinstance(result, tuple) and len(result) == 2:
            # User returned (f, grad)
            if self._gradient is True and not isinstance(result[0], (int, float)):
                raise StreamError(
                    "PROTOCOL_ERROR",
                    "gradient=True but fn returned something other than (float, list).",
                )
            f = float(result[0])
            grad = [float(v) for v in result[1]]
            return f, grad

        # Scalar return — use FD
        if self._gradient is True:
            raise StreamError(
                "GRADIENT_MISSING",
                "gradient=True but fn returned a scalar instead of (f, grad). "
                "Either return a tuple or set gradient='auto' or gradient=False.",
            )

        f = float(result)

        if not self._fd_warned:
            warnings.warn(
                "eoil: Using finite-difference gradient approximation. "
                f"This costs 2×dimension={2 * self._dimension} extra fn calls per step. "
                "Return (f, grad) from your function to avoid this.",
                stacklevel=6,
            )
            self._fd_warned = True

        grad = _finite_difference_grad(self._fn, x_raw, f, self._fd_step)
        return f, grad

    @staticmethod
    def _raise_for_upgrade_rejection(exc: Exception) -> None:
        """Map websockets InvalidStatus to typed EoilError subclasses."""
        # websockets>=14: exception has a .response attribute (http11.Response)
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response else getattr(exc, "status_code", None)
        body = ""
        try:
            body_bytes = getattr(response, "body", b"") if response else b""
            body = body_bytes.decode() if isinstance(body_bytes, bytes) else str(body_bytes)
        except Exception:
            pass

        if status == 401:
            raise AuthError(f"Invalid or missing API key. {body}".strip()) from exc
        if status == 403:
            raise AuthError(
                "API key is missing the optimizer:stream scope. "
                "Enable it in the dashboard under API Keys."
            ) from exc
        if status == 429:
            # Try to distinguish budget exhaustion from rate limit
            if "budget" in body.lower() or "balance" in body.lower():
                raise InsufficientBalanceError(
                    f"Compute budget exhausted or insufficient balance. {body}".strip()
                ) from exc
            raise RateLimitError(
                f"Concurrent session limit reached or rate limit exceeded. {body}".strip()
            ) from exc
        if status == 400:
            raise EoilError(f"Bad request (check dimension/budget_steps/bounds). {body}".strip()) from exc
        raise EoilError(f"WebSocket upgrade rejected with HTTP {status}. {body}".strip()) from exc
