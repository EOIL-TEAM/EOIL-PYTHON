"""Tests for StreamSession / client.stream_optimize()."""

from __future__ import annotations

import json
import threading
from typing import Generator

import pytest

import sys
import types

# ---------------------------------------------------------------------------
# Minimal websockets stub so tests run without the real package installed
# when running in CI without the dep.  If websockets IS installed the real
# package is used instead.
# ---------------------------------------------------------------------------
try:
    import websockets  # noqa: F401
    _WEBSOCKETS_REAL = True
except ImportError:
    _WEBSOCKETS_REAL = False


# ---------------------------------------------------------------------------
# Fixtures — mock WS server using the real websockets library
# ---------------------------------------------------------------------------

def _run_mock_server(handler, host="127.0.0.1", port=0, ready_event=None, port_holder=None):
    """Run a websockets server in a background thread. Calls handler(ws) per connection."""
    import asyncio
    import websockets.asyncio.server

    async def _serve():
        async with websockets.asyncio.server.serve(handler, host, port) as server:
            addr = server.sockets[0].getsockname()
            port_holder[0] = addr[1]
            ready_event.set()
            await asyncio.get_event_loop().create_future()  # run forever

    asyncio.run(_serve())


@pytest.fixture
def mock_server():
    """
    Returns a factory: `make_server(handler)` → `ws_url`.
    The handler is an async function `async def handler(websocket): ...`
    """
    import websockets.server

    def make_server(handler):
        ready = threading.Event()
        port_holder = [None]
        t = threading.Thread(
            target=_run_mock_server,
            args=(handler,),
            kwargs={"ready_event": ready, "port_holder": port_holder},
            daemon=True,
        )
        t.start()
        ready.wait(timeout=5)
        return f"ws://127.0.0.1:{port_holder[0]}"

    return make_server


# ---------------------------------------------------------------------------
# Helper: build a StreamSession pointed at a local URL
# ---------------------------------------------------------------------------

def make_session(fn, ws_url, **kwargs):
    from eoil.stream import StreamSession
    defaults = dict(
        api_key="test-key",
        base_url=ws_url.replace("ws://", "http://").replace("wss://", "https://"),
        fn=fn,
        dimension=2,
        budget_steps=100,
        bounds=(-5.0, 5.0),
        x0=None,
        gradient="auto",
        fd_step=1e-5,
        on_step=None,
        timeout_s=30.0,
        eval_timeout_s=10.0,
    )
    defaults.update(kwargs)
    # Patch _build_url to use the provided ws_url directly
    session = StreamSession(**defaults)
    session._build_url = lambda: ws_url
    return session


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_scalar_fn_activates_fd_and_warns(mock_server):
    """fn returns float → FD activated, warning emitted, correct eval frames sent."""
    received_frames = []

    async def handler(ws):
        # Send one x frame then a result
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [1.0, 2.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        msg = await ws.recv()
        received_frames.append(json.loads(msg))
        await ws.send(json.dumps({"type": "result", "x_best": [1.0, 2.0], "f_best": 0.5, "total_steps": 1, "converged": False, "escapes": 0}))

    url = mock_server(handler)

    def scalar_fn(x):
        return float(x[0] ** 2 + x[1] ** 2)

    session = make_session(scalar_fn, url, gradient="auto")

    with pytest.warns(UserWarning, match="finite-difference"):
        result = session.run()

    assert result.total_steps == 1
    assert result.f_best == 0.5
    # eval frame should have f and grad
    assert len(received_frames) == 1
    frame = received_frames[0]
    assert frame["type"] == "eval"
    assert isinstance(frame["f"], float)
    assert isinstance(frame["grad"], list)
    assert len(frame["grad"]) == 2


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_tuple_fn_uses_grad_directly_no_warning(mock_server):
    """fn returns (f, grad) → grad used directly, no FD warning."""
    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [1.0, 2.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        msg = await ws.recv()
        frame = json.loads(msg)
        assert frame["grad"] == pytest.approx([2.0, 4.0])
        await ws.send(json.dumps({"type": "result", "x_best": [1.0, 2.0], "f_best": 5.0, "total_steps": 1, "converged": False, "escapes": 0}))

    url = mock_server(handler)

    def grad_fn(x):
        f = float(x[0] ** 2 + x[1] ** 2)
        grad = [2.0 * float(x[0]), 2.0 * float(x[1])]
        return f, grad

    session = make_session(grad_fn, url, gradient="auto")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning → test fails
        result = session.run()

    assert result.f_best == 5.0


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_gradient_true_with_scalar_fn_raises(mock_server):
    """gradient=True but fn returns scalar → StreamError on first step."""
    from eoil.exceptions import StreamError

    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [1.0, 2.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        # keep connection open so client can raise locally
        await ws.wait_closed()

    url = mock_server(handler)

    session = make_session(lambda x: 1.0, url, gradient=True)

    with pytest.raises(StreamError, match="GRADIENT_MISSING"):
        session.run()


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_on_step_callback(mock_server):
    """on_step fires each step with correct args."""
    calls = []

    async def handler(ws):
        for step in [1, 2]:
            await ws.send(json.dumps({"type": "x", "step": step, "x": [1.0, 2.0], "x_best": None, "f_best": 0.1 * step, "converged": False, "escapes": 0}))
            await ws.recv()
        await ws.send(json.dumps({"type": "result", "x_best": [1.0, 2.0], "f_best": 0.0, "total_steps": 2, "converged": True, "escapes": 0}))

    url = mock_server(handler)

    def on_step(step, x, f_best):
        calls.append((step, list(x), f_best))

    session = make_session(lambda x: (1.0, [0.0, 0.0]), url, gradient=True, on_step=on_step)
    session.run()

    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_result_frame_returns_stream_result(mock_server):
    """Server sends result → correct StreamResult returned."""
    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [0.1, 0.2], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        await ws.recv()
        await ws.send(json.dumps({
            "type": "result",
            "x_best": [0.1, 0.2],
            "f_best": 0.005,
            "total_steps": 42,
            "converged": True,
            "escapes": 3,
            "compute_units_actual": 10,
            "eoil_charged": "0.001",
        }))

    url = mock_server(handler)
    session = make_session(lambda x: (0.5, [0.0, 0.0]), url, gradient=True)
    result = session.run()

    assert result.x_best == pytest.approx([0.1, 0.2])
    assert result.f_best == pytest.approx(0.005)
    assert result.total_steps == 42
    assert result.steps == 42  # alias
    assert result.converged is True
    assert result.escapes == 3
    assert result.eoil_charged == "0.001"


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_session_expired_error(mock_server):
    """Server sends error SESSION_GONE → SessionExpiredError raised."""
    from eoil.exceptions import SessionExpiredError

    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [0.0, 0.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        await ws.recv()
        await ws.send(json.dumps({"type": "error", "code": "SESSION_GONE", "message": "Session expired"}))

    url = mock_server(handler)
    session = make_session(lambda x: (1.0, [0.0, 0.0]), url, gradient=True)

    with pytest.raises(SessionExpiredError):
        session.run()


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_server_close_1011_raises_inactivity(mock_server):
    """Server close with code 1011 → StreamError INACTIVITY_TIMEOUT."""
    import websockets.frames
    from eoil.exceptions import StreamError

    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [0.0, 0.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        await ws.recv()
        await ws.close(1011, "Internal error")

    url = mock_server(handler)
    session = make_session(lambda x: (1.0, [0.0, 0.0]), url, gradient=True)

    with pytest.raises(StreamError) as exc_info:
        session.run()
    assert exc_info.value.code == "INACTIVITY_TIMEOUT"


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_eval_timeout_raises_and_details(mock_server):
    """fn exceeds eval_timeout_s → EoilError raised."""
    import time
    from eoil.exceptions import EoilError

    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [0.0, 0.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        # Don't expect an eval back — client will raise locally
        await ws.wait_closed()

    url = mock_server(handler)

    def slow_fn(x):
        time.sleep(5)
        return 1.0

    session = make_session(slow_fn, url, gradient="auto", eval_timeout_s=0.1)

    with pytest.raises(EoilError, match="eval_timeout_s"):
        session.run()


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_numpy_x_when_available(mock_server):
    """fn receives np.ndarray when numpy is installed."""
    pytest.importorskip("numpy")
    import numpy as np

    received_types = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "x", "step": 1, "x": [1.0, 2.0], "x_best": None, "f_best": None, "converged": False, "escapes": 0}))
        await ws.recv()
        await ws.send(json.dumps({"type": "result", "x_best": [1.0, 2.0], "f_best": 1.0, "total_steps": 1, "converged": False, "escapes": 0}))

    url = mock_server(handler)

    def capturing_fn(x):
        received_types.append(type(x).__name__)
        return (1.0, [0.0, 0.0])

    session = make_session(capturing_fn, url, gradient="auto")
    session.run()

    assert received_types[0] == "ndarray"


@pytest.mark.skipif(not _WEBSOCKETS_REAL, reason="websockets not installed")
def test_stream_result_aliases():
    """StreamResult .steps, .x, .f aliases work correctly."""
    from eoil.models import StreamResult
    r = StreamResult(x_best=[1.0, 2.0], f_best=0.5, total_steps=10, converged=True, escapes=1)
    assert r.steps == 10
    assert r.x == [1.0, 2.0]
    assert r.f == 0.5


def test_stream_error_and_session_expired_in_exceptions():
    """StreamError and SessionExpiredError importable and have correct hierarchy."""
    from eoil.exceptions import EoilError, SessionExpiredError, StreamError
    e = StreamError("FOO", "bar")
    assert isinstance(e, EoilError)
    assert e.code == "FOO"
    assert "FOO" in str(e)

    se = SessionExpiredError()
    assert isinstance(se, StreamError)
    assert isinstance(se, EoilError)
