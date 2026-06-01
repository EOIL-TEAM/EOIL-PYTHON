"""Tests for the EOIL Python SDK.

All HTTP calls are mocked — no real API requests are made.
"""

import pytest
import httpx
from pytest_httpx import HTTPXMock

import eoil
from eoil import Client, OptimizationResult, OptimizeResult, JobStatus
from eoil.models import AuthError, InsufficientBalanceError, RateLimitError, EoilError


BASE = "https://api.eoil.ltd"


# ---------------------------------------------------------------------------
# Client instantiation
# ---------------------------------------------------------------------------

def test_client_rejects_empty_api_key():
    with pytest.raises(AuthError):
        Client(api_key="")


def test_client_accepts_valid_api_key():
    c = Client(api_key="eoil_sk_test123")
    c.close()


def test_client_context_manager():
    with Client(api_key="eoil_sk_test") as c:
        assert c is not None


# ---------------------------------------------------------------------------
# optimize() — success path
# ---------------------------------------------------------------------------

MOCK_JOB_RESPONSE = {
    "ok": True,
    "jobId": "job_abc123",
    "requestId": "req_xyz",
    "status": "succeeded",
    "charged": {
        "fiatCurrency": "GBP",
        "fiatCost": "0.001",
        "eoilCharged": "1.0",
        "dimensionMultiplier": 1,
    },
    "result": {
        "x_best": [0.001, -0.002],
        "f_best": 0.000005,
        "total_steps": 847,
        "converged": True,
        "escapes": 2,
        "time_ms": 312.4,
        "compute_units_actual": 847,
    },
}


def test_optimize_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/jobs",
        json=MOCK_JOB_RESPONSE,
        status_code=200,
    )
    with Client(api_key="eoil_sk_test") as c:
        result = c.optimize(objective_type="sphere", dimension=2, budget_steps=1000)

    assert isinstance(result, OptimizationResult)
    assert result.job_id == "job_abc123"
    assert result.status == "succeeded"
    assert result.f_best == pytest.approx(0.000005)
    assert result.converged is True
    assert result.compute_units_actual == 847


def test_optimize_all_objective_types(httpx_mock: HTTPXMock):
    """All 7 objective types are accepted without raising."""
    for obj_type in ["sphere", "rastrigin", "rosenbrock", "quadratic", "ackley", "levy", "griewank"]:
        httpx_mock.add_response(method="POST", url=f"{BASE}/jobs", json=MOCK_JOB_RESPONSE, status_code=200)
        with Client(api_key="eoil_sk_test") as c:
            result = c.optimize(objective_type=obj_type, dimension=5)
        assert result.job_id == "job_abc123"


def test_optimize_with_bounds_and_x0(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/jobs", json=MOCK_JOB_RESPONSE)
    with Client(api_key="eoil_sk_test") as c:
        result = c.optimize(
            objective_type="rastrigin",
            dimension=2,
            bounds=(-5.12, 5.12),
            x0=[1.0, 1.0],
            budget_steps=500,
            compute_units=50,
        )
    assert result.status == "succeeded"


def test_optimize_sends_authorization_header(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/jobs", json=MOCK_JOB_RESPONSE)
    with Client(api_key="eoil_sk_secret") as c:
        c.optimize()
    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer eoil_sk_secret"


def test_optimize_sends_idempotency_key(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/jobs", json=MOCK_JOB_RESPONSE)
    with Client(api_key="eoil_sk_test") as c:
        c.optimize(idempotency_key="my-key-123")
    request = httpx_mock.get_requests()[0]
    assert request.headers["Idempotency-Key"] == "my-key-123"


def test_optimize_generates_idempotency_key_if_not_provided(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/jobs", json=MOCK_JOB_RESPONSE)
    with Client(api_key="eoil_sk_test") as c:
        c.optimize()
    request = httpx_mock.get_requests()[0]
    # Auto-generated UUID — just check it's present
    assert "Idempotency-Key" in request.headers


# ---------------------------------------------------------------------------
# optimize() — validation errors (no HTTP needed)
# ---------------------------------------------------------------------------

def test_optimize_rejects_unknown_objective_type():
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(ValueError, match="Unknown objective_type"):
            c.optimize(objective_type="banana")


def test_optimize_rejects_dimension_out_of_range():
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(ValueError, match="dimension"):
            c.optimize(dimension=0)
        with pytest.raises(ValueError, match="dimension"):
            c.optimize(dimension=1001)


def test_optimize_rejects_budget_steps_out_of_range():
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(ValueError, match="budget_steps"):
            c.optimize(budget_steps=50)
        with pytest.raises(ValueError, match="budget_steps"):
            c.optimize(budget_steps=200_000)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_optimize_raises_auth_error_on_401(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/jobs",
        json={"error": "Invalid API key"}, status_code=401,
    )
    with Client(api_key="eoil_sk_bad") as c:
        with pytest.raises(AuthError):
            c.optimize()


def test_optimize_raises_auth_error_on_403(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/jobs",
        json={"error": "Missing scope: optimizer:stream"}, status_code=403,
    )
    with Client(api_key="eoil_sk_limited") as c:
        with pytest.raises(AuthError, match="scopes"):
            c.optimize()


def test_optimize_raises_insufficient_balance_on_402(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/jobs",
        json={
            "ok": False,
            "error": "Insufficient EOIL credit balance",
            "balanceEoil": "0.5",
            "requiredEoil": "1.0",
        },
        status_code=402,
    )
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(InsufficientBalanceError) as exc_info:
            c.optimize()
    assert exc_info.value.balance_eoil == "0.5"
    assert exc_info.value.required_eoil == "1.0"


def test_optimize_raises_rate_limit_on_429(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/jobs",
        json={"error": "Too many requests"}, status_code=429,
    )
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(RateLimitError):
            c.optimize()


def test_optimize_raises_eoil_error_on_500(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/jobs",
        json={"error": "Internal server error"}, status_code=500,
    )
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(EoilError) as exc_info:
            c.optimize()
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_job()
# ---------------------------------------------------------------------------

MOCK_GET_JOB_RESPONSE = {
    "ok": True,
    "jobId": "job_abc123",
    "requestId": "req_xyz",
    "status": "succeeded",
    "charged": {"fiatCurrency": "GBP", "fiatCost": "0.001", "eoilCharged": "1.0"},
    "result": {
        "x_best": [0.0, 0.0],
        "f_best": 0.0,
        "total_steps": 100,
        "converged": True,
        "escapes": 0,
        "time_ms": 50.0,
        "compute_units_actual": 100,
    },
}


def test_get_job_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="GET", url=f"{BASE}/jobs/job_abc123",
        json=MOCK_GET_JOB_RESPONSE, status_code=200,
    )
    with Client(api_key="eoil_sk_test") as c:
        result = c.get_job("job_abc123")
    assert result.job_id == "job_abc123"
    assert result.f_best == 0.0


def test_get_job_not_found(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="GET", url=f"{BASE}/jobs/job_missing",
        json={"ok": False, "error": "Job not found"}, status_code=404,
    )
    with Client(api_key="eoil_sk_test") as c:
        with pytest.raises(EoilError):
            c.get_job("job_missing")


# ---------------------------------------------------------------------------
# submit_job() — queue mode
# ---------------------------------------------------------------------------

def test_submit_job_returns_job_id(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/jobs",
        json={"ok": True, "jobId": "job_queued_99", "requestId": "req_q", "status": "queued"},
        status_code=202,
    )
    with Client(api_key="eoil_sk_test") as c:
        job_id = c.submit_job(objective_type="ackley", dimension=5)
    assert job_id == "job_queued_99"


def test_optimize_result_backwards_compat_alias():
    """OptimizeResult is an alias for OptimizationResult."""
    assert OptimizeResult is OptimizationResult
