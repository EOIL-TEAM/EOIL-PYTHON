# EOIL Python SDK

Python client for the [EOIL](https://eoil.ltd) optimisation API.

> **Version:** `0.5.0` — beta. API is stable.

## Installation

```bash
pip install eoil
```

Requires Python ≥ 3.10. NumPy is optional but recommended — the SDK passes `np.ndarray` to
your objective function when it's installed, otherwise `list[float]`.

## Authentication

Get an API key from [app.eoil.ltd](https://app.eoil.ltd) → **API Keys**.

```python
import eoil

client = eoil.Client(api_key="eoil_sk_...")

# Or set the environment variable EOIL_API_KEY and omit api_key=
```

---

## 1. Custom objective functions (`stream_optimize`)

Run **your own Python function** against the EOIL optimizer. Your code never leaves your
machine — only candidate coordinate vectors and scalar values cross the network.

```python
def rosenbrock(x):
    # x is np.ndarray (if numpy installed) or list[float]
    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    grad = [
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2),
    ]
    return f, grad   # return (f, grad) to supply gradient directly

result = client.stream_optimize(
    rosenbrock,
    dimension=2,
    budget_steps=500,
    bounds=(-5.0, 5.0),
)

print(result.x_best)       # [1.0, 1.0]        best solution found
print(result.f_best)       # ~0.0              best objective value
print(result.converged)    # True / False
print(result.steps)        # int               evaluations consumed
print(result.eoil_charged) # "5.0"             compute units charged
```

### Gradient options

| `gradient=` | Behaviour |
|---|---|
| `"auto"` (default) | Inspect return type on first call — tuple → use grad; scalar → finite differences |
| `True` | Function must return `(f, grad)`. Raises `StreamError` if it returns a scalar. |
| `False` | Always use finite differences, even if function returns a tuple. |

When finite differences are used, the SDK prints a one-time warning and costs `2 × dimension`
extra function calls per step. Supply an analytic gradient to avoid this.

### Live progress

```python
def on_step(step, x, f_best):
    if f_best is not None:
        print(f"step {step:4d} | f_best={f_best:.6f}")

result = client.stream_optimize(
    my_fn,
    dimension=10,
    budget_steps=1000,
    on_step=on_step,
)
```

### All parameters

| Parameter | Default | Description |
|---|---|---|
| `fn` | required | Objective function. Returns `float` or `(float, list[float])`. |
| `dimension` | required | Problem dimensionality (1–1000). |
| `budget_steps` | `1000` | Approximate number of function evaluations (100–100 000). |
| `bounds` | `(-5.0, 5.0)` | Box bounds `(lower, upper)` applied uniformly. |
| `x0` | `None` | Initial point. Server picks randomly if omitted. |
| `gradient` | `"auto"` | Gradient supply mode — see table above. |
| `fd_step` | `1e-5` | Finite-difference step size. |
| `on_step` | `None` | Callback `on_step(step, x, f_best)` called after each evaluation. |
| `timeout_s` | `3600.0` | Total session wall-clock timeout in seconds. |
| `eval_timeout_s` | `80.0` | Per-evaluation timeout. Raises if function exceeds this. |
| `verify_ssl` | `True` | Set `False` for staging or self-signed certs. |

---

## 2. Catalogue (`client.catalogue`)

Run built-in optimisation problems — no objective function needed.

### Benchmark functions

```python
result = client.catalogue.rastrigin(dimension=20, budget_steps=1000)
result = client.catalogue.sphere(dimension=10, budget_steps=500)
result = client.catalogue.rosenbrock(dimension=15, budget_steps=2000)
result = client.catalogue.ackley(dimension=10, budget_steps=1000)
result = client.catalogue.levy(dimension=10, budget_steps=1000)
result = client.catalogue.griewank(dimension=10, budget_steps=1000)
result = client.catalogue.quadratic(dimension=10, budget_steps=1000)
```

### Portfolio optimisation

```python
import numpy as np

# Maximise Sharpe ratio — pass a (T × N) returns matrix
returns = np.random.randn(252, 5) * 0.01  # 252 trading days, 5 assets
result = client.catalogue.portfolio_sharpe(
    returns=returns,
    dimension=5,
    budget_steps=2000,
)
weights = result.x_best  # portfolio weights, sum to 1

# Minimise portfolio variance — pass an (N × N) covariance matrix
cov = np.cov(returns.T)
result = client.catalogue.portfolio_minvol(
    cov=cov,
    dimension=5,
    budget_steps=2000,
)
```

Plain Python nested lists are accepted wherever NumPy arrays are shown.

### Available catalogue methods

| Method | Description |
|---|---|
| `catalogue.sphere()` | Convex, global min at origin |
| `catalogue.rastrigin()` | Highly multimodal |
| `catalogue.rosenbrock()` | Narrow curved valley |
| `catalogue.quadratic()` | Coupled quadratic (ill-conditioned) |
| `catalogue.ackley()` | Many local minima |
| `catalogue.levy()` | Global min at x=(1,…,1) |
| `catalogue.griewank()` | Regularly spaced local minima |
| `catalogue.portfolio_sharpe()` | Maximise Sharpe ratio |
| `catalogue.portfolio_minvol()` | Minimise portfolio variance |

---

## 3. Optimizer presets

Presets are `heuristics_override` dicts tuned for common scenarios.

```python
from eoil import PRESETS, get_preset

result = client.catalogue.rastrigin(
    dimension=50,
    budget_steps=5000,
    heuristics_override=PRESETS["thorough"],
)

# get_preset() returns a copy — safe to modify
opts = get_preset("portfolio")
opts["escape_strength"] = 0.8
result = client.catalogue.portfolio_sharpe(
    returns=returns,
    dimension=5,
    budget_steps=3000,
    heuristics_override=opts,
)
```

| Preset | Best for |
|---|---|
| `fast` | CI checks and quick feasibility runs |
| `balanced` | General-purpose (default) |
| `thorough` | Production-quality results |
| `portfolio` | Sharpe ratio and minimum-variance problems |
| `ml_hyperparam` | Machine learning hyperparameter search |
| `high_dimensional` | Problems with dimension ≥ 100 |

---

## 4. Async job submission

```python
# Submit and get a job ID immediately
job_id = client.submit_job(objective_type="ackley", dimension=20)

# Retrieve the result later (polling)
result = client.get_job(job_id)
```

---

## 5. Result object

Both `stream_optimize()` and catalogue methods return a result object with these fields:

| Field | Alias | Description |
|---|---|---|
| `x_best` | `.x` | Best solution vector found |
| `f_best` | `.f` | Best objective value |
| `converged` | — | Whether the optimizer declared convergence |
| `total_steps` | `.steps` | Evaluations consumed |
| `eoil_charged` | — | Compute units charged (`stream_optimize` only) |

---

## 6. Error handling

```python
from eoil import (
    AuthError,
    InsufficientBalanceError,
    RateLimitError,
    StreamError,
    SessionExpiredError,
    EoilError,
)

try:
    result = client.stream_optimize(my_fn, dimension=10, budget_steps=500)
except AuthError:
    print("Invalid or missing API key — check app.eoil.ltd → API Keys")
except InsufficientBalanceError:
    print("Insufficient compute balance")
except RateLimitError:
    print("Rate limit hit — retry shortly")
except SessionExpiredError:
    print("Session expired — try again")
except StreamError as e:
    print(f"Stream error [{e.code}]: {e}")
except EoilError as e:
    print(f"API error: {e}")
```

---

## What's New in `0.5.0`

- **`eoil.adapters.scipy`** — drop-in replacement for `scipy.optimize.minimize`
- **`eoil.adapters.sklearn`** — `EOILSearchCV` replaces `GridSearchCV` / `RandomizedSearchCV`
- **`eoil.adapters.pytorch`** — `EOILOptimizer` for `nn.Module` training loops
- **`eoil.adapters.pandas`** — `EOILPortfolioOptimizer` for portfolio optimisation on `pd.DataFrame`
- Promoted to **Beta** — all adapters tested against production API

---

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (no API key needed — HTTP and WebSocket are mocked)
pytest tests/ -v
```

## Links

- [Dashboard](https://app.eoil.ltd)
- [API reference](https://eoil.ltd/docs)
- [Status](https://eoil.ltd/status)
