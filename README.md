# EOIL Python SDK

Python client for the [EOIL](https://eoil.ltd) optimisation API.

> **Status**: Alpha — `0.2.0a2`. API may change before 1.0.

## Installation

```bash
pip install eoil
```

## Quickstart

```python
import eoil

client = eoil.Client(api_key="eoil_sk_...")

result = client.optimize(
    objective_type="rastrigin",
    dimension=10,
    budget_steps=2000,
)

print(result.x_best)       # best solution found
print(result.f_best)       # best objective value
print(result.converged)    # True / False
print(result.credits_used) # compute credits consumed

# Short aliases
print(result.x)  # same as x_best
print(result.f)  # same as f_best
```

## Catalogue

`client.catalogue` provides named methods for every built-in problem type.

### Benchmark problems

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

### Available problems

| Method | Description |
|--------|-------------|
| `catalogue.sphere()` | Convex, global min at origin |
| `catalogue.rastrigin()` | Highly multimodal |
| `catalogue.rosenbrock()` | Narrow curved valley |
| `catalogue.quadratic()` | Coupled quadratic |
| `catalogue.ackley()` | Many local minima |
| `catalogue.levy()` | Global min at x=(1,…,1) |
| `catalogue.griewank()` | Regularly spaced local minima |
| `catalogue.portfolio_sharpe()` | Maximise Sharpe ratio |
| `catalogue.portfolio_minvol()` | Minimise portfolio variance |

## Presets

Presets let you tune the search effort for your use-case without touching
low-level parameters.

```python
from eoil import PRESETS, get_preset

# Pass a preset as heuristics_override
result = client.catalogue.rastrigin(
    dimension=50,
    budget_steps=5000,
    heuristics_override=PRESETS["thorough"],
)

# get_preset() returns a copy — safe to modify before passing
opts = get_preset("portfolio")
result = client.catalogue.portfolio_sharpe(
    returns=returns,
    dimension=5,
    budget_steps=3000,
    heuristics_override=opts,
)
```

| Preset | Best for |
|--------|----------|
| `fast` | CI checks and quick feasibility runs |
| `balanced` | General-purpose use |
| `thorough` | Production-quality results |
| `portfolio` | Sharpe ratio and minimum-variance problems |
| `ml_hyperparam` | Machine learning hyper-parameter search |
| `high_dimensional` | Problems with dimension ≥ 100 |

## Async job submission

```python
# Submit and get a job ID immediately
job_id = client.submit_job(objective_type="ackley", dimension=20)

# Retrieve the result later
result = client.get_job(job_id)
```

## Error handling

```python
from eoil import AuthError, InsufficientBalanceError, RateLimitError, EoilError

try:
    result = client.catalogue.rastrigin(dimension=10, budget_steps=1000)
except AuthError:
    print("Check your API key")
except InsufficientBalanceError as e:
    print(f"Insufficient balance — need {e.required_eoil} EOIL")
except RateLimitError:
    print("Rate limit reached — please retry shortly")
except EoilError as e:
    print(f"API error {e.status_code}: {e}")
```

## Coming soon

The following features are not yet available in `0.2.x`:

- Streaming / real-time progress events
- Custom objective functions
- Async (`await`) client

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Installation

```bash
pip install eoil
```

## Quickstart

```python
import eoil

client = eoil.Client(api_key="eoil_sk_...")

result = client.optimize(
    objective_type="rastrigin",
    dimension=10,
    bounds=(-5.12, 5.12),
    budget_steps=2000,
)

print(result.x_best)    # [0.001, -0.002, ...]  full coordinate vector
print(result.f_best)    # 0.000123               best function value
print(result.converged) # True
print(result.credits_used)  # 42                compute units consumed

# Short aliases
print(result.x)   # same as x_best
print(result.f)   # same as f_best
```

## Catalogue — named landscape methods

`client.catalogue` exposes typed methods for every built-in objective so you
don't need to remember string names or payload shapes.

### Benchmark functions

```python
# All accept dimension= and budget_steps= as keyword arguments
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

Both portfolio methods accept plain Python nested lists in place of NumPy arrays.

### Supported objective types

| Method | `objective_type` string | Description |
|--------|------------------------|-------------|
| `catalogue.sphere()` | `sphere` | Convex, global min at origin |
| `catalogue.rastrigin()` | `rastrigin` | Highly multimodal |
| `catalogue.rosenbrock()` | `rosenbrock` | Narrow curved valley |
| `catalogue.quadratic()` | `quadratic` | Coupled quadratic (ill-conditioned) |
| `catalogue.ackley()` | `ackley` | Multimodal, many local minima |
| `catalogue.levy()` | `levy` | Multimodal, global min at x=(1,…,1) |
| `catalogue.griewank()` | `griewank` | Regularly distributed local minima |
| `catalogue.portfolio_sharpe()` | `portfolio_sharpe` | Maximise Sharpe ratio |
| `catalogue.portfolio_minvol()` | `portfolio_minvol` | Minimise portfolio variance |

## Optimizer presets

Presets are `heuristics_override` dicts tuned for common scenarios. Pass them
to any `catalogue` method or to `client.optimize()` directly.

```python
from eoil import PRESETS, get_preset

# Use a preset directly
result = client.catalogue.rastrigin(
    dimension=50,
    budget_steps=5000,
    heuristics_override=PRESETS["thorough"],
)

# get_preset() returns a copy — safe to modify
opts = get_preset("portfolio")
result = client.catalogue.portfolio_sharpe(
    returns=returns,
    dimension=5,
    budget_steps=3000,
    heuristics_override=opts,
)
```

| Preset | Best for |
|--------|----------|
| `fast` | CI checks and quick feasibility runs |
| `balanced` | General-purpose use |
| `thorough` | Production-quality results |
| `portfolio` | Sharpe ratio and minimum-variance problems |
| `ml_hyperparam` | Machine learning hyper-parameter search |
| `high_dimensional` | Problems with dimension ≥ 100 |

## Async job submission

```python
# Queue for async execution
job_id = client.submit_job(objective_type="ackley", dimension=20)

# Poll later
result = client.get_job(job_id)
```

## Error handling

```python
from eoil import AuthError, InsufficientBalanceError, RateLimitError, EoilError

try:
    result = client.catalogue.rastrigin(dimension=10, budget_steps=1000)
except AuthError:
    print("Check your API key and scopes")
except InsufficientBalanceError as e:
    print(f"Need {e.required_eoil} EOIL, have {e.balance_eoil}")
except RateLimitError:
    print("Slow down — rate limit hit")
except EoilError as e:
    print(f"API error {e.status_code}: {e}")
```

## Not yet available (Phase 2 / 3)

The following features are on the roadmap but are **not available** in `0.2.x`:

- **Streaming results** — real-time iteration callbacks / SSE progress events
- **Custom objective functions** — uploading user-defined Python callables
- **Async `await` client** — `AsyncClient` planned for Phase 3

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (no API key needed — all HTTP is mocked)
pytest tests/ -v
```
