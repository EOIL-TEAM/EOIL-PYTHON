# EOIL Python SDK

Python client for the [EOIL](https://eoil.ltd) compute optimisation API.

> **Status**: Alpha — under active development. API may change before 1.0.

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

print(result.x_best)   # [0.001, -0.002, ...]
print(result.f_best)   # 0.000123
print(result.converged)  # True
```

## Supported objective types

| Type | Description |
|------|-------------|
| `sphere` | Simple convex, global min at origin |
| `rastrigin` | Highly multimodal |
| `rosenbrock` | Narrow curved valley |
| `quadratic` | Coupled quadratic (ill-conditioned) |
| `ackley` | Multimodal, many local minima |
| `levy` | Multimodal, global min at x=(1,...,1) |
| `griewank` | Regularly distributed local minima |

## Async job submission

```python
# Queue for async execution
job_id = client.submit_job(objective_type="ackley", dimension=20)

# Poll later
result = client.get_job(job_id)
```

## Error handling

```python
import eoil
from eoil import AuthError, InsufficientBalanceError, RateLimitError, EoilError

try:
    result = client.optimize(objective_type="sphere", dimension=5)
except AuthError:
    print("Check your API key and scopes")
except InsufficientBalanceError as e:
    print(f"Need {e.required_eoil} EOIL, have {e.balance_eoil}")
except RateLimitError:
    print("Slow down — rate limit hit")
except EoilError as e:
    print(f"API error {e.status_code}: {e}")
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (no API key needed — all HTTP is mocked)
pytest tests/ -v
```
