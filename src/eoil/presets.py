"""Named optimizer presets for common use-cases.

Each preset is a dict suitable for passing as ``heuristics_override`` to
:meth:`~eoil.Client.optimize` or any :class:`~eoil.CatalogueClient` method.

Example::

    from eoil import Client, PRESETS

    client = Client(api_key="eoil_sk_...")
    result = client.catalogue.rastrigin(
        dimension=20,
        budget_steps=1000,
        heuristics_override=PRESETS["fast"],
    )
"""

from __future__ import annotations

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Individual preset definitions
# ---------------------------------------------------------------------------

#: Minimal budget — useful for CI smoke tests or quick feasibility checks.
FAST: Dict[str, Any] = {
    "restarts": 1,
    "patience": 50,
    "depth": 1,
}

#: Default balanced trade-off between speed and solution quality.
BALANCED: Dict[str, Any] = {
    "restarts": 3,
    "patience": 100,
    "depth": 2,
}

#: High-quality search — more thorough, longer runtime.
THOROUGH: Dict[str, Any] = {
    "restarts": 8,
    "patience": 200,
    "depth": 4,
}

#: Tuned for portfolio problems (Sharpe ratio, minimum variance).
PORTFOLIO: Dict[str, Any] = {
    "restarts": 2,
    "patience": 300,
    "depth": 2,
}

#: Tuned for ML hyper-parameter search.
ML_HYPERPARAM: Dict[str, Any] = {
    "restarts": 5,
    "patience": 150,
    "depth": 3,
}

#: Tuned for high-dimensional problems (dim ≥ 100).
HIGH_DIMENSIONAL: Dict[str, Any] = {
    "restarts": 1,
    "patience": 500,
    "depth": 6,
}


# ---------------------------------------------------------------------------
# Registry and lookup helper
# ---------------------------------------------------------------------------

#: Mapping of preset name → heuristics dict.  Import and pass directly as
#: ``heuristics_override``.
PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": FAST,
    "balanced": BALANCED,
    "thorough": THOROUGH,
    "portfolio": PORTFOLIO,
    "ml_hyperparam": ML_HYPERPARAM,
    "high_dimensional": HIGH_DIMENSIONAL,
}


def get_preset(name: str) -> Dict[str, Any]:
    """Return a copy of the named preset dict.

    Args:
        name: One of ``"fast"``, ``"balanced"``, ``"thorough"``,
            ``"portfolio"``, ``"ml_hyperparam"``, ``"high_dimensional"``.

    Returns:
        A shallow copy of the preset dict — safe to mutate without affecting
        the module-level constant.

    Raises:
        KeyError: If *name* is not a known preset.
    """
    try:
        return dict(PRESETS[name])
    except KeyError:
        valid = ", ".join(f'"{k}"' for k in PRESETS)
        raise KeyError(f"Unknown preset {name!r}. Valid presets: {valid}") from None
