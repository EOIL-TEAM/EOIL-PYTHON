"""sklearn adapter for EOIL — EOILSearchCV replaces GridSearchCV / RandomizedSearchCV.

Usage::

    from eoil.adapters.sklearn import EOILSearchCV
    from sklearn.svm import SVC
    import eoil

    client = eoil.Client(api_key="eoil_sk_...")

    search = EOILSearchCV(
        SVC(),
        param_grid={
            "C":     (0.01, 100.0),        # continuous range
            "gamma": (1e-4, 1.0),          # continuous range
            "kernel": ["rbf", "linear"],   # categorical
        },
        client=client,
        scoring="accuracy",
        cv=5,
        budget_steps=200,
    )
    search.fit(X_train, y_train)

    print(search.best_params_)   # {"C": 12.4, "gamma": 0.003, "kernel": "rbf"}
    print(search.best_score_)    # 0.954
    print(search.best_estimator_.predict(X_test))

Parameter grid spec
-------------------
Each key maps the sklearn parameter name to one of:

- ``[v1, v2, ...]``           — categorical list (any type)
- ``(lo, hi)``                — continuous float range
- ``(lo, hi, "int")``         — integer range (values rounded)

EOIL encodes all parameters as continuous values in ``[0, 1]`` internally,
maps them to their declared ranges, then calls ``cross_val_score`` on each
candidate.  The cross-validation score is **maximised** (EOIL minimises
the negated score internally).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:
    import eoil as _eoil_pkg

__all__ = ["EOILSearchCV"]


# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------

@dataclass
class _ParamSpec:
    """Internal representation of one hyperparameter dimension."""
    name: str
    kind: str           # "continuous" | "integer" | "categorical"
    lo: float = 0.0     # for continuous / integer
    hi: float = 1.0     # for continuous / integer
    choices: list = field(default_factory=list)  # for categorical

    @property
    def n_choices(self) -> int:
        return len(self.choices)


def _build_param_specs(param_grid: dict[str, Any]) -> list[_ParamSpec]:
    """Parse a user-supplied param_grid into a list of _ParamSpec objects."""
    specs = []
    for name, spec in param_grid.items():
        if isinstance(spec, list):
            if len(spec) == 0:
                raise ValueError(f"param_grid['{name}']: list must not be empty.")
            specs.append(_ParamSpec(name=name, kind="categorical", choices=spec))

        elif isinstance(spec, tuple):
            if len(spec) == 2:
                lo, hi = spec
                specs.append(_ParamSpec(name=name, kind="continuous", lo=float(lo), hi=float(hi)))
            elif len(spec) == 3:
                lo, hi, kind = spec
                if kind not in ("int", int):
                    raise ValueError(
                        f"param_grid['{name}']: third tuple element must be 'int' or int, got {kind!r}."
                    )
                specs.append(_ParamSpec(name=name, kind="integer", lo=float(lo), hi=float(hi)))
            else:
                raise ValueError(
                    f"param_grid['{name}']: tuple must be (lo, hi) or (lo, hi, 'int'), "
                    f"got length {len(spec)}."
                )
        else:
            raise ValueError(
                f"param_grid['{name}']: unsupported type {type(spec).__name__!r}. "
                "Use a list for categorical values or a (lo, hi) tuple for ranges."
            )

        if specs[-1].kind in ("continuous", "integer") and specs[-1].lo >= specs[-1].hi:
            raise ValueError(
                f"param_grid['{name}']: lo ({specs[-1].lo}) must be less than hi ({specs[-1].hi})."
            )

    if not specs:
        raise ValueError("param_grid must not be empty.")
    return specs


def _decode_x(specs: list[_ParamSpec], x: list[float]) -> dict[str, Any]:
    """Map a normalised [0, 1]^n vector to a concrete param dict."""
    params: dict[str, Any] = {}
    for spec, xi in zip(specs, x):
        # Clamp to [0, 1] to guard against tiny floating-point overshoots
        xi = max(0.0, min(1.0, float(xi)))
        if spec.kind == "categorical":
            idx = round(xi * (spec.n_choices - 1))
            idx = max(0, min(spec.n_choices - 1, idx))
            params[spec.name] = spec.choices[idx]
        elif spec.kind == "integer":
            val = spec.lo + xi * (spec.hi - spec.lo)
            params[spec.name] = int(round(val))
        else:  # continuous
            params[spec.name] = spec.lo + xi * (spec.hi - spec.lo)
    return params


def _encode_x(specs: list[_ParamSpec], params: dict[str, Any]) -> list[float]:
    """Map a concrete param dict back to a normalised [0, 1]^n vector.

    Used to encode ``x0`` when the caller supplies a warm-start point.
    """
    x = []
    for spec in specs:
        val = params[spec.name]
        if spec.kind == "categorical":
            idx = spec.choices.index(val)
            x.append(idx / max(spec.n_choices - 1, 1))
        else:
            x.append((float(val) - spec.lo) / (spec.hi - spec.lo))
    return x


# ---------------------------------------------------------------------------
# cv_results_ builder
# ---------------------------------------------------------------------------

def _build_cv_results(evaluations: list[dict]) -> dict[str, Any]:
    """Convert raw evaluations list into a sklearn-style cv_results_ dict."""
    if not evaluations:
        return {}

    scores = [e["mean_test_score"] for e in evaluations]
    all_params = [e["params"] for e in evaluations]

    # rank: 1 = best (highest score)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    result: dict[str, Any] = {
        "params": all_params,
        "mean_test_score": scores,
        "rank_test_score": ranks,
    }

    if _HAS_NUMPY:
        result["mean_test_score"] = _np.array(scores)
        result["rank_test_score"] = _np.array(ranks, dtype=int)

    return result


# ---------------------------------------------------------------------------
# EOILSearchCV
# ---------------------------------------------------------------------------

class EOILSearchCV:
    """Hyperparameter search using the EOIL optimizer.

    A drop-in replacement for ``sklearn.model_selection.GridSearchCV`` and
    ``RandomizedSearchCV``.  Instead of exhaustively enumerating a grid or
    sampling randomly, EOIL navigates the hyperparameter space intelligently.

    Parameters
    ----------
    estimator:
        A sklearn-compatible estimator (implements ``fit`` and either
        ``predict`` or ``predict_proba``).
    param_grid:
        Dict mapping parameter names to search specs.  Three formats:

        - ``[v1, v2, ...]``         — categorical list
        - ``(lo, hi)``              — continuous float range
        - ``(lo, hi, "int")``       — integer range

    client:
        An authenticated :class:`eoil.Client` instance.
    scoring:
        A sklearn scoring string (e.g. ``"accuracy"``, ``"neg_mean_squared_error"``)
        or a callable ``scorer(estimator, X, y) -> float``.
        Defaults to the estimator's default score.
    cv:
        Number of cross-validation folds.  Default ``5``.
    budget_steps:
        Approximate number of hyperparameter evaluations.  Default ``200``.
    refit:
        If ``True`` (default), refit the best estimator on the full dataset
        after search completes and expose it as ``best_estimator_``.
    n_jobs:
        Passed to ``cross_val_score``.  Default ``1``.
    verify_ssl:
        Set ``False`` for staging environments with self-signed certificates.

    Attributes (set after ``fit``)
    ---------
    best_params_ : dict
        Hyperparameter dict that achieved the best cross-validation score.
    best_score_ : float
        Best mean cross-validation score.
    best_estimator_ :
        Estimator refitted with ``best_params_`` on the full dataset.
        Only set when ``refit=True``.
    cv_results_ : dict
        Dict with keys ``params``, ``mean_test_score``, ``rank_test_score``
        containing one entry per evaluated hyperparameter combination.
    n_splits_ : int
        The number of CV folds used.

    Examples
    --------
    ::

        from eoil.adapters.sklearn import EOILSearchCV
        from sklearn.ensemble import RandomForestClassifier
        import eoil

        client = eoil.Client(api_key="eoil_sk_...")

        search = EOILSearchCV(
            RandomForestClassifier(),
            param_grid={
                "n_estimators": (10, 200, "int"),
                "max_depth":    (2, 20, "int"),
                "min_samples_split": (2, 20, "int"),
            },
            client=client,
            scoring="accuracy",
            cv=3,
            budget_steps=100,
        )
        search.fit(X_train, y_train)
        print(search.best_params_, search.best_score_)
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: dict[str, Any],
        *,
        client: "eoil.Client",  # type: ignore[name-defined]
        scoring: Union[str, Callable, None] = None,
        cv: int = 5,
        budget_steps: int = 200,
        refit: bool = True,
        n_jobs: int = 1,
        verify_ssl: bool = True,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.client = client
        self.scoring = scoring
        self.cv = cv
        self.budget_steps = budget_steps
        self.refit = refit
        self.n_jobs = n_jobs
        self.verify_ssl = verify_ssl

        # Set after fit()
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = float("-inf")
        self.best_estimator_: Any = None
        self.cv_results_: dict[str, Any] = {}
        self.n_splits_: int = cv

    def fit(self, X: Any, y: Any = None) -> "EOILSearchCV":
        """Run the hyperparameter search.

        Parameters
        ----------
        X:
            Training features.
        y:
            Training labels / targets (may be ``None`` for unsupervised estimators).

        Returns
        -------
        self
        """
        try:
            from sklearn.base import clone as _clone
            from sklearn.model_selection import cross_val_score as _cv_score
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for EOILSearchCV. "
                "Install it with: pip install 'eoil[sklearn]'"
            ) from exc

        specs = _build_param_specs(self.param_grid)
        dimension = len(specs)
        evaluations: list[dict] = []

        def _objective(x: Any) -> float:
            raw = list(x) if not isinstance(x, list) else x
            params = _decode_x(specs, raw)
            est = _clone(self.estimator)
            est.set_params(**params)
            scores = _cv_score(
                est,
                X,
                y,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                error_score="raise",
            )
            mean_score = float(scores.mean())
            evaluations.append({"params": params, "mean_test_score": mean_score})
            return -mean_score  # EOIL minimises; we maximise score

        stream_result = self.client.stream_optimize(
            fn=_objective,
            dimension=dimension,
            budget_steps=self.budget_steps,
            bounds=(0.0, 1.0),
            gradient=False,   # CV scores have no analytic gradient
            verify_ssl=self.verify_ssl,
        )

        # Decode best solution
        self.best_params_ = _decode_x(specs, stream_result.x_best)
        self.best_score_ = -stream_result.f_best   # un-negate
        self.cv_results_ = _build_cv_results(evaluations)
        self.n_splits_ = self.cv

        if self.refit:
            best_est = _clone(self.estimator)
            best_est.set_params(**self.best_params_)
            best_est.fit(X, y)
            self.best_estimator_ = best_est

        return self

    # ------------------------------------------------------------------
    # Convenience passthrough for predict / predict_proba / score
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> Any:
        """Predict using ``best_estimator_``. Requires ``refit=True``."""
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities using ``best_estimator_``. Requires ``refit=True``."""
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(X)

    def score(self, X: Any, y: Any) -> float:
        """Score using ``best_estimator_``. Requires ``refit=True``."""
        self._check_is_fitted()
        return self.best_estimator_.score(X, y)

    def _check_is_fitted(self) -> None:
        if self.best_estimator_ is None:
            raise RuntimeError(
                "EOILSearchCV is not fitted yet. Call fit() first, "
                "and ensure refit=True."
            )
