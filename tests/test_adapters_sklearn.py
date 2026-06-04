"""Tests for eoil.adapters.sklearn — EOILSearchCV."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from eoil.adapters.sklearn import (
    EOILSearchCV,
    _ParamSpec,
    _build_cv_results,
    _build_param_specs,
    _decode_x,
    _encode_x,
)
from eoil.models import StreamResult

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream_result(
    x_best=None,
    f_best=0.05,   # negated score, so best_score_ will be 0.95
    converged=False,
    total_steps=50,
    eoil_charged="1.0",
) -> StreamResult:
    return StreamResult(
        x_best=x_best or [0.5],
        f_best=f_best,
        converged=converged,
        total_steps=total_steps,
        escapes=0,
        compute_units_actual=5,
        eoil_charged=eoil_charged,
    )


def _make_client(stream_result: StreamResult) -> MagicMock:
    client = MagicMock()
    client.stream_optimize.return_value = stream_result
    return client


# ---------------------------------------------------------------------------
# _build_param_specs
# ---------------------------------------------------------------------------

class TestBuildParamSpecs:
    def test_categorical_list(self):
        specs = _build_param_specs({"kernel": ["rbf", "linear", "poly"]})
        assert len(specs) == 1
        s = specs[0]
        assert s.name == "kernel"
        assert s.kind == "categorical"
        assert s.choices == ["rbf", "linear", "poly"]

    def test_continuous_tuple(self):
        specs = _build_param_specs({"C": (0.01, 100.0)})
        s = specs[0]
        assert s.name == "C"
        assert s.kind == "continuous"
        assert s.lo == pytest.approx(0.01)
        assert s.hi == pytest.approx(100.0)

    def test_integer_tuple(self):
        specs = _build_param_specs({"n_estimators": (10, 200, "int")})
        s = specs[0]
        assert s.name == "n_estimators"
        assert s.kind == "integer"
        assert s.lo == 10.0
        assert s.hi == 200.0

    def test_integer_tuple_int_type(self):
        specs = _build_param_specs({"depth": (1, 10, int)})
        assert specs[0].kind == "integer"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _build_param_specs({"x": []})

    def test_empty_param_grid_raises(self):
        with pytest.raises(ValueError, match="param_grid must not be empty"):
            _build_param_specs({})

    def test_lo_ge_hi_raises(self):
        with pytest.raises(ValueError, match="lo .* must be less than hi"):
            _build_param_specs({"C": (5.0, 1.0)})

    def test_invalid_tuple_third_elem_raises(self):
        with pytest.raises(ValueError, match="third tuple element must be"):
            _build_param_specs({"x": (0.0, 1.0, "float")})

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="unsupported type"):
            _build_param_specs({"x": 42})

    def test_multiple_params(self):
        specs = _build_param_specs({
            "C": (0.1, 10.0),
            "kernel": ["rbf", "linear"],
            "degree": (2, 5, "int"),
        })
        assert len(specs) == 3
        names = [s.name for s in specs]
        assert "C" in names
        assert "kernel" in names
        assert "degree" in names


# ---------------------------------------------------------------------------
# _decode_x / _encode_x
# ---------------------------------------------------------------------------

class TestDecodeEncodeX:
    def test_decode_continuous(self):
        specs = _build_param_specs({"C": (1.0, 100.0)})
        params = _decode_x(specs, [0.0])
        assert params["C"] == pytest.approx(1.0)
        params = _decode_x(specs, [1.0])
        assert params["C"] == pytest.approx(100.0)
        params = _decode_x(specs, [0.5])
        assert params["C"] == pytest.approx(50.5)

    def test_decode_integer(self):
        specs = _build_param_specs({"n": (1, 10, "int")})
        params = _decode_x(specs, [0.0])
        assert params["n"] == 1
        assert isinstance(params["n"], int)
        params = _decode_x(specs, [1.0])
        assert params["n"] == 10
        params = _decode_x(specs, [0.5])
        assert params["n"] == 6  # round(1 + 0.5*9) = round(5.5) = 6

    def test_decode_categorical(self):
        specs = _build_param_specs({"kernel": ["rbf", "linear", "poly"]})
        assert _decode_x(specs, [0.0])["kernel"] == "rbf"
        assert _decode_x(specs, [0.5])["kernel"] == "linear"
        assert _decode_x(specs, [1.0])["kernel"] == "poly"

    def test_decode_clamps_overshoot(self):
        specs = _build_param_specs({"C": (1.0, 10.0)})
        params = _decode_x(specs, [1.0001])
        assert params["C"] == pytest.approx(10.0)
        params = _decode_x(specs, [-0.0001])
        assert params["C"] == pytest.approx(1.0)

    def test_decode_categorical_clamps(self):
        specs = _build_param_specs({"k": ["a", "b"]})
        assert _decode_x(specs, [1.5])["k"] == "b"
        assert _decode_x(specs, [-0.5])["k"] == "a"

    def test_encode_continuous(self):
        specs = _build_param_specs({"C": (1.0, 100.0)})
        x = _encode_x(specs, {"C": 50.5})
        assert x[0] == pytest.approx(0.5)

    def test_encode_categorical(self):
        specs = _build_param_specs({"kernel": ["rbf", "linear", "poly"]})
        assert _encode_x(specs, {"kernel": "rbf"})[0] == pytest.approx(0.0)
        assert _encode_x(specs, {"kernel": "poly"})[0] == pytest.approx(1.0)

    def test_encode_decode_roundtrip(self):
        specs = _build_param_specs({
            "C": (0.1, 100.0),
            "kernel": ["rbf", "linear"],
            "degree": (2, 10, "int"),
        })
        original = {"C": 42.0, "kernel": "linear", "degree": 5}
        x = _encode_x(specs, original)
        decoded = _decode_x(specs, x)
        assert decoded["C"] == pytest.approx(42.0, rel=1e-3)
        assert decoded["kernel"] == "linear"
        assert decoded["degree"] == 5


# ---------------------------------------------------------------------------
# _build_cv_results
# ---------------------------------------------------------------------------

class TestBuildCvResults:
    def test_empty(self):
        assert _build_cv_results([]) == {}

    def test_rank_best_is_one(self):
        evals = [
            {"params": {"C": 1.0}, "mean_test_score": 0.8},
            {"params": {"C": 10.0}, "mean_test_score": 0.95},
            {"params": {"C": 0.1}, "mean_test_score": 0.7},
        ]
        r = _build_cv_results(evals)
        scores = list(r["mean_test_score"])
        ranks = list(r["rank_test_score"])
        best_idx = scores.index(0.95)
        assert ranks[best_idx] == 1

    def test_keys_present(self):
        evals = [{"params": {"C": 1.0}, "mean_test_score": 0.9}]
        r = _build_cv_results(evals)
        assert "params" in r
        assert "mean_test_score" in r
        assert "rank_test_score" in r


# ---------------------------------------------------------------------------
# EOILSearchCV — unit tests (mocked client + mocked cross_val_score)
# ---------------------------------------------------------------------------

class TestEOILSearchCV:
    """All CV calls are mocked — no real sklearn fitting happens."""

    def _mock_cv_score(self, mean: float):
        """Return a mock that makes cross_val_score return a fixed mean."""
        import numpy as np
        mock = MagicMock()
        mock.mean.return_value = mean
        mock.__iter__ = lambda s: iter([mean])
        # Make it behave like an ndarray with .mean()
        arr = MagicMock()
        arr.mean.return_value = mean
        return arr

    def test_fit_sets_best_params(self):
        specs = _build_param_specs({"C": (0.1, 10.0)})
        # x_best=[0.5] → C = 0.1 + 0.5*9.9 = 5.05
        sr = _make_stream_result(x_best=[0.5], f_best=-0.95)
        client = _make_client(sr)

        search = EOILSearchCV(
            MagicMock(),
            param_grid={"C": (0.1, 10.0)},
            client=client,
            cv=3,
            budget_steps=50,
            refit=False,
        )

        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.95, 0.95, 0.95])
            search.fit([[1]], [0])

        assert "C" in search.best_params_
        assert search.best_score_ == pytest.approx(0.95)

    def test_best_score_is_un_negated(self):
        sr = _make_stream_result(x_best=[0.0], f_best=-0.88)
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.88])
            search.fit([[1]], [0])
        assert search.best_score_ == pytest.approx(0.88)

    def test_gradient_is_false(self):
        """CV scores have no gradient — stream_optimize must be called with gradient=False."""
        sr = _make_stream_result()
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        kwargs = client.stream_optimize.call_args.kwargs
        assert kwargs["gradient"] is False

    def test_bounds_are_zero_one(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        kwargs = client.stream_optimize.call_args.kwargs
        assert kwargs["bounds"] == (0.0, 1.0)

    def test_dimension_matches_param_count(self):
        sr = _make_stream_result(x_best=[0.5, 0.5, 0.5])
        client = _make_client(sr)
        search = EOILSearchCV(
            MagicMock(),
            {"C": (0.1, 10.0), "gamma": (1e-4, 1.0), "kernel": ["rbf", "linear"]},
            client=client,
            refit=False,
        )
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        kwargs = client.stream_optimize.call_args.kwargs
        assert kwargs["dimension"] == 3

    def test_budget_steps_forwarded(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, budget_steps=99, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        assert client.stream_optimize.call_args.kwargs["budget_steps"] == 99

    def test_verify_ssl_forwarded(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, verify_ssl=False, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        assert client.stream_optimize.call_args.kwargs["verify_ssl"] is False

    def test_cv_results_populated(self):
        """cv_results_ should contain one entry per stream_optimize evaluation."""
        sr = _make_stream_result(x_best=[0.5])
        client = _make_client(sr)

        call_count = 0

        def fake_stream_optimize(fn, **kwargs):
            nonlocal call_count
            # Simulate 3 evaluations
            for x in [[0.2], [0.5], [0.8]]:
                fn(x)
                call_count += 1
            return sr

        client.stream_optimize.side_effect = fake_stream_optimize
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])

        assert len(search.cv_results_["params"]) == 3
        assert "mean_test_score" in search.cv_results_
        assert "rank_test_score" in search.cv_results_

    def test_refit_true_sets_best_estimator(self):
        sr = _make_stream_result(x_best=[0.5])
        client = _make_client(sr)
        mock_est = MagicMock()
        search = EOILSearchCV(mock_est, {"C": (1.0, 10.0)}, client=client, refit=True)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1, 2], [3, 4]], [0, 1])
        assert search.best_estimator_ is not None

    def test_refit_false_no_best_estimator(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        assert search.best_estimator_ is None

    def test_predict_raises_if_not_fitted(self):
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=MagicMock(), refit=True)
        with pytest.raises(RuntimeError, match="not fitted"):
            search.predict([[1, 2]])

    def test_n_splits_set(self):
        sr = _make_stream_result()
        client = _make_client(sr)
        search = EOILSearchCV(MagicMock(), {"C": (1.0, 10.0)}, client=client, cv=7, refit=False)
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        assert search.n_splits_ == 7

    def test_categorical_params_decoded(self):
        """Categorical param should appear in best_params_ as original value, not float."""
        # x_best=[0.0] → kernel index 0 = "rbf"
        sr = _make_stream_result(x_best=[0.0])
        client = _make_client(sr)
        search = EOILSearchCV(
            MagicMock(),
            {"kernel": ["rbf", "linear", "poly"]},
            client=client,
            refit=False,
        )
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        assert search.best_params_["kernel"] == "rbf"

    def test_integer_param_is_int_type(self):
        sr = _make_stream_result(x_best=[0.5])
        client = _make_client(sr)
        search = EOILSearchCV(
            MagicMock(),
            {"n_estimators": (10, 200, "int")},
            client=client,
            refit=False,
        )
        with patch("sklearn.model_selection.cross_val_score") as mock_cvs, \
             patch("sklearn.base.clone", side_effect=lambda e: e):
            import numpy as np
            mock_cvs.return_value = np.array([0.9])
            search.fit([[1]], [0])
        assert isinstance(search.best_params_["n_estimators"], int)


# ---------------------------------------------------------------------------
# Integration test — real sklearn, mocked client
# ---------------------------------------------------------------------------

class TestEOILSearchCVIntegration:
    """Uses a real DummyClassifier and a tiny dataset. No API calls."""

    def test_real_cv_with_dummy_classifier(self):
        from sklearn.dummy import DummyClassifier
        import numpy as np

        X = np.array([[i] for i in range(20)])
        y = np.array([i % 2 for i in range(20)])

        sr = _make_stream_result(x_best=[0.0], f_best=-0.5)
        client = _make_client(sr)

        search = EOILSearchCV(
            DummyClassifier(),
            param_grid={"strategy": ["most_frequent", "uniform", "stratified"]},
            client=client,
            scoring="accuracy",
            cv=2,
            budget_steps=50,
            refit=True,
        )

        def fake_stream_optimize(fn, **kwargs):
            # Simulate evaluating each categorical choice
            for x in [[0.0], [0.5], [1.0]]:
                fn(x)
            return sr

        client.stream_optimize.side_effect = fake_stream_optimize
        search.fit(X, y)

        assert search.best_params_["strategy"] in ["most_frequent", "uniform", "stratified"]
        assert isinstance(search.best_score_, float)
        assert search.best_estimator_ is not None
        assert len(search.cv_results_["params"]) == 3
        # predict should work
        preds = search.predict(X)
        assert len(preds) == 20
