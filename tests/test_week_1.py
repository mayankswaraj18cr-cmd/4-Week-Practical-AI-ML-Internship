import numpy as np
import pandas as pd
import pytest

from week_1_foundations.data_cleaning_eda import cap_iqr_outliers, impute_missing_values
from week_1_foundations.math_for_ml import eigenvalues, mse_gradient
from week_1_foundations.python_fundamentals import RecordManager, StudentRecord
from week_1_foundations.scikit_learn_models import build_random_forest_pipeline, build_ridge_pipeline


def test_record_manager_validates_and_averages() -> None:
    manager = RecordManager()
    manager.add(StudentRecord("s1", "Ada", 80))
    manager.add(StudentRecord("s2", "Lin", 100))
    assert manager.average_score() == 90
    with pytest.raises(KeyError):
        manager.add(StudentRecord("s1", "Duplicate", 90))


def test_cleaning_imputes_and_caps_outlier() -> None:
    frame = pd.DataFrame({"value": [1.0, np.nan, 3.0, 100.0], "label": ["a", None, "b", "c"]})
    cleaned = cap_iqr_outliers(impute_missing_values(frame))
    assert not cleaned.isna().any().any()
    assert cleaned["value"].max() < 100


def test_math_helpers() -> None:
    assert np.allclose(np.sort(eigenvalues(np.diag([2.0, 3.0]))), [2, 3])
    gradient, bias_gradient = mse_gradient(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]), np.array([1.0]))
    assert np.allclose(gradient, [-5.0])
    assert bias_gradient == -3.0


def test_sklearn_pipelines_fit() -> None:
    ridge = build_ridge_pipeline()
    ridge.fit([[1], [2], [3]], [2, 4, 6])
    assert ridge.predict([[4]]).shape == (1,)
    forest = build_random_forest_pipeline(n_estimators=5)
    forest.fit([[1, "a"], [2, "b"], [3, "a"]], [0, 1, 0])
    assert forest.predict([[2, "b"]])[0] == 1
