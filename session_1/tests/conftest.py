from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils import get_feature_target, load_config, load_data


@pytest.fixture(scope="session")
def real_config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def real_features_target(real_config):
    df = load_data(real_config)
    return get_feature_target(real_config, df)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small, deterministic dataset with the same shape of schema as the real
    one (numeric + categorical features, multi-class target) for fast,
    data-file-independent unit tests."""
    rng = np.random.RandomState(0)
    n = 40
    study_time = rng.uniform(1, 10, n)
    attendance = rng.uniform(50, 100, n)
    gender = rng.choice(["Male", "Female"], n)
    internet = rng.choice(["Yes", "No"], n)
    # deterministic, separable target so classifier tests are unambiguous
    grade = np.where(study_time + attendance / 10 > 12, "A", "C")
    return pd.DataFrame(
        {
            "study_time_hours": study_time,
            "attendance_percent": attendance,
            "gender": gender,
            "internet_access": internet,
            "final_grade": grade,
        }
    )


@pytest.fixture
def synthetic_X_y(synthetic_df):
    X = synthetic_df.drop(columns=["final_grade"])
    y = synthetic_df["final_grade"]
    return X, y
