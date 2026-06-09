import pandas as pd
import pytest

from train_model import train_model


@pytest.fixture
def big_sample_df():
    return pd.DataFrame({
        "gender": [
            "Female", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male",
            "Female", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male"
        ],
        "age": [
            25, 30, 35, 40, 45,
            50, 55, 60, 65, 70,
            28, 33, 38, 43, 48,
            53, 58, 63, 68, 73
        ],
        "hypertension": [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        ],
        "heart_disease": [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        ],
        "smoking_history": [
            "never", "never", "former", "former", "current",
            "current", "never", "former", "current", "never",
            "never", "former", "current", "never", "former",
            "current", "never", "former", "current", "never"
        ],
        "bmi": [
            22.1, 24.5, 25.0, 26.2, 27.8,
            30.1, 31.4, 32.0, 33.5, 35.0,
            23.1, 24.8, 26.0, 27.0, 28.5,
            31.0, 32.5, 34.0, 35.5, 37.0
        ],
        "HbA1c_level": [
            4.8, 5.0, 5.1, 5.2, 5.3,
            6.5, 6.6, 6.7, 6.8, 7.0,
            4.9, 5.0, 5.2, 5.3, 5.4,
            6.4, 6.5, 6.7, 6.8, 7.1
        ],
        "blood_glucose_level": [
            80, 85, 90, 95, 100,
            180, 190, 200, 210, 220,
            82, 87, 92, 97, 102,
            185, 195, 205, 215, 225
        ],
        "diabetes": [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        ]
    })


def test_train_model(big_sample_df):
    result = train_model(
        big_sample_df,
        model_name="decision_tree"
    )

    assert isinstance(result, dict)

    expected_keys = {
        "model_name",
        "predictions",
        "confusion_matrix",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
    }

    assert expected_keys.issubset(result.keys())

    assert result["model_name"] == "decision_tree"