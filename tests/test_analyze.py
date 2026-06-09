import pandas as pd
import pytest

import analyze


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gender": ["Female", "Female", "Female", "Male"],
        "age": [80.0, 36.0, 78.0, 67.0],
        "hypertension": [0, 0, 0, 0],
        "heart_disease": [1, 0, 0, 1],
        "smoking_history": ["never", "current", "former", "not current"],
        "bmi": [25.19, 23.45, 36.05, 27.32],
        "HbA1c_level": [6.6, 5.0, 5.0, 6.5],
        "blood_glucose_level": [140, 155, 130, 200],
        "diabetes": [0, 0, 0, 1],
    })

def test_detect_outliers(sample_df, capsys):
    """"""
    analyze.original_diabetes_prediction_dataset = sample_df
    analyze.detect_outliers()

    captured = capsys.readouterr()

    assert "age:" in captured.out
    assert "outliers detected" in captured.out

def test_show_dataset_info(sample_df, capsys):
    """"""
    analyze.original_diabetes_prediction_dataset = sample_df
    analyze.show_dataset_info()

    captured = capsys.readouterr()

    assert "Dataset information:" in captured.out
    assert "Amount of null values per column:" in captured.out
    assert "Duplicate rows:" in captured.out

def test_show_columns_info(sample_df, capsys):
    """"""
    analyze.original_diabetes_prediction_dataset = sample_df
    analyze.show_columns_info()

    captured = capsys.readouterr()

    assert "Description of column \"gender\"" in captured.out
    assert "Description of column \"age\"" in captured.out
    assert "Description of column \"hypertension\"" in captured.out
    assert "Description of column \"heart_disease\"" in captured.out
    assert "Description of column \"smoking_history\"" in captured.out
    assert "Description of column \"bmi\"" in captured.out
    assert "Description of column \"HbA1c_level\"" in captured.out
    assert "Description of column \"blood_glucose_level\"" in captured.out
    assert "Description of column \"diabetes\"" in captured.out
    assert "More information about the column:" in captured.out
    assert "Unique values: ['Female' 'Male']" in captured.out
