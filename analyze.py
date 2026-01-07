from graphs import *
import numpy as np

from print_utils import separator


def detect_outliers():
    """
    Detects outliers in the dataset using IQR method
    :return: Nothing
    """
    df = original_diabetes_prediction_dataset
    numeric_columns = df.select_dtypes(include=["float64"]).columns
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"{column}: {len(outliers)} outliers detected")

def show_dataset_info():
    """
    Shows dataset info such as:
        1. General dataset information
        2. Null values per column
        3. Duplicates
    :return: Nothing
    """
    print("Dataset information:")
    print(original_diabetes_prediction_dataset.info(show_counts=True, verbose=True))
    separator()
    print("Amount of null values per column:")
    print(original_diabetes_prediction_dataset.isnull().sum())
    separator()
    duplicate_rows = original_diabetes_prediction_dataset[original_diabetes_prediction_dataset.duplicated()]
    print("Duplicate rows:", duplicate_rows.shape)

def show_columns_info():
    """
    Shows columns info for each column of the dataset such as:
        1. General column description
        2. Unique values of the column
    :return: Nothing
    """
    for column in original_diabetes_prediction_dataset.columns:
        print("Description of column \"" + column + "\"")
        if original_diabetes_prediction_dataset[column].dtype != np.float64:
            print("\nUnique values: " + str(original_diabetes_prediction_dataset[column].unique()))
        print("\nMore information about the column:")
        print(original_diabetes_prediction_dataset[column].describe())
        separator()
