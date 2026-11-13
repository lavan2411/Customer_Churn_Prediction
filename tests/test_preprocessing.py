import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_preprocessing import inspect_and_clean_data
from feature_engineering import add_features, build_preprocessor


def test_inspect_and_clean_data():
    df = pd.DataFrame(
        {
            "customerID": ["001", "001", "002"],
            "tenure": [0, 0, 12],
            "MonthlyCharges": [50.0, 50.0, 70.0],
            "TotalCharges": [" ", " ", "840.0"],
            "Contract": ["Month-to-month", "Month-to-month", "One year"],
            "Churn": ["Yes", "Yes", "No"],
        }
    )
    cleaned = inspect_and_clean_data(df)
    assert "customerID" not in cleaned.columns
    assert len(cleaned) == 2
    assert cleaned["TotalCharges"].isnull().sum() == 0
    assert list(cleaned["Churn"]) == [1, 0]


def test_feature_engineering_and_preprocessor():
    df = pd.DataFrame(
        {
            "tenure": [2, 24],
            "MonthlyCharges": [80.0, 40.0],
            "TotalCharges": [160.0, 960.0],
            "Contract": ["Month-to-month", "Two year"],
        }
    )
    df = add_features(df)
    assert "avg_monthly_spend" in df.columns

    preprocessor = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 2
