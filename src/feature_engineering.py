import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features before preprocessing."""
    df = df.copy()
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build scaling and encoding pipeline for numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
