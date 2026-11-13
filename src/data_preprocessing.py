from pathlib import Path
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT_DIR / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("=== 1. Raw Dataset Inspection ===")
    print("Shape:", df.shape)
    print("Data types:\n", df.dtypes)

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Strip whitespace in text columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Fix incorrect format: TotalCharges has blank spaces -> convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Check missing values and fill numeric NaNs with median
    print("\n=== 2. Missing Values Before Cleaning ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Check and remove duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n=== 3. Duplicate Rows Found: {duplicates} ===")
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Removed duplicates. New shape:", df.shape)

    # Check invalid negative values in numeric columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            invalid_count = (df[col] < 0).sum()
            if invalid_count > 0:
                print(f"Removing {invalid_count} invalid negative rows in {col}")
                df = df[df[col] >= 0].reset_index(drop=True)

    # Map target Churn ('Yes'/'No') to binary (1/0)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
        if df["Churn"].isnull().any():
            raise ValueError("Unexpected values found in Churn column.")
        df["Churn"] = df["Churn"].astype(int)

    print("\n=== 4. Cleaned Dataset Shape ===", df.shape)
    return df


def main() -> None:
    cfg = load_config()
    raw_path = ROOT_DIR / cfg["paths"]["raw_data"]
    interim_path = ROOT_DIR / cfg["paths"]["interim_data"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching Telco Customer Churn dataset from OpenML...")
    dataset = fetch_openml(data_id=cfg["dataset"]["openml_id"], as_frame=True, parser="auto")
    raw_df = dataset.frame
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")

    cleaned_df = inspect_and_clean_data(raw_df)
    cleaned_df.to_csv(interim_path, index=False)
    print(f"Saved cleaned data to {interim_path}")


if __name__ == "__main__":
    main()
