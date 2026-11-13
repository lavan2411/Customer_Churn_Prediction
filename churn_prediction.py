"""
Customer Churn Prediction Pipeline (Real Dataset via Free OpenML API)
====================================================================
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENML_DATASET_ID = 42178  # telco-customer-churn (7,043 rows, 20 features)
RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent


def load_telco_churn_from_openml(dataset_id: int = OPENML_DATASET_ID) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch and clean the Telco Customer Churn dataset from OpenML's free API."""
    print(f"Fetching OpenML dataset ID {dataset_id} (telco-customer-churn)...")
    dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")

    X = dataset.data.copy()
    y_raw = dataset.target.copy()

    # Drop identifier column if present
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    # TotalCharges is stored as string with blank spaces for new customers; coerce to numeric
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    # Convert categorical columns to object/string so SimpleImputer & OneHotEncoder handle them cleanly
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].astype(object)

    # Map target ('Yes'/'No' or '1'/'0') to binary 1/0
    y = (
        y_raw.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0})
    )
    if y.isna().any():
        raise ValueError(f"Unexpected target values found: {y_raw.unique()}")

    print(f"Loaded {X.shape[0]} rows and {X.shape[1]} features.")
    return X, y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Train, cross-validate, and evaluate candidate churn classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "CV ROC-AUC": round(cv_auc, 4),
                "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred), 4),
            }
        )
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values(by="Test ROC-AUC", ascending=False)
    return results_df, fitted_pipelines


def get_feature_importances(pipe: Pipeline, top_n: int = 10) -> pd.DataFrame:
    """Extract top feature importances or absolute coefficients from the fitted pipeline."""
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    clf = pipe.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scores = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": scores})
    imp_df["Feature"] = (
        imp_df["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    )
    return imp_df.sort_values(by="Importance", ascending=False).head(top_n)


def save_evaluation_plot(
    fitted_pipelines: dict[str, Pipeline],
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save side-by-side ROC curve comparison and top feature importances."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    ax = axes[0]
    for name, pipe in fitted_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_title("ROC Curve Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    top_features = get_feature_importances(fitted_pipelines[best_model_name], top_n=10).iloc[::-1]
    ax2.barh(top_features["Feature"], top_features["Importance"], color="#2563eb")
    ax2.set_title(f"Top 10 Churn Drivers ({best_model_name})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Relative Importance")
    ax2.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def predict_customer_risk(pipe: Pipeline, customer_record: dict) -> tuple[float, str]:
    """Score a single customer dictionary and return (churn_probability, risk_tier)."""
    sample_df = pd.DataFrame([customer_record])
    prob = float(pipe.predict_proba(sample_df)[0, 1])
    if prob >= 0.65:
        tier = "HIGH RISK"
    elif prob >= 0.35:
        tier = "MEDIUM RISK"
    else:
        tier = "LOW RISK"
    return prob, tier


def main() -> None:
    X, y = load_telco_churn_from_openml(OPENML_DATASET_ID)
    print(f"Churn rate: {y.mean():.2%} ({y.sum()} churned / {len(y)} total)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)
    results_df, fitted_pipelines = evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    print("=== Model Comparison (Test Set) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_pipe = fitted_pipelines[best_name]
    y_pred_best = best_pipe.predict(X_test)

    print(f"\n=== Best Model: {best_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=["Retained", "Churned"]))

    print("=== Top 10 Churn Drivers ===")
    print(get_feature_importances(best_pipe, top_n=10).to_string(index=False))

    model_path = OUTPUT_DIR / "churn_model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"\nSaved trained pipeline to: {model_path}")

    plot_path = OUTPUT_DIR / "churn_evaluation.png"
    save_evaluation_plot(fitted_pipelines, best_name, X_test, y_test, plot_path)
    print(f"Saved evaluation chart to: {plot_path}")

    # Example single-customer prediction matching OpenML 42178 schema
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 268.50,
    }
    # Align columns with training data in case any extra/fewer columns exist
    aligned_sample = {col: sample_customer.get(col, np.nan) for col in X.columns}
    prob, tier = predict_customer_risk(best_pipe, aligned_sample)
    print(f"\nSample Customer Churn Probability: {prob:.1%} -> {tier}")


if __name__ == "__main__":
    main()