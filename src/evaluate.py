import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preprocessing import ROOT_DIR, load_config


def main() -> None:
    cfg = load_config()
    test_path = ROOT_DIR / cfg["paths"]["test_data"]
    model_path = ROOT_DIR / cfg["paths"]["model_path"]

    if not model_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python src/train.py` first.")

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Model Evaluation on Test Set ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))


if __name__ == "__main__":
    main()
