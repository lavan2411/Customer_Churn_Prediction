import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_preprocessing import ROOT_DIR, load_config


def main() -> None:
    cfg = load_config()
    test_path = ROOT_DIR / cfg["paths"]["test_data"]
    model_path = ROOT_DIR / cfg["paths"]["model_path"]
    img_dir = ROOT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python src/train.py` first.")

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    print("=== Model Evaluation on Test Set ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # Save ROC curve & Confusion Matrix to images/
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(img_dir / "churn_evaluation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation chart to {img_dir / 'churn_evaluation.png'}")


if __name__ == "__main__":
    main()