from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import ROOT_DIR, load_config
from feature_engineering import add_features, build_preprocessor


def main() -> None:
    cfg = load_config()
    interim_path = ROOT_DIR / cfg["paths"]["interim_data"]
    train_path = ROOT_DIR / cfg["paths"]["train_data"]
    test_path = ROOT_DIR / cfg["paths"]["test_data"]
    model_path = ROOT_DIR / cfg["paths"]["model_path"]

    if not interim_path.exists():
        raise FileNotFoundError("Run `python src/data_preprocessing.py` first.")

    df = pd.read_csv(interim_path)
    df = add_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["dataset"]["test_size"],
        stratify=y,
        random_state=cfg["dataset"]["random_state"],
    )

    # Save processed train and test splits
    train_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
    print(f"Saved train/test splits to {train_path.parent}")

    preprocessor = build_preprocessor(X_train)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42),
    }

    best_model = None
    best_name = ""
    best_auc = 0.0

    print("\n=== Model Training & Comparison ===")
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"{name:22s} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = pipe

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved best model ({best_name}, ROC-AUC={best_auc:.4f}) to {model_path}")


if __name__ == "__main__":
    main()
