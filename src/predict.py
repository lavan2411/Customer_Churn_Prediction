import pickle
import pandas as pd

from data_preprocessing import ROOT_DIR, load_config
from feature_engineering import add_features


def predict_churn(customer_data: dict) -> tuple[str, float]:
    cfg = load_config()
    model_path = ROOT_DIR / cfg["paths"]["model_path"]

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = add_features(pd.DataFrame([customer_data]))
    prob = float(model.predict_proba(df)[0, 1])
    label = "Churn" if prob >= 0.5 else "No Churn"
    return label, prob


if __name__ == "__main__":
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
    label, prob = predict_churn(sample_customer)
    print(f"Prediction: {label} (Churn Probability: {prob:.1%})")
