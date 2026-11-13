import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict_churn

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 6)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
with col2:
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

if st.button("Predict"):
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": round(tenure * monthly_charges, 2),
    }
    try:
        label, prob = predict_churn(customer)
        st.metric("Prediction", label, f"{prob:.1%} churn probability")
    except FileNotFoundError as exc:
        st.error(str(exc))
