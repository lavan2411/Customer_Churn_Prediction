# Customer Churn Prediction

End-to-end customer churn prediction project using the [Telco Customer Churn dataset (OpenML ID 42178)](https://www.openml.org/d/42178) fetched via scikit-learn's free [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) API.

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   ├── raw/                 # Raw dataset downloaded from OpenML
│   ├── interim/             # Cleaned dataset after validation & duplicate removal
│   └── processed/           # Train/test split CSVs ready for modeling
│
├── notebooks/
│   └── EDA.ipynb            # Exploratory Data Analysis & visualizations
│
├── src/
│   ├── data_preprocessing.py   # Data loading, inspection, invalid/missing/duplicate handling
│   ├── feature_engineering.py  # Feature creation, scaling & encoding pipeline
│   ├── train.py                # Train & compare models, save best model to models/churn_model.pkl
│   ├── evaluate.py             # Evaluate saved model on test data
│   └── predict.py              # Predict churn for a new customer
│
├── models/
│   └── churn_model.pkl      # Saved trained pipeline
│
├── config/
│   └── config.yaml          # Dataset & file path settings
│
├── tests/
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── app.py                   # Streamlit web app
```

## How to Run in VS Code

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run data loading, inspection, and cleaning:
   ```powershell
   python src/data_preprocessing.py
   ```
3. Explore the cleaned data in `notebooks/EDA.ipynb`.
4. Train and save the best model:
   ```powershell
   python src/train.py
   ```
5. Evaluate the model:
   ```powershell
   python src/evaluate.py
   ```
6. Run a sample prediction or launch the Streamlit app:
   ```powershell
   python src/predict.py
   streamlit run app.py
   ```
7. Run tests:
   ```powershell
   pytest
   ```
