import sys, os, pytest, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import DataPreprocessor, prepare_data


SAMPLE_DATA = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Female"], "SeniorCitizen": [0, 1, 0, 0],
    "Partner": ["Yes", "No", "No", "Yes"], "Dependents": ["No", "Yes", "No", "Yes"],
    "tenure": [1, 72, 12, 36], "PhoneService": ["No", "Yes", "Yes", "Yes"],
    "MultipleLines": ["No phone service", "Yes", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic"],
    "OnlineSecurity": ["No", "Yes", "No", "Yes"], "OnlineBackup": ["Yes", "No", "No", "Yes"],
    "DeviceProtection": ["No", "Yes", "No", "Yes"], "TechSupport": ["No", "Yes", "No", "No"],
    "StreamingTV": ["No", "No", "No", "Yes"], "StreamingMovies": ["No", "Yes", "No", "No"],
    "Contract": ["Month-to-month", "Two year", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No", "Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "MonthlyCharges": [29.85, 99.45, 55.0, 80.0], "TotalCharges": [29.85, 7164.0, 660.0, 2880.0],
    "Churn": ["Yes", "No", "Yes", "No"],
})


class TestDataPreprocessor:
    def test_fit_transform_smoke(self):
        pre = DataPreprocessor()
        X = SAMPLE_DATA.drop(columns=["Churn"])
        out, names = pre.fit_transform(X)
        assert out.shape[0] == 4
        assert len(names) > 0

    def test_fit_transform_with_smote(self):
        pre = DataPreprocessor()
        X = SAMPLE_DATA.drop(columns=["Churn"])
        y = SAMPLE_DATA["Churn"]
        out, names = pre.fit_transform(X, use_smote=True, y=y)
        assert getattr(pre, "_smote_applied", False)

    def test_transform_after_fit(self):
        pre = DataPreprocessor()
        X = SAMPLE_DATA.drop(columns=["Churn"])
        pre.fit_transform(X)
        out = pre.transform(X)
        assert out.shape[0] == 4

    def test_engineered_features(self):
        pre = DataPreprocessor()
        X = SAMPLE_DATA.drop(columns=["Churn"])
        out, names = pre.fit_transform(X)
        assert any("tenure_group" in c for c in out.columns), "Missing tenure_group features"
        for col in ["avg_charge", "services_count", "has_contract", "high_value_new"]:
            assert col in out.columns, f"Missing engineered feature: {col}"

    def test_convert_numeric_columns(self):
        pre = DataPreprocessor()
        X = SAMPLE_DATA.drop(columns=["Churn"]).copy()
        X["TotalCharges"] = X["TotalCharges"].astype(object)
        X.loc[0, "TotalCharges"] = " "
        result = pre.convert_numeric_columns(X)
        assert pd.api.types.is_float_dtype(result["TotalCharges"])


class TestPrepareData:
    def test_prepare_data_smoke(self):
        # Use bigger synthetic data to avoid OneHotEncoder column mismatch
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(1, 72, n),
            "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
            "MonthlyCharges": np.random.uniform(18, 120, n),
            "TotalCharges": np.random.uniform(18, 8000, n),
            "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        })
        X_train, X_test, y_train, y_test, pre = prepare_data(df, test_size=0.3)
        assert X_train.shape[0] >= 5
        assert len(y_train) == X_train.shape[0]
