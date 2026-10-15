
import pandas as pd
import pytest

from src.predict import load_artifact, score_customers, RAW_FEATURES


@pytest.fixture(scope="module")
def artifact():
    """Memuat model satu kali untuk seluruh pengujian."""
    return load_artifact()


@pytest.fixture
def sample_customers():
    """Membuat tiga contoh pelanggan tanpa kolom target Churn."""
    return pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 70.0,
            },
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 65.0,
                "TotalCharges": 1500.0,
            },
            {
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 72,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 20.0,
                "TotalCharges": 1440.0,
            },
        ],
        columns=RAW_FEATURES,
    )


def test_model_artifact_loads(artifact):
    """Model harus memiliki komponen yang diperlukan."""
    assert artifact["model"] is not None
    assert 0 <= float(artifact["threshold"]) <= 1
    assert len(artifact["features"]) > 0


def test_predict_returns_valid_results(artifact, sample_customers):
    """Prediksi harus menghasilkan output yang valid."""
    results = score_customers(sample_customers, artifact)

    assert len(results) == len(sample_customers)
    assert {
        "churn_proba",
        "predicted_churn",
        "risk_segment",
    }.issubset(results.columns)

    assert results["churn_proba"].between(0, 1).all()
    assert results["predicted_churn"].isin([0, 1]).all()
    assert results["risk_segment"].notna().all()


def test_missing_required_column_raises_error(
    artifact, sample_customers
):
    """Kolom wajib yang hilang harus menghasilkan error."""
    invalid_customers = sample_customers.drop(
        columns=["tenure"]
    )

    with pytest.raises(ValueError, match="Kolom wajib"):
        score_customers(invalid_customers, artifact)


def test_empty_customer_data_raises_error(artifact):
    """Data pelanggan kosong harus ditolak."""
    empty_customers = pd.DataFrame(columns=RAW_FEATURES)

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        score_customers(empty_customers, artifact)
