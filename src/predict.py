
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_artifact(path=None):
    """Memuat model final, threshold, dan urutan fitur."""
    if path is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "churn_model.joblib"
        )

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Artefak model tidak ditemukan: {path}"
        )

    artifact = joblib.load(path)

    for key in ("model", "threshold", "features"):
        if key not in artifact:
            raise ValueError(
                f"Artefak model tidak memiliki kunci: {key}"
            )

    return artifact


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Menerapkan feature engineering yang sama dengan notebook."""
    d = data.copy()

    d["n_services"] = sum(
        (d[c] == "Yes").astype(int) for c in SERVICE_COLS
    )

    d["n_protect"] = sum(
        (d[c] == "Yes").astype(int)
        for c in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
        ]
    )

    d["n_stream"] = sum(
        (d[c] == "Yes").astype(int)
        for c in ["StreamingTV", "StreamingMovies"]
    )

    d["has_internet"] = (d["InternetService"] != "No").astype(int)
    d["fiber"] = (d["InternetService"] == "Fiber optic").astype(int)
    d["mtm_contract"] = (d["Contract"] == "Month-to-month").astype(int)
    d["e_check"] = (d["PaymentMethod"] == "Electronic check").astype(int)

    d["autopay"] = (
        d["PaymentMethod"].astype(str)
        .str.contains("automatic")
        .astype(int)
    )

    d["family"] = (
        (d["Partner"] == "Yes") | (d["Dependents"] == "Yes")
    ).astype(int)

    d["avg_monthly_paid"] = np.where(
        d["tenure"] > 0,
        d["TotalCharges"] / d["tenure"],
        d["MonthlyCharges"],
    )

    d["charge_delta"] = (
        d["MonthlyCharges"] - d["avg_monthly_paid"]
    )

    d["charge_per_service"] = (
        d["MonthlyCharges"] / (1 + d["n_services"])
    )

    d["new_customer"] = (d["tenure"] <= 6).astype(int)

    d["risky_combo"] = (
        d["mtm_contract"] * d["fiber"] * d["e_check"]
    )

    d["no_protect_internet"] = (
        (d["has_internet"] == 1) & (d["n_protect"] == 0)
    ).astype(int)

    d["tenure_bin"] = pd.cut(
        d["tenure"],
        [-1, 6, 12, 24, 48, 100],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype(str)

    return d


def score_customers(
    raw_df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """
    Menerima data pelanggan Telco tanpa perlu kolom Churn.
    Menghasilkan probabilitas, prediksi biner, dan segmen risiko.
    """
    if raw_df.empty:
        raise ValueError("Data pelanggan tidak boleh kosong.")

    missing = [c for c in RAW_FEATURES if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}"
        )

    d = raw_df.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    ).copy()

    d["tenure"] = pd.to_numeric(d["tenure"], errors="raise")
    d["MonthlyCharges"] = pd.to_numeric(
        d["MonthlyCharges"], errors="raise"
    )
    d["TotalCharges"] = pd.to_numeric(
        d["TotalCharges"], errors="coerce"
    )

    d.loc[d["tenure"] == 0, "TotalCharges"] = 0.0

    if d["TotalCharges"].isna().any():
        raise ValueError(
            "TotalCharges memiliki nilai kosong pada pelanggan "
            "dengan tenure lebih dari 0."
        )

    d = add_features(d)

    features = artifact["features"]
    missing_model_features = [
        c for c in features if c not in d.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Fitur model tidak tersedia: "
            f"{missing_model_features}"
        )

    probabilities = artifact["model"].predict_proba(
        d[features]
    )[:, 1]

    threshold = float(artifact["threshold"])

    segments = pd.cut(
        probabilities,
        [0, 0.3, 0.6, 1.0],
        labels=["Rendah", "Sedang", "Tinggi"],
        include_lowest=True,
    )

    return pd.DataFrame(
        {
            "churn_proba": probabilities,
            "predicted_churn": (
                probabilities >= threshold
            ).astype(int),
            "risk_segment": segments,
        },
        index=raw_df.index,
    )
