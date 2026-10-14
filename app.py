
import streamlit as st
import pandas as pd

from src.predict import load_artifact, score_customers, RAW_FEATURES


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Prediksi risiko pelanggan berhenti berlangganan "
    "menggunakan model Machine Learning."
)


@st.cache_resource
def get_artifact():
    return load_artifact()


try:
    artifact = get_artifact()
except Exception as exc:
    st.error(f"Gagal memuat model: {exc}")
    st.stop()


st.subheader("Profil Pelanggan")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (bulan)", min_value=0, max_value=72, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )

    with col2:
        device = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
        )
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(monthly * tenure),
        )

    submitted = st.form_submit_button(
        "Prediksi Churn", type="primary"
    )


if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    try:
        result = score_customers(customer, artifact)
        probability = float(result["churn_proba"].iloc[0])
        prediction = int(result["predicted_churn"].iloc[0])
        risk = str(result["risk_segment"].iloc[0])

        left, middle, right = st.columns(3)
        left.metric("Probabilitas Churn", f"{probability:.2%}")
        middle.metric(
            "Prediksi",
            "Berisiko Churn" if prediction else "Tidak Diprediksi Churn",
        )
        right.metric("Kategori Risiko", risk)

        if prediction:
            st.warning(
                "Pelanggan ini melewati threshold model. "
                "Pertimbangkan evaluasi tindakan retensi."
            )
        else:
            st.success(
                "Pelanggan ini berada di bawah threshold "
                "klasifikasi churn."
            )

    except Exception as exc:
        st.error(f"Prediksi gagal: {exc}")
