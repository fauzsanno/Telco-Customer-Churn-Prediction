
import pandas as pd

from src.predict import load_artifact, score_customers

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

# Muat model final
artifact = load_artifact()

# Ambil lima contoh pelanggan tanpa kolom target
customers = pd.read_csv(DATA_URL).drop(columns=["Churn"])

# Jalankan prediksi
results = score_customers(customers.head(5), artifact)

print("Model berhasil dimuat.")
print(results.to_string(index=False))

assert len(results) == 5
assert results["churn_proba"].between(0, 1).all()
assert results["predicted_churn"].isin([0, 1]).all()

print("\nSemua pemeriksaan dasar berhasil.")
