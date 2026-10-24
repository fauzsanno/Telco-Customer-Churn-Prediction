"""
Telco Customer Churn Risk — Streamlit app
Overview + interactive prediction, what-if analysis, business simulator, and batch scoring.

Run:  streamlit run app.py
Data: put Telco-Customer-Churn.csv next to this file (or in ./data/). If it is missing,
      the app downloads the public IBM copy automatically.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Telco Churn Risk", page_icon="📡", layout="wide")

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
SEED = 42
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_CANDIDATES = ["Telco-Customer-Churn.csv", "data/Telco-Customer-Churn.csv",
                   "WA_Fn-UseC_-Telco-Customer-Churn.csv", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"]
CACHE_PATH = "model_cache.joblib"

# Hyperparameter LightGBM hasil tuning Optuna pada penelitian (objektif PR-AUC, CV 5-fold)
LGBM_PARAMS = dict(n_estimators=242, learning_rate=0.02735, num_leaves=5, min_child_samples=36,
                   subsample=0.664, colsample_bytree=0.755, reg_alpha=0.00238, reg_lambda=0.00146)
DEFAULT_THRESHOLD = 0.36          # threshold optimal berbasis biaya pada penelitian
SEG_BINS, SEG_LABELS = [0, 0.3, 0.6, 1.0], ["Low", "Medium", "High"]
SEG_COLORS = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#c0392b"}
ACTIONS = {
    "High": "Contact immediately: offer a 1–2 year contract upgrade, loyalty discount, and a bundle with Online Security / Tech Support.",
    "Medium": "Automated campaign: encourage autopay, educate on add-on services, monitor usage.",
    "Low": "Maintain: loyalty program, no extra promo spend.",
}
# Hasil notebook asli (direproduksi pada split yang sama) sebagai pembanding
BASELINE = {"Model": "Original notebook (Gradient Boosting)", "Accuracy": 0.7982, "Recall": 0.5187,
            "F1": 0.5774, "ROC-AUC": 0.8417}

RAW_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


# ----------------------------------------------------------------------------
# Data & fitur
# ----------------------------------------------------------------------------
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["n_services"] = sum((d[c] == "Yes").astype(int) for c in SERVICE_COLS)
    d["n_protect"] = sum((d[c] == "Yes").astype(int) for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"])
    d["n_stream"] = sum((d[c] == "Yes").astype(int) for c in ["StreamingTV", "StreamingMovies"])
    d["has_internet"] = (d["InternetService"] != "No").astype(int)
    d["fiber"] = (d["InternetService"] == "Fiber optic").astype(int)
    d["mtm_contract"] = (d["Contract"] == "Month-to-month").astype(int)
    d["e_check"] = (d["PaymentMethod"] == "Electronic check").astype(int)
    d["autopay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    d["family"] = ((d["Partner"] == "Yes") | (d["Dependents"] == "Yes")).astype(int)
    d["avg_monthly_paid"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"].replace(0, np.nan), d["MonthlyCharges"])
    d["charge_delta"] = d["MonthlyCharges"] - d["avg_monthly_paid"]
    d["charge_per_service"] = d["MonthlyCharges"] / (1 + d["n_services"])
    d["new_customer"] = (d["tenure"] <= 6).astype(int)
    d["risky_combo"] = d["mtm_contract"] * d["fiber"] * d["e_check"]
    d["no_protect_internet"] = ((d["has_internet"] == 1) & (d["n_protect"] == 0)).astype(int)
    d["tenure_bin"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 48, 100],
                             labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)
    return d


def prepare(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Data mentah (kolom asli Telco, tanpa Churn) -> fitur model."""
    d = raw_df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d.loc[d["tenure"] == 0, "TotalCharges"] = 0.0
    d["TotalCharges"] = d["TotalCharges"].fillna(d["MonthlyCharges"] * d["tenure"])
    return add_features(d[RAW_COLS])


@st.cache_data(show_spinner=False)
def load_raw():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return pd.read_csv(p)
    try:
        return pd.read_csv(URL)
    except Exception:
        return None


def make_pre(num_cols, cat_cols, scale):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols)])


def expected_cost(y_true, proba, thr, cost_fn, cost_fp, save_rate):
    pred = proba >= thr
    fn = ((y_true == 1) & (~pred)).sum()
    fp = ((y_true == 0) & pred).sum()
    tp = ((y_true == 1) & pred).sum()
    return fn * cost_fn + fp * cost_fp + tp * (cost_fp + cost_fn * (1 - save_rate))


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def seg_of(p):
    return pd.cut(np.atleast_1d(p), SEG_BINS, labels=SEG_LABELS, include_lowest=True)


# ----------------------------------------------------------------------------
# Training (di-cache). Hanya berjalan sekali; hasil disimpan ke model_cache.joblib
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_artifacts():
    raw = load_raw()
    if raw is None:
        return None

    if os.path.exists(CACHE_PATH):
        try:
            art = joblib.load(CACHE_PATH)
            if art.get("sklearn") == sklearn.__version__ and art.get("v") == 2:
                return art
        except Exception:
            pass

    y = (raw["Churn"] == "Yes").astype(int)
    X = prepare(raw.drop(columns=[c for c in ["customerID", "Churn"] if c in raw.columns]))
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    lr = Pipeline([("pre", make_pre(num_cols, cat_cols, True)),
                   ("m", LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000, random_state=SEED))])
    cat = Pipeline([("pre", make_pre(num_cols, cat_cols, False)),
                    ("m", CatBoostClassifier(iterations=400, learning_rate=0.04, depth=4, auto_class_weights="Balanced",
                                             verbose=0, random_seed=SEED, thread_count=-1, allow_writing_files=False))])
    lgb = Pipeline([("pre", make_pre(num_cols, cat_cols, False)),
                    ("m", LGBMClassifier(subsample_freq=1, is_unbalance=True, verbose=-1, n_jobs=-1,
                                         random_state=SEED, **LGBM_PARAMS))])
    stack = StackingClassifier(estimators=[("lr", lr), ("cat", cat), ("lgb", lgb)],
                               final_estimator=LogisticRegression(C=1.0, max_iter=1000),
                               stack_method="predict_proba", cv=5)
    model = CalibratedClassifierCV(stack, method="sigmoid", cv=3).fit(X_tr, y_tr)
    lgb_explain = clone(lgb).fit(X_tr, y_tr)         # komponen LightGBM untuk penjelasan fitur

    p_te = model.predict_proba(X_te)[:, 1]
    names = [n.split("__", 1)[1] for n in lgb_explain.named_steps["pre"].get_feature_names_out()]
    Xt = lgb_explain.named_steps["pre"].transform(X_te)
    contrib = lgb_explain.named_steps["m"].predict(Xt, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=names).sort_values(ascending=False)

    eda = X.copy()
    eda["Churn"] = y.values
    art = dict(v=2, sklearn=sklearn.__version__, model=model, lgb=lgb_explain, features=X.columns.tolist(),
               names=names, imp=imp, X_test=X_te, y_test=y_te.values, p_test=p_te, eda=eda,
               train_size=len(X_tr))
    try:
        joblib.dump(art, CACHE_PATH)
    except Exception:
        pass
    return art


def score(art, raw_df):
    feats = prepare(raw_df)[art["features"]]
    return art["model"].predict_proba(feats)[:, 1], feats


def local_drivers(art, feats, top=6):
    pre, m = art["lgb"].named_steps["pre"], art["lgb"].named_steps["m"]
    Xt = pre.transform(feats)
    c = m.predict(Xt, pred_contrib=True)[0][:-1]
    s = pd.Series(c, index=art["names"])
    s = s.iloc[np.argsort(-np.abs(s.values))[:top]]
    return s.sort_values()


# ----------------------------------------------------------------------------
# Muat data & model
# ----------------------------------------------------------------------------
st.sidebar.title("📡 Telco Churn Risk")
raw_data = load_raw()
if raw_data is None:
    st.error("Dataset not found and the download failed. Upload Telco-Customer-Churn.csv below.")
    up = st.file_uploader("Telco-Customer-Churn.csv", type="csv")
    if up is None:
        st.stop()
    raw_data = pd.read_csv(up)
    raw_data.to_csv(DATA_CANDIDATES[0], index=False)
    load_raw.clear()

with st.spinner("Preparing the model (first run only, ~1–3 min: stacking + calibration)…"):
    art = get_artifacts()
if art is None:
    st.error("Could not prepare the model.")
    st.stop()

y_test, p_test = art["y_test"], art["p_test"]

# Threshold global (sidebar)
st.session_state.setdefault("thr", DEFAULT_THRESHOLD)
st.sidebar.slider("Decision threshold", 0.05, 0.95, step=0.01, key="thr",
                  help="Customers with churn probability ≥ threshold are flagged as 'will churn'. "
                       "Lower = catch more churners (higher recall) but more false alarms.")
thr = st.session_state["thr"]
st.sidebar.caption(f"Research default: **{DEFAULT_THRESHOLD}** (cost-optimal). Standard 0.50 is shown for comparison in the Business tab.")
st.sidebar.divider()
st.sidebar.caption(f"Training: {art['train_size']:,} customers · Test: {len(y_test):,} customers\n\n"
                   "Model: stacking ensemble (Logistic Regression + CatBoost + LightGBM) with sigmoid probability calibration.")

st.title("Telco Customer Churn Risk")
st.caption("Predict which telecom customers are likely to leave, understand why, and decide whom to target first.")
tab_over, tab_pred, tab_biz, tab_batch = st.tabs(
    ["📊 Overview", "🔮 Predict a customer", "💰 Business simulator", "📁 Batch scoring"])

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tab_over:
    m = metrics_at(y_test, p_test, thr)
    roc_auc, pr_auc = roc_auc_score(y_test, p_test), average_precision_score(y_test, p_test)
    st.subheader("Model performance (held-out test set)")
    c = st.columns(6)
    c[0].metric("ROC-AUC", f"{roc_auc:.3f}")
    c[1].metric("PR-AUC", f"{pr_auc:.3f}")
    c[2].metric("Recall (churn)", f"{m['Recall']:.1%}", f"{m['Recall'] - BASELINE['Recall']:+.1%} vs original")
    c[3].metric("Precision", f"{m['Precision']:.1%}")
    c[4].metric("F1", f"{m['F1']:.3f}", f"{m['F1'] - BASELINE['F1']:+.3f} vs original")
    c[5].metric("Accuracy", f"{m['Accuracy']:.1%}", f"{m['Accuracy'] - BASELINE['Accuracy']:+.1%} vs original")
    st.caption(f"Metrics at decision threshold **{thr:.2f}** (change it in the sidebar). "
               "The test set was never used for model selection, tuning, or threshold choice.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Original notebook vs. this approach**")
        cmp = pd.DataFrame({
            "Metric": ["Accuracy", "Recall (churn)", "F1", "ROC-AUC"],
            BASELINE["Model"]: [BASELINE["Accuracy"], BASELINE["Recall"], BASELINE["F1"], BASELINE["ROC-AUC"]],
            f"Stacking ensemble (thr {thr:.2f})": [m["Accuracy"], m["Recall"], m["F1"], roc_auc]})
        cmp["Change"] = cmp.iloc[:, 2] - cmp.iloc[:, 1]
        st.dataframe(cmp.style.format({c_: "{:.3f}" for c_ in cmp.columns[1:]}), hide_index=True)
        st.info("Accuracy drops slightly **by design**: a lower threshold catches many more churners (recall ↑) "
                "at the cost of more false alarms. On imbalanced data (73% non-churn), accuracy alone is misleading — "
                "predicting “no churn” for everyone already scores 73%.")
    with right:
        fig = px.imshow([[m["TN"], m["FP"]], [m["FN"], m["TP"]]], text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: stay", "Predicted: churn"], y=["Actual: stay", "Actual: churn"],
                        title=f"Confusion matrix (threshold {thr:.2f})")
        fig.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=50, b=10))
        st.plotly_chart(fig)

    sub1, sub2, sub3 = st.tabs(["Curves & lift", "Data insights", "Churn drivers"])

    with sub1:
        c1, c2 = st.columns(2)
        fpr, tpr, _ = roc_curve(y_test, p_test)
        f1_ = go.Figure([go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}", line=dict(color="#2980b9")),
                         go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False)])
        f1_.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=340)
        c1.plotly_chart(f1_)
        pr_, rc_, _ = precision_recall_curve(y_test, p_test)
        f2_ = go.Figure([go.Scatter(x=rc_, y=pr_, name=f"PR-AUC = {pr_auc:.3f}", line=dict(color="#8e44ad")),
                         go.Scatter(x=[0, 1], y=[y_test.mean()] * 2, line=dict(dash="dash", color="grey"), name="Base rate")])
        f2_.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision", height=340)
        c2.plotly_chart(f2_)

        res = pd.DataFrame({"y": y_test, "p": p_test})
        res["decile"] = pd.qcut(res["p"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
        lift = res.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_proba=("p", "mean"))
        lift["churn_rate"] = lift["churners"] / lift["customers"]
        lift["lift"] = lift["churn_rate"] / res["y"].mean()
        lift["cum_capture"] = lift["churners"].cumsum() / lift["churners"].sum()
        c3, c4 = st.columns(2)
        f3 = px.bar(lift.reset_index(), x="decile", y="churn_rate", title="Churn rate by risk decile (1 = highest risk)",
                    color_discrete_sequence=["#c0392b"])
        f3.add_hline(y=res["y"].mean(), line_dash="dash", annotation_text="average")
        f3.update_layout(height=340, yaxis_tickformat=".0%")
        c3.plotly_chart(f3)
        gain = pd.DataFrame({"decile": [0] + list(lift.index), "captured": [0] + list(lift["cum_capture"])})
        f4 = px.line(gain, x="decile", y="captured", markers=True, title="Cumulative gain: % of churners reached")
        f4.add_shape(type="line", x0=0, y0=0, x1=10, y1=1, line=dict(dash="dash", color="grey"))
        f4.update_layout(height=340, yaxis_tickformat=".0%")
        c4.plotly_chart(f4)
        st.success(f"Targeting the top **20%** riskiest customers reaches **{lift['cum_capture'].iloc[1]:.0%}** of all churners; "
                   f"the top **40%** reaches **{lift['cum_capture'].iloc[3]:.0%}**.")

        st.markdown("**Risk segments (test set)**")
        seg = pd.DataFrame({"Segment": seg_of(p_test), "Churned": y_test})
        seg_tbl = seg.groupby("Segment", observed=True).agg(Customers=("Churned", "size"), Churned=("Churned", "sum"),
                                                            ChurnRate=("Churned", "mean")).reindex(SEG_LABELS[::-1])
        seg_tbl["Recommended action"] = seg_tbl.index.map(ACTIONS)
        st.dataframe(seg_tbl.style.format({"ChurnRate": "{:.1%}"}))

    with sub2:
        eda = art["eda"]
        st.markdown(f"**{len(eda):,} customers** · churn rate **{eda['Churn'].mean():.1%}** · "
                    f"avg tenure **{eda['tenure'].mean():.0f} months** · avg monthly charge **${eda['MonthlyCharges'].mean():.2f}**")
        c1, c2 = st.columns(2)
        for col, title, holder in [("Contract", "Churn rate by contract", c1),
                                   ("InternetService", "Churn rate by internet service", c2),
                                   ("PaymentMethod", "Churn rate by payment method", c1),
                                   ("tenure_bin", "Churn rate by tenure (months)", c2)]:
            g = eda.groupby(col)["Churn"].mean().reset_index()
            if col == "tenure_bin":
                g[col] = pd.Categorical(g[col], ["0-6", "7-12", "13-24", "25-48", "49+"], ordered=True)
                g = g.sort_values(col)
            fig = px.bar(g, x=col, y="Churn", title=title, color_discrete_sequence=["#e67e22"])
            fig.update_layout(height=320, yaxis_tickformat=".0%", yaxis_title="Churn rate", xaxis_title="")
            holder.plotly_chart(fig)

    with sub3:
        st.markdown("Average absolute contribution of each feature to the churn score (LightGBM component, test set).")
        top = art["imp"].head(15)[::-1]
        fig = px.bar(x=top.values, y=[n.replace("_", " ") for n in top.index], orientation="h",
                     labels={"x": "mean |contribution| (log-odds)", "y": ""}, color_discrete_sequence=["#2c3e50"])
        fig.update_layout(height=520)
        st.plotly_chart(fig)
        st.caption("Reading: month-to-month contracts, short tenure, high price per service, no online security, "
                   "fiber internet and electronic-check payment are the strongest churn signals. "
                   "These are statistical associations, not proven causes.")

    with st.expander("Methodology"):
        st.markdown(
            "- **Leakage-free pipeline**: all preprocessing lives inside scikit-learn Pipelines; the test set is used once.\n"
            "- **Feature engineering**: service counts, price per service, charge delta vs. historical average, new-customer flag, "
            "risky combination (month-to-month × fiber × e-check), tenure bins.\n"
            "- **Model**: stacking of Logistic Regression, CatBoost and tuned LightGBM (Optuna, PR-AUC objective, 5-fold CV), "
            "final logistic meta-learner, sigmoid probability calibration.\n"
            "- **Decision threshold**: chosen from out-of-fold predictions on the training data using business costs, not the default 0.50.\n"
            "- **Dataset**: public IBM Telco Customer Churn sample (7,043 customers, 26.5% churn). ROC-AUC on this dataset "
            "plateaus around 0.85, so large accuracy jumps are not realistic.")

# ============================================================================
# TAB 2 — PREDIKSI INTERAKTIF + WHAT-IF
# ============================================================================
PRESETS = {
    "Typical loyal customer": dict(in_gender="Female", in_senior="No", in_partner="Yes", in_dependents="Yes", in_tenure=60,
                                   in_PhoneService="Yes", in_MultipleLines="Yes", in_InternetService="DSL",
                                   in_OnlineSecurity="Yes", in_OnlineBackup="Yes", in_DeviceProtection="Yes",
                                   in_TechSupport="Yes", in_StreamingTV="No", in_StreamingMovies="No",
                                   in_Contract="Two year", in_PaperlessBilling="No",
                                   in_PaymentMethod="Bank transfer (automatic)", in_monthly=65.0,
                                   in_auto_total=True, in_total=3900.0),
    "High-risk newcomer": dict(in_gender="Male", in_senior="No", in_partner="No", in_dependents="No", in_tenure=2,
                               in_PhoneService="Yes", in_MultipleLines="No", in_InternetService="Fiber optic",
                               in_OnlineSecurity="No", in_OnlineBackup="No", in_DeviceProtection="No",
                               in_TechSupport="No", in_StreamingTV="Yes", in_StreamingMovies="Yes",
                               in_Contract="Month-to-month", in_PaperlessBilling="Yes",
                               in_PaymentMethod="Electronic check", in_monthly=95.0,
                               in_auto_total=True, in_total=190.0),
    "Mid-risk customer": dict(in_gender="Female", in_senior="Yes", in_partner="No", in_dependents="No", in_tenure=18,
                              in_PhoneService="Yes", in_MultipleLines="No", in_InternetService="Fiber optic",
                              in_OnlineSecurity="No", in_OnlineBackup="Yes", in_DeviceProtection="No",
                              in_TechSupport="No", in_StreamingTV="No", in_StreamingMovies="No",
                              in_Contract="One year", in_PaperlessBilling="Yes",
                              in_PaymentMethod="Mailed check", in_monthly=80.0,
                              in_auto_total=True, in_total=1440.0),
}
for k, v in PRESETS["Mid-risk customer"].items():
    st.session_state.setdefault(k, v)


def apply_preset():
    st.session_state.update(PRESETS[st.session_state["preset_choice"]])


def yn(label, key, col=None):
    return (col or st).selectbox(label, ["No", "Yes"], key=key)


with tab_pred:
    st.subheader("Predict churn risk for one customer")
    st.selectbox("Load an example profile", list(PRESETS.keys()), index=2, key="preset_choice", on_change=apply_preset)
    form, result = st.columns([1.2, 1])

    with form:
        a, b = st.columns(2)
        with a:
            st.markdown("**Profile**")
            gender = st.selectbox("Gender", ["Female", "Male"], key="in_gender")
            senior = yn("Senior citizen", "in_senior")
            partner = yn("Has partner", "in_partner")
            dependents = yn("Has dependents", "in_dependents")
            tenure = st.slider("Tenure (months)", 0, 72, key="in_tenure")
            st.markdown("**Contract & billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="in_Contract")
            paperless = yn("Paperless billing", "in_PaperlessBilling")
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                       "Bank transfer (automatic)", "Credit card (automatic)"], key="in_PaymentMethod")
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, step=0.5, key="in_monthly")
            auto_total = st.checkbox("Estimate total charges (monthly × tenure)", key="in_auto_total")
            if auto_total:
                total = float(monthly * tenure)
                st.caption(f"Total charges ≈ ${total:,.2f}")
            else:
                total = st.number_input("Total charges ($)", 0.0, 9000.0, step=10.0, key="in_total")
        with b:
            st.markdown("**Services**")
            phone = yn("Phone service", "in_PhoneService")
            multi = (st.selectbox("Multiple lines", ["No", "Yes"], key="in_MultipleLines")
                     if phone == "Yes" else "No phone service")
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="in_InternetService")
            if internet == "No":
                st.caption("No internet → add-on services are set to “No internet service”.")
                sec = backup = prot = tech = tv = mov = "No internet service"
            else:
                sec = yn("Online security", "in_OnlineSecurity")
                backup = yn("Online backup", "in_OnlineBackup")
                prot = yn("Device protection", "in_DeviceProtection")
                tech = yn("Tech support", "in_TechSupport")
                tv = yn("Streaming TV", "in_StreamingTV")
                mov = yn("Streaming movies", "in_StreamingMovies")

    cust = dict(gender=gender, SeniorCitizen=int(senior == "Yes"), Partner=partner, Dependents=dependents,
                tenure=int(tenure), PhoneService=phone, MultipleLines=multi, InternetService=internet,
                OnlineSecurity=sec, OnlineBackup=backup, DeviceProtection=prot, TechSupport=tech,
                StreamingTV=tv, StreamingMovies=mov, Contract=contract, PaperlessBilling=paperless,
                PaymentMethod=payment, MonthlyCharges=float(monthly), TotalCharges=float(total))
    prob, feats = score(art, pd.DataFrame([cust]))
    prob = float(prob[0])
    segment = str(seg_of(prob)[0])

    with result:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2c3e50"},
                   "steps": [{"range": [0, 30], "color": "#a9dfbf"}, {"range": [30, 60], "color": "#fad7a0"},
                             {"range": [60, 100], "color": "#f5b7b1"}],
                   "threshold": {"line": {"color": "black", "width": 4}, "value": thr * 100}}))
        gauge.update_layout(height=260, margin=dict(t=60, b=0, l=20, r=20))
        st.plotly_chart(gauge)
        flagged = prob >= thr
        st.markdown(f"### Risk segment: :{'red' if segment == 'High' else 'orange' if segment == 'Medium' else 'green'}[{segment}]")
        st.markdown(f"Decision at threshold {thr:.2f}: **{'⚠️ likely to churn' if flagged else '✅ likely to stay'}**")
        st.info(ACTIONS[segment])

        drv = local_drivers(art, feats)
        fig = px.bar(x=drv.values, y=[n.replace("_", " ") for n in drv.index], orientation="h",
                     color=drv.values > 0, color_discrete_map={True: "#c0392b", False: "#27ae60"},
                     labels={"x": "contribution to churn score (log-odds)", "y": ""}, title="Why this score? (top drivers)")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=50, b=10))
        st.plotly_chart(fig)
        st.caption("Red pushes the score toward churn, green toward staying (LightGBM component of the ensemble).")

    # ---------------- What-if ----------------
    st.divider()
    st.subheader("What-if: which retention offer lowers the risk most?")
    scenarios = {}
    if contract != "One year":
        scenarios["Move to a 1-year contract"] = {"Contract": "One year"}
    if contract != "Two year":
        scenarios["Move to a 2-year contract"] = {"Contract": "Two year"}
    if "automatic" not in payment:
        scenarios["Switch to automatic payment"] = {"PaymentMethod": "Bank transfer (automatic)"}
    if internet != "No" and (sec == "No" or tech == "No"):
        scenarios["Add Online Security + Tech Support"] = {"OnlineSecurity": "Yes", "TechSupport": "Yes"}
    if scenarios:
        rows = []
        for name, change in scenarios.items():
            p_new = float(score(art, pd.DataFrame([{**cust, **change}]))[0][0])
            rows.append({"Offer": name, "New probability": p_new, "Change (pp)": (p_new - prob) * 100})
        wi = pd.DataFrame(rows).sort_values("Change (pp)")
        fig = px.bar(wi, x="Change (pp)", y="Offer", orientation="h", color="Change (pp)",
                     color_continuous_scale=["#27ae60", "#bdc3c7", "#c0392b"], color_continuous_midpoint=0)
        fig.update_layout(height=260, coloraxis_showscale=False, margin=dict(t=10, b=10))
        c1, c2 = st.columns([1.2, 1])
        c1.plotly_chart(fig)
        c2.dataframe(wi.style.format({"New probability": "{:.1%}", "Change (pp)": "{:+.1f}"}), hide_index=True)
        st.caption("The model shows how the predicted risk changes for customers with these attributes. "
                   "These are associations in historical data, not guaranteed causal effects — validate offers with an A/B test.")
    else:
        st.write("This customer already has the main protective attributes; no what-if offers apply.")

# ============================================================================
# TAB 3 — BUSINESS SIMULATOR
# ============================================================================
with tab_biz:
    st.subheader("Business simulator: pick the threshold that minimizes cost")
    c1, c2, c3 = st.columns(3)
    cost_fn = c1.number_input("Cost of a missed churner (lost customer value)", 50.0, 5000.0, 500.0, step=50.0)
    cost_fp = c2.number_input("Cost of a retention offer", 1.0, 1000.0, 50.0, step=5.0)
    save_rate = c3.slider("Share of targeted churners successfully retained", 0.05, 0.95, 0.30, step=0.05)

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = np.array([expected_cost(y_test, p_test, t, cost_fn, cost_fp, save_rate) for t in grid])
    best_t = float(grid[int(np.argmin(costs))])
    cur_cost = expected_cost(y_test, p_test, thr, cost_fn, cost_fp, save_rate)
    default_cost = expected_cost(y_test, p_test, 0.5, cost_fn, cost_fp, save_rate)
    do_nothing = float(y_test.sum() * cost_fn)
    contact_all = expected_cost(y_test, p_test, 0.0, cost_fn, cost_fp, save_rate)

    fig = go.Figure(go.Scatter(x=grid, y=costs, line=dict(color="#2980b9"), name="Total cost"))
    fig.add_vline(x=thr, line_color="black", annotation_text=f"current {thr:.2f}")
    fig.add_vline(x=best_t, line_color="#27ae60", line_dash="dash", annotation_text=f"optimal {best_t:.2f}",
                  annotation_position="bottom right")
    fig.update_layout(title="Total expected cost vs. decision threshold (test set)", xaxis_title="Threshold",
                      yaxis_title="Cost", height=360)
    st.plotly_chart(fig)
    st.button(f"Apply optimal threshold ({best_t:.2f})", on_click=lambda: st.session_state.update(thr=best_t))

    k = st.columns(4)
    k[0].metric("Cost at current threshold", f"{cur_cost:,.0f}")
    k[1].metric("Cost at 0.50 threshold", f"{default_cost:,.0f}", f"{cur_cost - default_cost:+,.0f}", delta_color="inverse")
    k[2].metric("Cost if we do nothing", f"{do_nothing:,.0f}")
    k[3].metric("Net saving vs. doing nothing", f"{do_nothing - cur_cost:,.0f}")

    mt = metrics_at(y_test, p_test, thr)
    st.markdown(f"At threshold **{thr:.2f}** the team would contact **{mt['TP'] + mt['FP']:,}** of {len(y_test):,} customers "
                f"({(mt['TP'] + mt['FP']) / len(y_test):.0%}), reaching **{mt['TP']:,}** of {int(y_test.sum()):,} real churners "
                f"({mt['Recall']:.0%}) with {mt['FP']:,} unnecessary offers.")
    st.caption("Cost per customer: missed churner = lost value; targeted churner = offer cost + the unretained share of lost value; "
               "false alarm = offer cost. Defaults are illustrative assumptions — replace them with your real numbers. "
               "With these defaults the cost curve is quite flat, so the choice mainly shifts the recall/precision balance.")

# ============================================================================
# TAB 4 — BATCH SCORING
# ============================================================================
with tab_batch:
    st.subheader("Score a customer file")
    st.caption("Upload a CSV with the original Telco columns (customerID optional, Churn not required).")
    sample = art["X_test"][RAW_COLS].head(50)
    st.download_button("Download a sample template (50 rows)", sample.to_csv(index=False).encode(),
                       "telco_sample.csv", "text/csv")
    up = st.file_uploader("Customer CSV", type="csv", key="batch_up")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            missing = [c for c in RAW_COLS if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                p, _ = score(art, df_in)
                out = df_in.copy()
                out["churn_proba"] = p
                out["predicted_churn"] = (p >= thr).astype(int)
                out["risk_segment"] = seg_of(p).astype(str)
                out = out.sort_values("churn_proba", ascending=False)
                k = st.columns(3)
                k[0].metric("Customers scored", f"{len(out):,}")
                k[1].metric(f"Flagged (≥ {thr:.2f})", f"{int(out['predicted_churn'].sum()):,}")
                k[2].metric("High-risk segment", f"{int((out['risk_segment'] == 'High').sum()):,}")
                dist = out["risk_segment"].value_counts().reindex(SEG_LABELS).fillna(0).reset_index()
                dist.columns = ["Segment", "Customers"]
                fig = px.bar(dist, x="Segment", y="Customers", color="Segment", color_discrete_map=SEG_COLORS)
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig)
                st.dataframe(out)
                st.download_button("Download scored file", out.to_csv(index=False).encode(),
                                   "churn_scored.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")
