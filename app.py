"""Loan Approval Predictor - Streamlit web app.  Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from train import add_features, CAT, NUM  # noqa: E402

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "loan_model.joblib")


@st.cache_data
def load_metrics():
    return json.load(open(ROOT / "reports" / "metrics.json"))


model, metrics = load_model(), load_metrics()

st.title("🏦 Loan Approval Prediction")
st.caption("B.Tech CSE (AI & ML) final year project - ML-based credit decision support")

tab_pred, tab_perf, tab_about = st.tabs(["Predict", "Model performance", "About"])

with tab_pred:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Applicant")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_emp = st.selectbox("Self employed", ["No", "Yes"])
    with c2:
        st.subheader("Finances")
        app_inc = st.number_input("Applicant income (monthly)", 0, 200000, 5000, 500)
        co_inc = st.number_input("Co-applicant income (monthly)", 0, 100000, 0, 500)
        credit = st.selectbox("Credit history meets guidelines?", ["Yes (1)", "No (0)"])
    with c3:
        st.subheader("Loan")
        loan_amt = st.number_input("Loan amount (in thousands)", 5, 1000, 130, 5)
        term = st.selectbox("Term (months)", [360, 300, 240, 180, 120, 480, 84, 60])
        area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": app_inc, "CoapplicantIncome": co_inc,
            "LoanAmount": loan_amt, "Loan_Amount_Term": float(term),
            "Credit_History": 1.0 if credit.startswith("Yes") else 0.0,
            "Property_Area": area}])
        X = add_features(row)[CAT + NUM]
        p = float(model.predict_proba(X)[0, 1])
        emi = X["EMI"].iloc[0]
        ratio = X["EMI_to_Income"].iloc[0]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Approval probability", f"{p:.1%}")
        m2.metric("Estimated EMI / month", f"Rs {emi:,.0f}")
        m3.metric("EMI as % of income", f"{ratio:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= 0.5:
            st.success("Likely APPROVED")
        else:
            st.error("Likely REJECTED")

        st.markdown("**Factors worth noting**")
        notes = []
        if credit.startswith("No"):
            notes.append("Credit history is the strongest driver in this model; a poor record sharply lowers approval odds.")
        if ratio > 0.4:
            notes.append("EMI is above 40% of income, which lenders treat as high repayment burden.")
        if loan_amt * 1000 > 60 * (app_inc + co_inc):
            notes.append("Loan amount is large relative to income.")
        if not notes:
            notes.append("No major red flags detected on the key factors.")
        for n in notes:
            st.write("- " + n)
        st.caption("Educational demo trained on synthetic data - not for real lending decisions.")

with tab_perf:
    st.subheader(f"Selected model: {metrics['best_model']}")
    cols = st.columns(5)
    for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(k.replace("_", " ").title(), f"{metrics[k]:.3f}")
    fig = ROOT / "reports" / "figures"
    a, b = st.columns(2)
    a.image(str(fig / "03_confusion_matrix.png"), caption="Confusion matrix")
    b.image(str(fig / "04_roc.png"), caption="ROC curves")
    a.image(str(fig / "05_model_comparison.png"), caption="Model comparison (5-fold CV)")
    b.image(str(fig / "06_feature_importance.png"), caption="Feature importance")

with tab_about:
    st.markdown("""
**Pipeline:** data cleaning -> feature engineering (total income, EMI, EMI-to-income ratio, log transforms)
-> imputation + scaling + one-hot encoding -> model comparison (5-fold CV) -> hyper-parameter tuning
-> hold-out evaluation -> deployment in this app.

**Models compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
""")
