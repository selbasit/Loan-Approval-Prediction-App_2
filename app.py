import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the full pipeline (includes preprocessing + model)
with open("loan_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# App Title
st.title("🏦 Loan Approval Prediction App")
st.write("Provide applicant and loan details to predict loan approval status.")

# Input Fields
age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (USD)", min_value=1000, value=50000)
emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amount = st.number_input("Loan Amount (USD)", min_value=500, value=10000)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
loan_purpose = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "VENTURE", "PERSONAL"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
credit_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
default_on_file = st.selectbox("Default on File", ["Y", "N"])

# Derived feature
loan_percent_income = loan_amount / income

# 🔧 Build input with the exact same columns as training
input_data = {
    "id": 0,  # Required dummy column
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home_ownership,
    "person_emp_length": emp_length,
    "loan_intent": loan_purpose,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amount,
    "loan_int_rate": interest_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": default_on_file,
    "cb_person_cred_hist_length": credit_hist_length,
}

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    try:
        prediction = pipeline.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Denied")

        # Download button
        result_df = input_df.copy()
        result_df["prediction"] = "Approved" if prediction == 1 else "Denied"
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result", csv, "loan_prediction.csv", "text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
