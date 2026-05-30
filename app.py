import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Telco Customer Churn Prediction")

st.title("📞 Telco Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will churn.")

# User Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# Encoding
gender = 1 if gender == "Male" else 0
Partner = 1 if Partner == "Yes" else 0
Dependents = 1 if Dependents == "Yes" else 0
PhoneService = 1 if PhoneService == "Yes" else 0
PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0

# Prediction
if st.button("Predict Churn"):

    input_data = np.array([[
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        PaperlessBilling,
        MonthlyCharges,
        TotalCharges
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    if probability > 0.5:
        st.error(
            f"⚠ Customer is likely to CHURN\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\nProbability: {(1-probability):.2%}"
        )

    st.write("Raw Churn Probability:", round(probability, 4))
