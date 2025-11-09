# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Employee Salary Predictor")
st.write("Predict employee salary based on experience and age using a machine learning model.")

# --- Load model and scaler ---
@st.cache_resource
@st.cache_resource
def load_model():
    model = joblib.load("models/salary_model.joblib")   # changed here
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler


model, scaler = load_model()

# --- User Input ---
st.sidebar.header("Input Features")

experience = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1, value=2.0)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1, value=25)

if st.sidebar.button("Predict Salary"):
    X_new = pd.DataFrame([[experience, age]], columns=['YearsExperience', 'Age'])
    X_scaled = scaler.transform(X_new)
    prediction = model.predict(X_scaled)[0]
    st.subheader("Predicted Salary")
    st.success(f"💰 Estimated Salary: ₹ {prediction:,.2f}")

st.write("---")
st.write("Model trained using Linear Regression, Random Forest, and Gradient Boosting (Voting Regressor).")
