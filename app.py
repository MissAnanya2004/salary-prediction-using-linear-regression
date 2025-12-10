import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Employee Salary Predictor")
st.write("Enter the details below to estimate the salary.")

@st.cache_resource
def load_model():
    model = joblib.load("models/salary_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler

model, scaler = load_model()

st.sidebar.header("Input")

experience = st.sidebar.number_input("Years of Experience", 0.0, 50.0, 2.0, step=0.1)
age = st.sidebar.number_input("Age", 18, 100, 25)

if st.sidebar.button("Predict Salary"):
    df = pd.DataFrame([[experience, age]], columns=["YearsExperience", "Age"])
    scaled = scaler.transform(df)
    result = model.predict(scaled)[0]
    st.subheader("Predicted Salary")
    st.success(f"Estimated Salary: ₹ {result:,.2f}")

st.write("---")
st.caption("Model uses a Voting Regressor combining Linear Regression, Random Forest, and Gradient Boosting.")
