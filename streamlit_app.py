# ----------------------------------------
# Imports
# ----------------------------------------
import streamlit as st
import requests

# ----------------------------------------
# Page config
# ----------------------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("Diabetes Risk Predictor")

# ----------------------------------------
# Input form
# ----------------------------------------
with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose", min_value=0, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
    insulin = st.number_input("Insulin", min_value=0, value=80)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=0, value=30)

    submitted = st.form_submit_button("Predict")

# ----------------------------------------
# Call FastAPI on submit
# ----------------------------------------
if submitted:
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.subheader(f"Risk: {result['risk_label']}")
        st.write(f"Probability: {result['probability']}")
    else:
        st.error(f"Error: {response.status_code} — {response.text}")