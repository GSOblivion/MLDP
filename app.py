import streamlit as st
import pandas as pd
import numpy as np
import joblib

# to run the app, open andaconda prompt, cd C:\Users\ruint\visual studio code\ML\Project, then conda activate mldp and use streamlit run app.py
# page config
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

# load model load feature names
@st.cache_resource
def load_model():
    model = joblib.load("stroke_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

st.image("Man Talking to Doctor.jpg", use_column_width=True)
st.caption("Have you ever consulted a doctor about your stroke risk? This tool can help you understand your risk factors.")

# app title and desc
st.title("🧠 Stroke Risk Prediction Tool")

st.markdown("""
This application estimates **stroke risk** based on patient health and lifestyle information.

⚠️ **Disclaimer:**  
This tool is for **educational and decision-support purposes only**.  
It is **not a medical diagnosis**.
""")

st.divider()

# input form
st.subheader("👤 Patient Information")

with st.form("stroke_form"):
    age = st.slider("Age (years)", 0, 100, 45)

    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox(
        "Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"]
    )

    avg_glucose = st.number_input(
        "Average Glucose Level (mg/dL)",
        min_value=50.0,
        max_value=300.0,
        value=100.0
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

# input validation
def validate_inputs():
    if age <= 0:
        return "Age must be greater than 0."
    if bmi < 10 or bmi > 60:
        return "BMI value is unrealistic."
    if avg_glucose < 50 or avg_glucose > 300:
        return "Glucose level is outside expected range."
    return None

# prediction and result display
if submitted:
    error = validate_inputs()

    if error:
        st.error(error)
    else:
        # Base input dictionary
        input_data = {
            "age": age,
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
        }

        # Initialize all features as 0
        model_input = dict.fromkeys(feature_names, 0)

        # Fill numerical features
        for key in input_data:
            model_input[key] = input_data[key]

        # Handle categorical (one-hot encoded)
        def set_ohe(prefix, value):
            col = f"{prefix}_{value}"
            if col in model_input:
                model_input[col] = 1

        set_ohe("gender", gender)
        set_ohe("ever_married", ever_married)
        set_ohe("work_type", work_type)
        set_ohe("Residence_type", residence)
        set_ohe("smoking_status", smoking_status)

        # Convert to DataFrame
        input_df = pd.DataFrame([model_input])

        # Prediction
        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ **High Stroke Risk Detected**\n\nEstimated Risk: **{probability:.2%}**")
        else:
            st.success(f"✅ **Low Stroke Risk Detected**\n\nEstimated Risk: **{probability:.2%}**")

        st.markdown("""
        **How to interpret this result:**
        - The percentage represents the model’s estimated probability of stroke.
        - This model should not be used for clinical diagnosis but as a supporting tool to assess stroke risk.
        """)

