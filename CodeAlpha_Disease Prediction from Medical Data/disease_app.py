import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Models load karein
model = joblib.load('disease_model.pkl')
scaler = joblib.load('disease_scaler.pkl')

st.title("🩺 Medical Disease Prediction System")
st.write("Enter patient vitals to predict Diabetes risk.")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    glu = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 80)
    ins = st.number_input("Insulin Level", 0, 900, 30)
    age = st.number_input("Age", 1, 120, 25)
with col2:
    bmi = st.number_input("BMI Index", 0.0, 70.0, 25.0)
    preg = st.number_input("Pregnancies", 0, 20, 0)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

if st.button("Predict Disease Risk"):
    features = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)
    prob = model.predict_proba(scaled_features)

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes (Confidence: {prob[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk (Confidence: {prob[0][0]*100:.2f}%)")
    # python -m streamlit run disease_app.py