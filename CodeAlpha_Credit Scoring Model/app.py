import streamlit as st
import pandas as pd
import joblib

# 1. Model load karo jo humne save kiya tha
model = joblib.load('credit_model.pkl')

# 2. Web App ki Heading
st.title("🏦 AI Credit Scoring System")
st.write("Enter customer details to check loan eligibility.")

# 3. User se Input lene ke liye sliders aur boxes
age = st.slider("Select Age", 18, 100, 30)
income = st.number_input("Monthly Income (INR)", min_value=0, value=50000)
debt_ratio = st.slider("Debt Ratio (0.0 to 1.0)", 0.0, 1.0, 0.3)

# 4. Predict Button
if st.button("Check Creditworthiness"):
    # Input data ko model ke format mein convert karo
    input_data = pd.DataFrame([[age, income, debt_ratio]], 
                            columns=['Age', 'MonthlyIncome', 'DebtRatio'])
    
    # Prediction karo
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    # Result dikhao
    st.subheader("Analysis Result:")
    if prediction[0] == 0:
        st.success(f"✅ LOW RISK: Loan can be approved! (Confidence: {probability[0][0]*100:.2f}%)")
    else:
        st.error(f"❌ HIGH RISK: Loan might be rejected! (Confidence: {probability[0][1]*100:.2f}%)")

st.info("Note: This is a demo model for academic purposes.")
# python -m streamlit run app.py