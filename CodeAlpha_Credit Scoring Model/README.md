# CodeAlpha_Credit Scoring Model 📊💳

## 📌 Project Overview
This repository contains a **Credit Scoring Model** developed as part of the CodeAlpha internship. The primary objective of this project is to evaluate the creditworthiness of loan applicants and predict potential credit risks using machine learning algorithms. This system helps banks and financial institutions automate lending decisions, minimize default rates, and optimize risk management.

To provide a seamless experience, a clean and interactive web user interface has been built using **Streamlit**, allowing users to input financial parameters and get real-time credit risk predictions instantly.

---

## 🚀 Key Features
- **Data Preprocessing & Cleaning:** Robust handling of missing values, outlier detection, and categorical encoding.
- **Feature Engineering:** Calculation of critical financial indicators like Debt-to-Income ratio, credit history score, and utilization rates.
- **Machine Learning Pipeline:** Optimized classification models (such as Random Forest / Logistic Regression) trained to deliver high accuracy.
- **Interactive UI (Streamlit):** A user-friendly web application where anyone can input data fields and view instant results (Good/Bad/Fair Credit Risk).

---

## 🛠️ Technologies & Libraries Used
- **Programming Language:** Python 3.13
- **Web Framework / GUI:** Streamlit
- **Machine Learning & Data Science:** Scikit-learn, Pandas, NumPy
- **Model Deployment:** Joblib

---

## 📂 Project Structure
text
## 📂 Project Structure

- **CodeAlpha_Credit Scoring Model/** - Main project directory.
- **app.py** - Streamlit Frontend Web Application.
- **train_model.py** - Machine Learning Model Training Script.
- **credit_scoring_model.pkl** - Saved Pre-trained ML Model.
- **dataset/** - Folder containing historical financial data.
- **README.md** - Project Documentation.

💻 How to Run the Project Locally
1. Prerequisites
Ensure you have Python installed on your system. Install the required dependencies using pip:
pip install streamlit pandas numpy scikit-learn joblib

2. Launch the Web Application
Navigate to the project directory in your terminal and run the following command:
python -m streamlit run app.py

🎯 Evaluation & Performance
Primary Algorithm: Random Forest Classifier / Logistic Regression

Prediction Output: Categorizes applicant risk as Low, Medium, or High Risk based on credit data.

Key Features Evaluated: Annual Income, Total Existing Debts, Credit History, Age, and Employment Duration.

👤 Developer Details
Developer: Asmit Gupta


