# CodeAlpha_Disease Prediction from Medical Data 

## 📌 Project Overview
This repository features a **Disease Prediction System** developed as part of the CodeAlpha internship. The core focus of this project is to implement predictive health analytics by leveraging historical medical records, patient symptoms, and physiological parameters. The machine learning pipeline accurately classifies and forecasts potential health conditions, acting as a supportive decision-making utility for diagnostic screening.

An elegant, user-friendly interface has been structured using **Streamlit** to enable non-technical users and medical professionals to input clinical test parameters and retrieve real-time automated risk predictions seamlessly.

---

##  Key Features
- **Clinical Data Wrangling:** Complete data processing pipeline featuring missing value imputation, skewness reduction, and feature scaling.
- **Predictive ML Modeling:** Implements robust supervised learning models optimized for disease hazard classification.
- **Instant Risk Analysis:** Evaluates composite health metrics instantly to yield structured diagnostic warnings.
- **Intuitive Web UI:** A secure, clean Streamlit frontend environment allowing frictionless data input.

---

##  Technologies & Libraries Used
- **Programming Language:** Python 3.13
- **Web UI & Framework:** Streamlit
- **Core Data Science & ML:** Scikit-learn, Pandas, NumPy
- **Model Serialization:** Joblib

---

## 📂 Project Structure

- **CodeAlpha_Disease Prediction from Medical Data/** - Main project directory.
- **app.py** - Main Streamlit UI deployment file.
- **disease_prediction_model.pkl** - Pre-trained serialized predictive model file.
- **train_model.py** - Script containing pipeline modeling logic.
- **README.md** - Project Documentation.

---

##  How to Run the Project Locally

### 1. Prerequisites
Install the required infrastructure packages using your terminal environment:
pip install streamlit pandas numpy scikit-learn joblib

2. Launch the Application
Run the interface server within the project root directory:
python -m streamlit run app.py

👤 Developer Details
Developer: Asmit Gupta
