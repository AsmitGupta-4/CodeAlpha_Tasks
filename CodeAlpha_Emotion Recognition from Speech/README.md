# CodeAlpha_Emotion Recognition from Speech 🎙️🎭

## 📌 Project Overview
This project is a **Speech Emotion Recognition (SER)** system developed during the CodeAlpha internship. The system analysis human speech/audio files (`.wav` format) to detect and classify underlying emotions such as Happy, Sad, Angry, Neutral, Calm, and Fearful. 

It leverages advanced digital signal processing techniques for feature extraction and machine learning pipelines for accurate classification. A sleek, interactive web interface has been built using **Streamlit**, which delivers instant emotional insights alongside corresponding emojis for an enhanced user experience.

---

## 🚀 Key Features
- **Audio Feature Extraction:** Uses the `librosa` library to extract Mel-Frequency Cepstral Coefficients (MFCCs) from raw audio signals.
- **Feature Processing:** Normalizes and shapes audio arrays to match machine learning model configurations.
- **Real-Time Classification:** Features a pre-trained optimized model capable of delivering instantaneous classification outputs.
- **Dynamic Streamlit UI:** Features a streamlined file uploader supporting audio playback and clean, modern results display.

---

## 🛠️ Technologies & Libraries Used
- **Programming Language:** Python 3.13
- **Audio Processing:** Librosa
- **Web Framework / GUI:** Streamlit
- **Machine Learning Stack:** Scikit-learn, NumPy, Joblib

---

## 📂 Project Structure

- **CodeAlpha_Emotion Recognition from Speech/** - Main project directory.
- **app.py** - Streamlit Frontend Web Application with dynamic mapping.
- **train_emotion_model.py** - Architecture training script.
- **speech_emotion_model.pkl** - Pre-trained serialized classifier.
- **emotion_classes.pkl** - Enrolled emotion categorical array mapping.
- **README.md** - Project Documentation.

---

## 💻 How to Run the Project Locally

### 1. Prerequisites
Ensure you have Python installed, then set up the environment using your terminal:
pip install streamlit librosa numpy scikit-learn joblib

2. Launch the Application
Run your clean execution command within the project root folder:
python -m streamlit run app.py

👤 Developer Details
Developer: Asmit Gupta
