import streamlit as st
import os
import joblib
import numpy as np
import librosa

# Base Directory Setup
BASE_DIR = r"C:\Users\Asus\OneDrive\Desktop\Emotion Recognition from Speech"

# App Title & Styling
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech Emotion Recognition System")
st.write("Apni audio file (.wav) upload kijiye aur dekhiye model kya emotion pehchanta hai!")

# Model aur Classes load karna
@st.cache_resource
def load_project_files():
    model = joblib.load(os.path.join(BASE_DIR, 'speech_emotion_model.pkl'))
    emotion_list = joblib.load(os.path.join(BASE_DIR, 'emotion_classes.pkl'))
    return model, emotion_list

try:
    model, emotion_list = load_project_files()
    st.success("Model and configurations loaded successfully!")
except Exception as e:
    st.error(f"Files load karne mein dikkat aayi: {e}")

# File Uploader GUI Element
uploaded_file = st.file_uploader("Kripya ek .wav file choose karein...", type=["wav"])

if uploaded_file is not None:
    # Play Audio block
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔮 Predict Emotion"):
        with st.spinner("Model audio ko analyze kar raha hai... Please wait..."):
            try:
                # Temporary file save karna librosa ke load karne ke liye
                temp_path = os.path.join(BASE_DIR, "temp_upload.wav")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Feature Extraction
                audio, sr = librosa.load(temp_path, res_type='scipy')
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
                
                # Prediction logic
                raw_prediction = model.predict(mfccs_processed)[0]
                
                # Model numeric index de ya direct text, dono handle karne ke liye fix:
                if isinstance(raw_prediction, (int, np.integer)):
                    predicted_emotion = emotion_list[raw_prediction]
                else:
                    predicted_emotion = str(raw_prediction)
                
                # Cleanup temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Text ko formatting ke liye clean aur uppercase karna
                clean_emotion = str(predicted_emotion).strip().upper()
                
                # Emojis aur English/Hindi Text ki perfect dictionary
                emoji_dict = {
                    'HAPPY': '😊 HAPPY (Khush)', 
                    'SAD': '😢 SAD (Udas)', 
                    'ANGRY': '😡 ANGRY (Gussa)', 
                    'NEUTRAL': '😐 NEUTRAL (Normal)', 
                    'CALM': '😌 CALM (Shant)', 
                    'FEARFUL': '😨 FEARFUL (Dara hua)', 
                    'DISGUST': '🤢 DISGUST (Nafrat)', 
                    'SURPRISED': '😲 SURPRISED (Chounk jana)'
                }
                
                # Output ready karna
                output_text = emoji_dict.get(clean_emotion, f" {clean_emotion}")
                
                # Web screen par display karna
                st.markdown("---")
                st.subheader("Prediction Result:")
                st.info(f"### {output_text}")
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
               
                # python -m streamlit run app.py
