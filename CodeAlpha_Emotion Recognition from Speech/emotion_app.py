import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import os

# Page Configuration
st.set_page_config(page_title="Speech Emotion Recognizer", page_icon="🎙️")
st.title("🎙️ Emotion Recognition from Speech")
st.write("Upload an audio file (.wav) to detect the human emotion behind it.")

# Load saved assets safely
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('speech_emotion_model.h5')
    classes = joblib.load('emotion_classes.pkl')
    return model, classes

if os.path.exists('speech_emotion_model.h5'):
    model, classes = load_assets()
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])
    
    if uploaded_file is not None:
        # Play audio in app
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Detect Emotion"):
            with st.spinner("Analyzing speech patterns..."):
                # Temporarily save uploaded file to extract features
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract MFCC
                audio, sr = librosa.load("temp.wav", res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfccs_processed = np.mean(mfccs.T, axis=0)
                
                # Reshape for model input
                input_data = np.expand_dims(mfccs_processed, axis=0)
                input_data = np.expand_dims(input_data, axis=2)
                
                # Prediction
                prediction = model.predict(input_data)
                max_index = np.argmax(prediction)
                final_emotion = classes[max_index]
                confidence = prediction[0][max_index] * 100
                
                # Clean up temp file
                if os.path.exists("temp.wav"):
                    os.remove("temp.wav")
                
                # Display Results beautifully
                st.success(f"### Detected Emotion: **{final_emotion.upper()}**")
                st.info(f"💡 Confidence Level: {confidence:.2f}%")
else:
    st.warning("⚠️ Model file not found. Please run 'train_emotion_model.py' first to train the model!")