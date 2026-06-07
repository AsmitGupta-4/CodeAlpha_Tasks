import os
import joblib
import numpy as np
import re

try:
    import librosa
    print("✅ Librosa loaded successfully!")
except ImportError:
    print("❌ Librosa missing!")

# Base Directory Setup
BASE_DIR = r"C:\Users\Asus\OneDrive\Desktop\Emotion Recognition from Speech"

def predict_emotion(audio_path):
    # 1. Check if file exists
    if not os.path.exists(audio_path):
        return "❌ File nahi mili! Kripya sahi path check karein."
    
    try:
        # 2. Model, classes aur features configuration load karna
        model = joblib.load(os.path.join(BASE_DIR, 'speech_emotion_model.pkl'))
        emotion_list = joblib.load(os.path.join(BASE_DIR, 'emotion_classes.pkl'))
        
        # 3. Audio file se MFCC features extract karna (Same preprocessing logic)
        audio, sr = librosa.load(audio_path, res_type='scipy')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
        
        # 4. Predict karna
        prediction_index = model.predict(mfccs_processed)[0]
        predicted_emotion = emotion_list[prediction_index]
        
        # Ground truth check karne ke liye filename se asli emotion nikalna (Optional Debug)
        parts = re.findall(r'\d+', os.path.basename(audio_path))
        actual_emotion = "Unknown"
        ravdess_mapping = {'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'}
        if len(parts) >= 3 and parts[2] in ravdess_mapping:
            actual_emotion = ravdess_mapping[parts[2]]

        print(f"\n📁 File Name: {os.path.basename(audio_path)}")
        print(f"🎯 Actual Emotion (Dataset ke mutabik): {actual_emotion.upper()}")
        print(f"🧠 Predicted Emotion (Model ka jawab): {predicted_emotion.upper()}")
        
        if actual_emotion == predicted_emotion:
            print("🎉 Waah! Model ne ekdum sahi pehchana!")
        else:
            print("😅 Model thoda sa chuk gaya, par koshish achhi thi!")
            
    except Exception as e:
        return f"❌ Error aaya prediction ke waqt: {e}"

# --- YAHAN APNI KOI BHI FILE TEST KAREIN ---
# Aap dataset ke andar se kisi bhi file ka path de kar test kar sakte hain
test_file_path = r"C:\Users\Asus\OneDrive\Desktop\Emotion Recognition from Speech\dataset\Actor_01\03-01-03-01-01-01-01.wav"

predict_emotion(test_file_path)