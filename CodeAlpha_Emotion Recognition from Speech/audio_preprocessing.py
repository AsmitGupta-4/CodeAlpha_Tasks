import os
import numpy as np
import pandas as pd
import joblib
import re

try:
    import librosa
    print("✅ Librosa library successfully import ho gayi hai!")
except ImportError as e:
    print(f"❌ Librosa import nahi ho pa rahi hai! Error: {e}")

# Exact Full Absolute Path
DATA_PATH = r"C:\Users\Asus\OneDrive\Desktop\Emotion Recognition from Speech\dataset"

# RAVDESS Mapping
ravdess_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

print("🔄 Features extraction shuru ho raha hai... Isme thoda time lag sakta hai.")
features = []
labels = []
file_count = 0
success_count = 0

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_count += 1
            file_path = os.path.join(root, file)
            
            # Filename se numbers extract karna
            parts = re.findall(r'\d+', file)
            
            if len(parts) >= 3:
                emotion_code = parts[2]
                if len(emotion_code) == 1:
                    emotion_code = '0' + emotion_code
                
                if emotion_code in ravdess_mapping:
                    emotion_name = ravdess_mapping[emotion_code]
                    
                    try:
                        # FIX: res_type='scipy' use kiya hai taaki resampy par dependency kam ho jaye
                        audio, sr = librosa.load(file_path, res_type='scipy')
                        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                        mfccs_processed = np.mean(mfccs.T, axis=0)
                        
                        features.append(mfccs_processed)
                        labels.append(emotion_name)
                        success_count += 1
                    except Exception as e:
                        print(f"⚠️ File load error ({file}): {e}")

print(f"📊 Scan Complete! Folder mein total files mili: {file_count}")
print(f"🎯 Successfully extracted features for: {success_count} files")

if len(features) > 0:
    X = np.array(features)
    y = pd.get_dummies(labels).values
    emotion_list = pd.get_dummies(labels).columns.tolist()

    output_dir = r"C:\Users\Asus\OneDrive\Desktop\Emotion Recognition from Speech"
    np.save(os.path.join(output_dir, 'X_data.npy'), X)
    np.save(os.path.join(output_dir, 'y_data.npy'), y)
    joblib.dump(emotion_list, os.path.join(output_dir, 'emotion_classes.pkl'))
    print(f"✅ Preprocessing Done! Successfully processed {len(X)} samples.")
else:
    print("❌ Output abhi bhi zero hai. Check karein ki pip install sahi se hua ya nahi.")