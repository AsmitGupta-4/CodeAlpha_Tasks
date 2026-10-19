import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Base Directory Setup
BASE_DIR = r"C:\Users\Asus\OneDrive\Desktop\Emotion Recognition from Speech"

print("Data load ho raha hai...")
# Preprocessed data load karna
X = np.load(os.path.join(BASE_DIR, 'X_data.npy'))
y_encoded = np.load(os.path.join(BASE_DIR, 'y_data.npy'))
emotion_list = joblib.load(os.path.join(BASE_DIR, 'emotion_classes.pkl'))

# Categorical labels ko wapas 1D array mein convert karna training ke liye
y = np.argmax(y_encoded, axis=1)

print(f"Dataset Size: {X.shape[0]} samples")
print(f"Total Emotions: {len(emotion_list)} ({', '.join(emotion_list)})")

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Model Training shuru ho rahi hai... Isme thoda time lag sakta hai.")
# Classifier initialize aur train karna
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Training Complete!")

# Evaluate karna
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy * 100:.2f}%")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=emotion_list))

# Trained model ko save karna future prediction ke liye
model_path = os.path.join(BASE_DIR, 'speech_emotion_model.pkl')
joblib.dump(model, model_path)
print(f" Model successfully save ho gaya hai yahan: {model_path}")
