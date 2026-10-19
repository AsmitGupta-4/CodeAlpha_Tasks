import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset Load
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=names)

# 2. Data Cleaning
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# 3. Features & Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Training (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Accuracy Check
y_pred = rf_model.predict(X_test)
print(f" Random Forest Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 7. Final Saving
joblib.dump(rf_model, 'disease_model.pkl')
joblib.dump(scaler, 'disease_scaler.pkl')
print(" Model and Scaler saved as .pkl files!")
