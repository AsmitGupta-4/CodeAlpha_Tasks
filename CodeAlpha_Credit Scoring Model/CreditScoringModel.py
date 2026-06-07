import pandas as pd             # Data manipulation ke liye
import numpy as np              # Numerical calculations ke liye
import matplotlib.pyplot as plt  # Visualization ke liye
import seaborn as sns           # Advanced visualization ke liye

print("Libraries imported successfully!")
# 1. Ek chota sample data banate hain (Agar aapke paas apni CSV nahi hai)
data = {
    'Age': [25, 45, 35, np.nan, 50, 23, 40, 60, 48, 33],
    'MonthlyIncome': [50000, 80000, np.nan, 60000, 120000, 20000, 75000, np.nan, 90000, 55000],
    'DebtRatio': [0.3, 0.5, 0.2, 0.4, 0.8, 0.1, 0.45, 0.6, 0.2, 0.35],
    'Target': [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  # 0 = Safe, 1 = Default (Risk)
}

df = pd.DataFrame(data)

# 2. Data ki pehli 5 rows dekhte hain
print("--- Pehli 5 Rows ---")
print(df.head())

# 3. Check karte hain ki kitni values MISSING hain
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# 4. Basic Statistics (Mean, Max, Min) dekhte hain
print("\n--- Data Summary ---")
print(df.describe())

# 1. Missing values ko Median se bharo
df['Age'] = df['Age'].fillna(df['Age'].median())
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

# 2. Check karo ki ab koi missing value bachi toh nahi
print("--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# 3. Final cleaned data dekho
print("\n--- Cleaned Dataset ---")
print(df) 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Features (X) aur Target (y) alag karein
X = df.drop('Target', axis=1)
y = df['Target']

# 2. Data ko Train aur Test mein divide karein (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Logistic Regression Model banayein
model = LogisticRegression()

# 4. Model ko train karein (Machine ko sikhayein)
model.fit(X_train, y_train)

# 5. Test data par predictions karein
y_pred = model.predict(X_test)

# 6. Results check karein
print("--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred)) 

# 1. Ek naye bande ka data banate hain
# Format: [Age, MonthlyIncome, DebtRatio]
new_customer = pd.DataFrame([[28, 45000, 0.7]], columns=['Age', 'MonthlyIncome', 'DebtRatio'])

# 2. Model se pucho ki ye banda kaisa hai?
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer) # Default hone ka chance kitna hai

print("--- New Customer Analysis ---")
if prediction[0] == 1:
    print("Result: HIGH RISK (Loan Reject hone ke chances hain)")
else:
    print("Result: LOW RISK (Loan Approve ho sakta hai)")

print(f"Confidence: {probability[0][prediction[0]] * 100:.2f}%")
# Debt Ratio vs Target ka graph
plt.figure(figsize=(7,3))
sns.boxplot(x='Target', y='DebtRatio', data=df)
plt.title('Debt Ratio comparison for Safe vs Risk Customers')
plt.show()
from sklearn.ensemble import RandomForestClassifier

# 1. Random Forest Model banayein
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Model ko train karein
rf_model.fit(X_train, y_train)

# 3. Naye customer par check karein (wahi jo pehle banaya tha)
rf_prediction = rf_model.predict(new_customer)
rf_probability = rf_model.predict_proba(new_customer)

print("--- Random Forest Result ---")
if rf_prediction[0] == 1:
    print("Result: HIGH RISK")
else:
    print("Result: LOW RISK")

print(f"Confidence: {rf_probability[0][rf_prediction[0]] * 100:.2f}%")

# 4. Feature Importance dekhein (Kaunsa column sabse zaroori hai?)
importances = rf_model.feature_importances_
for i, val in enumerate(importances):
    print(f"Feature: {X.columns[i]}, Importance: {val:.4f}")
    import joblib

# 1. Model ko save karein (Ek file ban jayegi 'credit_model.pkl')
joblib.dump(rf_model, 'credit_model.pkl')
print("Model saved successfully as 'credit_model.pkl'")

# 2. Future mein jab use karna ho, toh aise load karein:
# loaded_model = joblib.load('credit_model.pkl')
# result = loaded_model.predict(new_customer)