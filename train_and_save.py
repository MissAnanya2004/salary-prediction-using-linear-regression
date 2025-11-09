# train_and_save.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- STEP 1: Load your dataset ---
df = pd.read_csv("Salary_Data.csv")


print("✅ Data loaded successfully!\n")
print(df.head())

# --- STEP 2: Explore the data ---
print("\nData Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

sns.pairplot(df)
plt.show()

plt.figure(figsize=(5,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 3: Prepare features and target ---
X = df[['YearsExperience', 'Age']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- STEP 4: Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STEP 5: Define models ---
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

voting_model = VotingRegressor([
    ('Linear', lr),
    ('RF', rf),
    ('GB', gb)
])

# --- STEP 6: Train model ---
voting_model.fit(X_train_scaled, y_train)

# --- STEP 7: Evaluate ---
y_pred = voting_model.predict(X_test_scaled)

print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# --- STEP 8: Visualize actual vs predicted ---
plt.scatter(y_test, y_pred, color='purple')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()

# --- STEP 9: Save model and scaler ---
os.makedirs("models", exist_ok=True)
joblib.dump(voting_model, "models/salary_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\n✅ Model and Scaler saved successfully in 'models' folder.")
