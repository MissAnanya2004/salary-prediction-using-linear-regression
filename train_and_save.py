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

df = pd.read_csv("Salary_Data.csv")
print("Data loaded.\n")
print(df.head())

print("\nData Info:")
df.info()
print("\nSummary:")
print(df.describe())

sns.pairplot(df)
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df[['YearsExperience', 'Age']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

model = VotingRegressor([
    ("lr", lr),
    ("rf", rf),
    ("gb", gb)
])

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\nEvaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")
plt.show()

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\nModel and scaler saved.")
