import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load dataset
df = pd.read_csv("data/medical_insurance_dataset.csv")

# Features and target variable
X = df.drop(columns=["PremiumPrice"])  # Drop the target variable
y = df["PremiumPrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train, y_train)

# Save models
with open("models/linear_regression.pkl", "wb") as f:
    pickle.dump(linear_model, f)

with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("models/xgboost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("Models trained and saved successfully!")
