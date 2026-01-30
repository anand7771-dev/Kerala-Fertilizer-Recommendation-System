# File: train_kerala_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

# --- 1. Load Data ---
# --- MODIFIED FOR KERALA ---
print("Loading the Kerala-specific dataset...")
df = pd.read_csv("fertilizer_data_kerala.csv")

# --- 2. Preprocessing ---
print("Preprocessing the data for Kerala...")

# --- MODIFIED FOR KERALA ---
# Manual encoding for input features to match the new Kerala data
soil_map = {"Laterite Soil": 0, "Coastal Alluvium": 1, "Sandy Loam": 2, "Forest Loam": 3, "Red Soil": 4}
crop_map = {"Rice (Paddy)": 0, "Coconut": 1, "Rubber": 2, "Pepper": 3, "Cardamom": 4, "Cashew": 5, "Banana": 6}
season_map = {"Virippu (Kharif)": 0, "Mundakan (Rabi)": 1, "Puncha (Summer)": 2}

df['Soil_Type'] = df['Soil_Type'].map(soil_map)
df['Crop_Type'] = df['Crop_Type'].map(crop_map)
df['Season'] = df['Season'].map(season_map)

# This part remains the same
target_encoder = LabelEncoder()
df['Fertilizer_Name_Encoded'] = target_encoder.fit_transform(df['Fertilizer_Name'])

# Drop rows with NaN values that might result from mapping if a category is missing
df.dropna(inplace=True)


# --- 3. Define Features (X) and Targets (y) ---
# The feature names are the same, so no change here
X = df[['Soil_Type', 'Crop_Type', 'Season', 'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium']]
y_clf = df['Fertilizer_Name_Encoded']
y_reg = df['Fertilizer_Amount_kg_acre']


# --- 4. Split Data ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
)


# --- 5. Scale Numerical Features ---
# The logic is the same, but the scaler will learn from Kerala's data distribution
print("Scaling numerical features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 6. Train Models ---
# The models are the same, but they will learn different patterns from the new data
print("Training the classification and regression models on Kerala data...")
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_scaled, y_train_clf)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_scaled, y_train_reg)
print("Models trained successfully.")


# --- 7. Evaluate Models ---
# This will show how well the models learned from the Kerala dataset
print("\n--- Model Evaluation (Kerala) ---")
y_pred_clf = clf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Classification Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=target_encoder.classes_, zero_division=0))

y_pred_reg = reg_model.predict(X_test_scaled)
r2 = r2_score(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"\nRegression R-squared (R2) Score: {r2:.4f}")
print(f"Regression Mean Absolute Error (MAE): {mae:.4f}")


# --- 8. Save the Entire System ---
# --- MODIFIED FOR KERALA ---
# Saving to a new file to keep it separate from the original model
print("\nSaving the complete Kerala system to 'fertilizer_system_kerala.pkl'...")
system_kerala = {
    "clf_model": clf_model,
    "reg_model": reg_model,
    "scaler": scaler,
    "target_encoder": target_encoder
}

with open("fertilizer_system_kerala.pkl", "wb") as f:
    pickle.dump(system_kerala, f)

print("Kerala-specific system saved successfully.")