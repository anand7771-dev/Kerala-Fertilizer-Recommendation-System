# File: generate_kerala_data.py

import pandas as pd
import numpy as np

# --- Configuration ---
NUM_SAMPLES = 500
FILE_NAME = "fertilizer_data_kerala.csv"

# --- Kerala-Specific Categorical Features ---
soil_types = ["Laterite Soil", "Coastal Alluvium", "Sandy Loam", "Forest Loam", "Red Soil"]
crop_types = ["Rice (Paddy)", "Coconut", "Rubber", "Pepper", "Cardamom", "Cashew", "Banana"]
seasons = ["Virippu (Kharif)", "Mundakan (Rabi)", "Puncha (Summer)"]
fertilizer_names = ["Urea", "DAP", "Potash", "20-20-20", "Factamfos", "18-18-18"]

# --- Create DataFrame ---
df = pd.DataFrame()

# --- Generate random base data with probabilities reflecting Kerala's agriculture ---
df['Soil_Type'] = np.random.choice(soil_types, NUM_SAMPLES, p=[0.5, 0.2, 0.15, 0.1, 0.05]) # Laterite is most common
df['Crop_Type'] = np.random.choice(crop_types, NUM_SAMPLES, p=[0.25, 0.3, 0.15, 0.1, 0.1, 0.05, 0.05]) # Coconut is very common
df['Season'] = np.random.choice(seasons, NUM_SAMPLES)
df['Soil_pH'] = np.random.uniform(4.5, 7.5, NUM_SAMPLES).round(1) # Kerala soils are often acidic
df['Nitrogen'] = np.random.randint(20, 120, NUM_SAMPLES)
df['Phosphorus'] = np.random.randint(10, 80, NUM_SAMPLES)
df['Potassium'] = np.random.randint(20, 150, NUM_SAMPLES)


# --- Define logical rules for target variables tailored to Kerala crops ---

def get_fertilizer(row):
    """Creates a logical fertilizer recommendation for Kerala crops."""
    crop = row['Crop_Type']
    n, p, k, ph = row['Nitrogen'], row['Phosphorus'], row['Potassium'], row['Soil_pH']
    
    if crop == 'Coconut':
        if k < 80: return np.random.choice(["Potash", "18-18-18"], p=[0.8, 0.2])
        if n < 50: return "Urea"
        return "20-20-20"

    if crop == 'Rubber':
        if p < 40 and n < 50: return "Factamfos" # A popular P & N source
        return np.random.choice(["18-18-18", "Urea"])

    if crop == 'Rice (Paddy)':
        if n < 60: return np.random.choice(["Urea", "Factamfos"], p=[0.7, 0.3])
        if p < 40: return "DAP"
        return "Potash"

    if crop in ['Pepper', 'Cardamom']:
        if ph < 5.5 and p < 50: return "DAP" # Address phosphorus lock-up in acidic soil
        return np.random.choice(["20-20-20", "18-18-18", "Potash"])

    # Fallback for other cases
    return np.random.choice(fertilizer_names)

def get_amount(row):
    """Calculates a logical amount, considering higher needs for plantation crops."""
    crop = row['Crop_Type']
    n, p, k = row['Nitrogen'], row['Phosphorus'], row['Potassium']
    
    # Base amount on general nutrient deficiency
    base_amount = 220 - (n + p + k) / 3
    
    # Plantation crops often require higher dosages over a year
    if crop in ['Coconut', 'Rubber', 'Cashew']:
        base_amount *= 1.2
    
    noise = np.random.normal(0, 15)
    amount = base_amount + noise
    
    return max(30, round(amount, 2))


# --- Apply the rules to generate target columns ---
df['Fertilizer_Name'] = df.apply(get_fertilizer, axis=1)
df['Fertilizer_Amount_kg_acre'] = df.apply(get_amount, axis=1)


# --- Save to CSV ---
df.to_csv(FILE_NAME, index=False)

print(f"Successfully generated synthetic dataset for Kerala with {NUM_SAMPLES} samples.")
print(f"File saved as: '{FILE_NAME}'")
print("\nFirst 5 rows of the new dataset:")
print(df.head())