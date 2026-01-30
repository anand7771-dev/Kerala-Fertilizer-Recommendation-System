# File: app_kerala.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

# -------------------------
# --- KERALA SPECIFIC ---
# Load the Kerala-specific Fertilizer System
# -------------------------
try:
    with open("fertilizer_system_kerala.pkl", "rb") as f:
        system = pickle.load(f)
    clf_model = system["clf_model"]
    reg_model = system["reg_model"]
    scaler = system["scaler"]
    target_le = system["target_encoder"]
except FileNotFoundError:
    st.error("Model file 'fertilizer_system_kerala.pkl' not found! Please run `train_kerala_model.py` first.")
    st.stop()


# -----------------------------------
# --- KERALA SPECIFIC ---
# Updated helper dictionary for application instructions
# -----------------------------------
instructions = {
    "Urea": "For crops like Paddy, apply in 2-3 split doses. Avoid application during heavy rain to prevent runoff.",
    "DAP": "Best applied as a basal dose at the time of sowing. Good for pulses and cereals.",
    "Potash": "Crucial for plantation crops like Coconut and Banana. For sandy soils, apply in split doses. For coconut, apply in a circular basin around the trunk.",
    "20-20-20": "A balanced NPK fertilizer. Can be applied via fertigation or as a foliar spray for horticultural crops.",
    "Factamfos": "An excellent source of Nitrogen and Phosphorus, with added Sulphur. Apply as a basal dose for crops like rubber and rice.",
    "18-18-18": "A balanced water-soluble fertilizer, ideal for fertigation in spice and fruit crops.",
    "N/A": "No specific instructions."
}


# -------------------------
# App Title and Configuration
# -------------------------
st.set_page_config(page_title="Kerala Fertilizer Recommender", page_icon="🥥", layout="wide")
# --- KERALA SPECIFIC ---
st.title("🥥 Kerala Fertilizer Recommendation System")
st.write(f"An intelligent tool for farmers in Kerala, providing tailored recommendations for local crops and soils. (Data as of: {datetime.now().strftime('%d %b %Y')})")


# -------------------------
# --- KERALA SPECIFIC ---
# Load persistent history from a separate file
# -------------------------
history_file = "history_kerala.csv"

if "history" not in st.session_state:
    if os.path.exists(history_file):
        st.session_state.history = pd.read_csv(history_file).to_dict("records")
    else:
        st.session_state.history = []

# -------------------------
# Input fields with Kerala context
# -------------------------
st.header("1. Enter Your Farm's Details")
col1, col2 = st.columns(2)

with col1:
    # --- KERALA SPECIFIC ---
    soil_type = st.selectbox("Soil Type", ["Laterite Soil", "Coastal Alluvium", "Sandy Loam", "Forest Loam", "Red Soil"])
    crop_type = st.selectbox("Crop Type", ["Rice (Paddy)", "Coconut", "Rubber", "Pepper", "Cardamom", "Cashew", "Banana"])
    season = st.selectbox("Season", ["Virippu (Kharif)", "Mundakan (Rabi)", "Puncha (Summer)"])

with col2:
    N = st.number_input("Nitrogen (N) content (kg/ha)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P) content (kg/ha)", min_value=0, max_value=200, value=40)
    K = st.number_input("Potassium (K) content (kg/ha)", min_value=0, max_value=200, value=60)

soil_ph = st.slider("Soil pH Level", min_value=4.0, max_value=9.0, value=6.0, step=0.1, help="Kerala soils are often acidic. Enter the correct pH for a better recommendation.")

# -------------------------
# --- KERALA SPECIFIC ---
# Encoding (must exactly match the training script for Kerala)
# -------------------------
soil_dict = {"Laterite Soil": 0, "Coastal Alluvium": 1, "Sandy Loam": 2, "Forest Loam": 3, "Red Soil": 4}
crop_dict = {"Rice (Paddy)": 0, "Coconut": 1, "Rubber": 2, "Pepper": 3, "Cardamom": 4, "Cashew": 5, "Banana": 6}
season_dict = {"Virippu (Kharif)": 0, "Mundakan (Rabi)": 1, "Puncha (Summer)": 2}

features = [soil_dict[soil_type], crop_dict[crop_type], season_dict[season], soil_ph, N, P, K]

# -------------------------
# Prediction Trigger
# -------------------------
st.header("2. Get Your Recommendation")
if st.button("🌿 Get Recommendation", type="primary"):
    features_scaled = scaler.transform([features])
    fert_result, amount_result = None, None
    
    st.markdown("---")
    st.header("✅ Your Personalized Recommendation")

    # Combined prediction logic
    fert_pred = clf_model.predict(features_scaled)
    fert_result = target_le.inverse_transform(fert_pred)[0]
    st.success(f"**Recommended Fertilizer:** {fert_result}")

    st.markdown("#### 📝 Application Instructions")
    st.info(instructions.get(fert_result, "No specific instructions available."))
    
    amount_pred = reg_model.predict(features_scaled)[0]
    amount_result = round(amount_pred, 2)
    st.success(f"**Recommended Amount:** {amount_result} kg/acre")

    # Cost Analysis
    with st.expander("💰 Calculate Estimated Cost"):
        price_per_kg = st.number_input("Enter market price per kg (₹)", min_value=0.0, value=30.0, step=0.5)
        total_cost = round(amount_result * price_per_kg, 2)
        st.write(f"#### Estimated Total Cost for one acre: **₹ {total_cost:,.2f}**")

    # Save result to history
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Soil Type": soil_type, "Crop Type": crop_type, "Season": season, "Soil pH": soil_ph,
        "N": N, "P": P, "K": K,
        "Recommended Fertilizer": fert_result, "Recommended Amount (kg/acre)": amount_result
    }
    st.session_state.history.insert(0, record)
    pd.DataFrame(st.session_state.history).to_csv(history_file, index=False)
    st.balloons()


# -------------------------
# History and Insights Section (No changes needed here, it adapts automatically)
# -------------------------
if st.session_state.history:
    st.markdown("---")
    st.header("📊 Prediction History & Insights")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fert_counts = history_df["Recommended Fertilizer"].value_counts().reset_index()
        fert_counts.columns = ["Fertilizer", "Count"]
        fig1 = px.pie(fert_counts, names="Fertilizer", values="Count", title="Distribution of Recommendations", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        avg_amounts = history_df.groupby("Crop Type")["Recommended Amount (kg/acre)"].mean().reset_index()
        fig2 = px.bar(avg_amounts, x="Crop Type", y="Recommended Amount (kg/acre)", title="Avg. Recommended Amount by Crop", color="Crop Type")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Download and Clear History
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.download_button(label="📥 Download History", data=history_df.to_csv(index=False).encode("utf-8"), file_name="kerala_fertilizer_history.csv", mime="text/csv")
    with col2:
        if st.button("🗑️ Clear All History"):
            st.session_state.history = []
            if os.path.exists(history_file): os.remove(history_file)
            st.rerun()