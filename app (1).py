# 💧 Water Quality Prediction App (Streamlit)
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧", layout="centered")
st.title("💧 Real-Time Water Quality Prediction")
st.markdown("Predict whether water is **Safe** or **Unsafe** for drinking based on water quality parameters.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.sidebar.header("Input Parameters")
ph = st.sidebar.slider("pH (0–14)", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness (mg/L)", 0.0, 400.0, 200.0)
solids = st.sidebar.slider("Total Dissolved Solids (ppm)", 0.0, 60000.0, 20000.0, step=100.0)
chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 15.0, 7.0)
sulfate = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 330.0)
conductivity = st.sidebar.slider("Conductivity (μS/cm)", 0.0, 800.0, 400.0)
organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 30.0, 15.0)
trihalomethanes = st.sidebar.slider("Trihalomethanes (μg/L)", 0.0, 125.0, 60.0)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 4.0)

features = np.array([[
    ph, hardness, solids, chloramines, sulfate, conductivity,
    organic_carbon, trihalomethanes, turbidity
]])

features_s = scaler.transform(features)

if st.button("🔍 Predict Potability"):
    proba_safe = float(model.predict_proba(features_s)[0][1]) * 100.0
    pred = int(model.predict(features_s)[0])
    if pred == 1:
        st.success(f"✅ SAFE to drink — confidence: {proba_safe:.1f}%")
    else:
        st.error(f"⚠️ NOT SAFE — safety probability: {proba_safe:.1f}%")

    # Inputs table
    result_df = pd.DataFrame({
        "Parameter": ["pH","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic Carbon","Trihalomethanes","Turbidity"],
        "Value": [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
    })
    st.subheader("Your Input Snapshot")
    st.dataframe(result_df, use_container_width=True)

st.caption("Model: RandomForestClassifier (class_weight='balanced'), inputs standardized with StandardScaler.")
