# 💧 Water Quality Prediction App (Streamlit) — Resilient Load
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ML imports inside functions to avoid import cost if just showing error
def _train_and_save_from_csv(csv_path: str, model_path: str, scaler_path: str):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier

    data = pd.read_csv(csv_path)
    imputer = SimpleImputer(strategy="median")
    data[data.columns] = imputer.fit_transform(data)

    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧", layout="centered")
st.title("💧 Real-Time Water Quality Prediction")

MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
CSV_PATH = "water_potability.csv"

@st.cache_resource
def load_artifacts():
    # 1) Try to load existing pickles
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, "loaded_pickles"
    except Exception as e:
        # 2) If loading fails, try to retrain from CSV (if present)
        if Path(CSV_PATH).exists():
            model, scaler = _train_and_save_from_csv(CSV_PATH, MODEL_PATH, SCALER_PATH)
            return model, scaler, "retrained_from_csv"
        else:
            raise RuntimeError(
                f"Failed to load pickles ({e.__class__.__name__}). "
                "No CSV found to retrain. Please add water_potability.csv or re-upload compatible pickles."
            )

model, scaler, source = load_artifacts()
st.caption(f"Artifacts source: **{source}**")

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

with st.expander("⚙️ Environment Info"):
    import sys, sklearn, numpy, joblib as _jb
    st.write({
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "joblib": _jb.__version__,
    })
st.caption("If pickles fail to load due to version mismatch, the app retrains from CSV and resaves compatible pickles.")
