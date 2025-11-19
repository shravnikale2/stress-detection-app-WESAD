import streamlit as st
import numpy as np
import pickle

# Load model
with open("best_model_gb.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Stress Detection", layout="wide")

# Sidebar
st.sidebar.title("Sample Inputs")

if st.sidebar.button("Fill Relaxed Sample"):
    st.session_state.mean_ecg = 0.03
    st.session_state.std_ecg = 0.02
    st.session_state.skew_ecg = 0.10
    st.session_state.kurt_ecg = 3.10
    st.session_state.max_ecg = 0.28
    st.session_state.min_ecg = -0.22
    st.session_state.mean_eda = 2.50
    st.session_state.std_eda = 0.04
    st.session_state.mean_temp = 34.50
    st.session_state.std_temp = 0.05

if st.sidebar.button("Fill Stressed Sample"):
    st.session_state.mean_ecg = 0.06
    st.session_state.std_ecg = 0.09
    st.session_state.skew_ecg = 0.65
    st.session_state.kurt_ecg = 4.80
    st.session_state.max_ecg = 0.88
    st.session_state.min_ecg = -0.48
    st.session_state.mean_eda = 6.20
    st.session_state.std_eda = 0.16
    st.session_state.mean_temp = 31.80
    st.session_state.std_temp = 0.07

st.title("🧠 Wearable Stress Detection System")
st.write("Enter physiological features to predict stress using the WESAD dataset ML model.")

# Collect Inputs
col1, col2 = st.columns(2)

with col1:
    mean_ecg = st.number_input("Mean ECG", format="%.5f", key="mean_ecg")
    std_ecg = st.number_input("STD ECG", format="%.5f", key="std_ecg")
    skew_ecg = st.number_input("Skew ECG", format="%.5f", key="skew_ecg")
    kurt_ecg = st.number_input("Kurtosis ECG", format="%.5f", key="kurt_ecg")
    max_ecg = st.number_input("Max ECG", format="%.5f", key="max_ecg")
    min_ecg = st.number_input("Min ECG", format="%.5f", key="min_ecg")

with col2:
    mean_eda = st.number_input("Mean EDA", format="%.5f", key="mean_eda")
    std_eda = st.number_input("STD EDA", format="%.5f", key="std_eda")
    mean_temp = st.number_input("Mean Temperature", format="%.5f", key="mean_temp")
    std_temp = st.number_input("STD Temperature", format="%.5f", key="std_temp")

# Predict Button
if st.button("Predict Stress Level"):
    features = np.array([[mean_ecg, std_ecg, skew_ecg, kurt_ecg,
                          max_ecg, min_ecg, mean_eda, std_eda,
                          mean_temp, std_temp]])

    pred_class = model.predict(features)[0]

    # Probability (if supported)
    try:
        prob = model.predict_proba(features)[0]
        stress_prob = prob[1] if len(prob) > 1 else None
    except:
        stress_prob = None

    if pred_class == 2:
        st.markdown("<h2 style='color:red;'>🔴 PREDICTION: STRESSED</h2>", unsafe_allow_html=True)
        if stress_prob is not None:
            st.write(f"Confidence: **{stress_prob*100:.2f}%**")
    else:
        st.markdown("<h2 style='color:green;'>🟢 PREDICTION: RELAXED</h2>", unsafe_allow_html=True)
        if stress_prob is not None:
            st.write(f"Confidence: **{stress_prob*100:.2f}%**")
