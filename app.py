import streamlit as st
import numpy as np
import pickle

# ------------------ LOAD MODEL ------------------
with open("best_model_gb.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Stress Detection", layout="wide")

# ------------------ SIDEBAR: SAMPLE INPUTS ------------------
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


# ------------------ MAIN UI ------------------
st.title("🧠 Wearable Stress Detection System")

st.write("Enter physiological features to predict stress using the WESAD dataset ML model.")

col1, col2 = st.columns(2)

# Left Column Inputs
with col1:
    mean_ecg = st.number_input("Mean ECG", format="%.5f", key="mean_ecg")
    std_ecg = st.number_input("STD ECG", format="%.5f", key="std_ecg")
    skew_ecg = st.number_input("Skew ECG", format="%.5f", key="skew_ecg")
    kurt_ecg = st.number_input("Kurtosis ECG", format="%.5f", key="kurt_ecg")
    max_ecg = st.number_input("Max ECG", format="%.5f", key="max_ecg")
    min_ecg = st.number_input("Min ECG", format="%.5f", key="min_ecg")

# Right Column Inputs
with col2:
    mean_eda = st.number_input("Mean EDA", format="%.5f", key="mean_eda")
    std_eda = st.number_input("STD EDA", format="%.5f", key="std_eda")
    mean_temp = st.number_input("Mean Temperature", format="%.5f", key="mean_temp")
    std_temp = st.number_input("STD Temperature", format="%.5f", key="std_temp")


# ------------------ PREDICT BUTTON ------------------
if st.button("Predict Stress Level"):
    # Correct feature order
    features = np.array([[mean_ecg, std_ecg, skew_ecg, kurt_ecg,
                          max_ecg, min_ecg, mean_eda, std_eda,
                          mean_temp, std_temp]])

    stressed_prob = None
    pred_class = None

    # ------------ PROBABILITY-BASED FIX (IMPORTANT) ------------
    try:
        probs = model.predict_proba(features)[0]

        # find index of class "2"
        classes = model.classes_
        if 2 in classes:
            idx_2 = list(classes).index(2)
            stressed_prob = probs[idx_2]
        else:
            stressed_prob = probs[-1]  # fallback

        # Threshold fix — THIS MAKES STRESS DETECTION WORK
        THRESHOLD = 0.35  # adjust between 0.30–0.40 if needed
        if stressed_prob >= THRESHOLD:
            pred_class = 2
        else:
            pred_class = 0

    except:
        # If model does not support predict_proba()
        pred_class = int(model.predict(features)[0])

    # ------------------ DISPLAY RESULT ------------------
    if pred_class == 2:
        st.markdown("<h2 style='color:red;'>🔴 PREDICTION: STRESSED</h2>", unsafe_allow_html=True)
        if stressed_prob is not None:
            st.write(f"Confidence (P(stress)) = **{stressed_prob*100:.2f}%**")
    else:
        st.markdown("<h2 style='color:green;'>🟢 PREDICTION: RELAXED</h2>", unsafe_allow_html=True)
        if stressed_prob is not None:
            st.write(f"Confidence (P(stress)) = **{stressed_prob*100:.2f}%**")

