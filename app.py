import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("best_model_gb.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Wearable Stress Detection System")
st.write("Enter physiological features to predict stress level.")

# Input fields
mean_ecg = st.number_input("Mean ECG")
std_ecg = st.number_input("STD ECG")
skew_ecg = st.number_input("Skew ECG")
kurt_ecg = st.number_input("Kurtosis ECG")
max_ecg = st.number_input("Max ECG")
min_ecg = st.number_input("Min ECG")

mean_eda = st.number_input("Mean EDA")
std_eda = st.number_input("STD EDA")

mean_temp = st.number_input("Mean Temp")
std_temp = st.number_input("STD Temp")

# Button
if st.button("Predict Stress Level"):
    features = np.array([[mean_ecg, std_ecg, skew_ecg, kurt_ecg,
                          max_ecg, min_ecg, mean_eda, std_eda,
                          mean_temp, std_temp]])

    # Scale the input features
    scaled_features = scaler.transform(features)

    pred = model.predict(scaled_features)[0]

    if pred == 0:
        st.success("Prediction: RELAXED (Label 0)")
    elif pred == 2:
        st.error("Prediction: STRESSED (Label 2)")
    else:
        st.warning(f"Prediction: {pred}")
