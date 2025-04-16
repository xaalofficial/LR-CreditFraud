import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("💳 Credit Card Fraud Detector")

st.write("Enter transaction details below:")

v1 = st.slider('V1', -50.0, 50.0, 0.0)
v2 = st.slider('V2', -50.0, 50.0, 0.0)
v3 = st.slider('V3', -50.0, 50.0, 0.0)
amount = st.number_input('Amount', min_value=0.0, step=1.0)

if st.button('Check for Fraud'):
    features = np.array([[v1, v2, v3, amount]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Looks Legit.")
