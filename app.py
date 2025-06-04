import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_fraud_model.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict if it's **Fraudulent or Legitimate**.")

# Input fields
def user_input():
    scaled_amount = st.number_input("💰 Scaled Transaction Amount", value=0.0, format="%.4f")
    scaled_time = st.number_input("🕒 Scaled Time of Transaction", value=0.0, format="%.4f")
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)

    # Add more Vx if needed, for now keeping top 5 PCA features + scaled ones
    data = {
        'scaled_amount': scaled_amount,
        'scaled_time': scaled_time,
        'V1': v1,
        'V2': v2,
        'V3': v3,
        'V4': v4,
        'V5': v5
    }

    return pd.DataFrame(data, index=[0])

# Get input data
input_data = user_input()

# Predict
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of fraud

        if prediction == 1:
            st.error(f"⚠️ Alert: This transaction is **Fraudulent** (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ This transaction is **Legitimate** (Confidence: {1 - proba:.2f})")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
