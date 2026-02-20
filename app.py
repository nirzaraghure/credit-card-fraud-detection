```python
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_fraud_model.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection")

def get_user_input():
    data = {
        'scaled_amount': st.number_input("Scaled Transaction Amount", value=0.0, format="%.4f"),
        'scaled_time': st.number_input("Scaled Time of Transaction", value=0.0, format="%.4f"),
        *st.columns(5),
        'V1': st.number_input("V1", value=0.0),
        'V2': st.number_input("V2", value=0.0),
        'V3': st.number_input("V3", value=0.0),
        'V4': st.number_input("V4", value=0.0),
        'V5': st.number_input("V5", value=0.0)
    }
    
    df = pd.DataFrame(data, index=[0])
    return df

input_data = get_user_input()

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of fraud
        
        if prediction == 1:
            st.error(f"Alert: This transaction is Fraudulent (Confidence: {proba:.2f})")
        else:
            st.success(f"This transaction is Legitimate (Confidence: {(1 - proba):.2f})")
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
```