```python
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------------------------------- #
# 1. Load the trained model – cached to avoid re‑loading on every interaction
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(path: str = "rf_fraud_model.pkl"):
    return joblib.load(path)

model = load_model()

# --------------------------------------------------------------------------- #
# 2. Page configuration and UI layout
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    **Enter transaction details** to predict whether a credit card transaction is **Fraudulent** or **Legitimate**.
    """
)

# Sidebar with a quick model overview
st.sidebar.header("Model Overview")
st.sidebar.write(
    """
    - **Algorithm**: Random Forest Classifier  
    - **Training data**: Credit Card Fraud dataset (scaled)  
    - **Features**: `scaled_amount`, `scaled_time`, `V1`–`V28`  
    """
)

# --------------------------------------------------------------------------- #
# 3. Build an input form that collects all required features
# --------------------------------------------------------------------------- #
def build_input_form() -> tuple[pd.DataFrame | None, bool]:
    """
    Creates a form for user input and returns a DataFrame of the values
    together with a flag indicating if the form was submitted.
    """
    with st.form("transaction_form", clear_on_submit=True):
        st.subheader("Transaction Features")

        # Two columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            scaled_amount = st.number_input(
                "💰 Scaled Amount",
                value=0.0,
                format="%.4f",
                help="Scaled transaction amount (already standardized).",
            )
            scaled_time = st.number_input(
                "🕒 Scaled Time",
                value=0.0,
                format="%.4f",
                help="Scaled time of transaction (already standardized).",
            )
            # V1 – V10
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)
            v10 = st.number_input("V10", value=0.0)

        with col2:
            # V11 – V20
            v11 = st.number_input("V11", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v13 = st.number_input("V13", value=0.0)
            v14 = st.number_input("V14", value=0.0)
            v15 = st.number_input("V15", value=0.0)
            v16 = st.number_input("V16", value=0.0)
            v17 = st.number_input("V17", value=0.0)
            v18 = st.number_input("V18", value=0.0)
            v19 = st.number_input("V19", value=0.0)
            v20 = st.number_input("V20", value=0.0)

            # V21 – V28
            v21 = st.number_input("V21", value=0.0)
            v22 = st.number_input("V22", value=0.0)
            v23 = st.number_input("V23", value=0.0)
            v24 = st.number_input("V24", value=0.0)
            v25 = st.number_input("V25", value=0.0)
            v26 = st.number_input("V26", value=0.0)
            v27 = st.number_input("V27", value=0.0)
            v28 = st.number_input("V28", value=0.0)

        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            data = {
                "scaled_amount": scaled_amount,
                "scaled_time": scaled_time,
                "V1": v1,
                "V2": v2,
                "V3": v3,
                "V4": v4,
                "V5": v5,
                "V6": v6,
                "V7": v7,
                "V8": v8,
                "V9": v9,
                "V10": v10,
                "V11": v11,
                "V12": v12,
                "V13": v13,
                "V14": v14,
                "V15": v15,
                "V16": v16,
                "V17": v17,
                "V18": v18,
                "V19": v19,
                "V20": v20,
                "V21": v21,
                "V22": v22,
                "V23": v23,
                "V24": v24,
                "V25": v25,
                "V26": v26,
                "V27": v27,
                "V28": v28,
            }
            return pd.DataFrame(data, index=[0]), True

    return None, False

input_df, should_predict = build_input_form()

# --------------------------------------------------------------------------- #
# 4. Perform prediction and display the result
# --------------------------------------------------------------------------- #
if should_predict and input_df is not None:
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]  # Probability of fraud
        if pred == 1:
            st.error(f"⚠️ **Fraudulent** transaction detected! Confidence: {proba:.2%}")
        else:
            st.success(f"✅ Transaction is **Legitimate**. Confidence: {(1 - proba):.2%}")
    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")

# --------------------------------------------------------------------------- #
# 5. Show the entered values for transparency
# --------------------------------------------------------------------------- #
if input_df is not None:
    st.write("### Input Features")
    st.dataframe(input_df.style.format("{:.4f}"))
```