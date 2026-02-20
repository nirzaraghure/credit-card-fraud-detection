```python
import streamlit as st
import pandas as pd
import joblib
from typing import Tuple, Optional
import numpy as np

# Constants
FEATURES = ['scaled_amount', 'scaled_time', 'V1', 'V2', 'V3', 'V4', 'V5']
DEFAULT_VALUES = {f: 0.0 for f in FEATURES}

# Load the trained model with error handling
try:
    model = joblib.load('rf_fraud_model.pkl')
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Enter transaction details to predict if it's **Fraudulent or Legitimate**.
*All values should be scaled/normalized before input*
""")

def get_input_data() -> pd.DataFrame:
    """Collect and validate user input"""
    cols = st.columns(len(FEATURES))
    data = {}

    for i, (col, default) in enumerate(FEATURES.items()):
        with cols[i]:
            data[col] = st.number_input(
                f"{col.replace('_', ' ').title()}",
                value=default,
                format="%.4f",
                min_value=-1000.0,
                max_value=1000.0
            )

    return pd.DataFrame([data])

def display_prediction(prediction: int, proba: float) -> None:
    """Display prediction result with appropriate styling"""
    if prediction == 1:
        st.error(
            f"""⚠️ **Fraud Detected** ({proba:.2%})
            *Action recommended: Flag for review*""",
            icon="🚨"
        )
    else:
        st.success(
            f"""✅ **Legitimate Transaction** ({1-proba:.2%})""",
            icon="👍"
        )

def main():
    input_data = get_input_data()

    if st.button("🔍 Predict"):
        try:
            if input_data.isna().any().any():
                st.error("❌ Missing values detected. Please fill all fields.")
                return

            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            display_prediction(prediction, proba)

            # Show feature importance for fraudulent cases
            if prediction == 1:
                st.subheader("Key Indicators")
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': FEATURES,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    st.dataframe(
                        feature_importance.style
                        .highlight_max(color='lightgreen')
                        .highlight_min(color='salmon')
                    )

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
```