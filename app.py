```python
import streamlit as st
import pandas as pd
import joblib
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model(path: str = "rf_fraud_model.pkl"):
    """Load and return the trained RandomForest model."""
    return joblib.load(path)

model = load_model()

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
    Enter transaction details to predict whether the transaction is
    **Fraudulent** or **Legitimate**.
    """
)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def build_feature_dict(
    scaled_amount: float,
    scaled_time: float,
    pca_features: List[float],
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Build a feature dictionary from user inputs.

    Parameters
    ----------
    scaled_amount : float
        Scaled transaction amount.
    scaled_time : float
        Scaled time of transaction.
    pca_features : List[float]
        Values for PCA features V1..Vn.
    feature_names : List[str]
        Corresponding feature names for PCA features.

    Returns
    -------
    Dict[str, float]
        Dictionary suitable for DataFrame creation.
    """
    return {
        "scaled_amount": scaled_amount,
        "scaled_time": scaled_time,
        **{name: val for name, val in zip(feature_names, pca_features)},
    }


def predict_transaction(df: pd.DataFrame) -> tuple[int, float]:
    """
    Predict fraud probability for a single transaction.

    Returns
    -------
    prediction : int
        1 for Fraudulent, 0 for Legitimate.
    probability : float
        Probability of fraud.
    """
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return int(pred), float(prob)


# --------------------------------------------------------------------------- #
# Sidebar: Configure number of PCA features
# --------------------------------------------------------------------------- #
st.sidebar.header("Transaction Features")
n_features = st.sidebar.slider(
    "Number of PCA features (V1..Vn)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Select how many PCA components to include.",
)

# --------------------------------------------------------------------------- #
# Main input form
# --------------------------------------------------------------------------- #
with st.form(key="transaction_form"):
    scaled_amount = st.number_input(
        "💰 Scaled Transaction Amount", value=0.0, format="%.4f"
    )
    scaled_time = st.number_input("🕒 Scaled Time of Transaction", value=0.0, format="%.4f")

    # Dynamically create inputs for PCA features
    pca_inputs = []
    for i in range(1, n_features + 1):
        val = st.number_input(f"V{i}", value=0.0)
        pca_inputs.append(val)

    submit_button = st.form_submit_button(label="🔍 Predict")

# --------------------------------------------------------------------------- #
# Prediction logic
# --------------------------------------------------------------------------- #
if submit_button:
    try:
        feature_names = [f"V{i}" for i in range(1, n_features + 1)]
        feature_dict = build_feature_dict(
            scaled_amount, scaled_time, pca_inputs, feature_names
        )
        df_input = pd.DataFrame(feature_dict, index=[0])

        pred, prob = predict_transaction(df_input)

        # Display results
        if pred == 1:
            st.error(
                f"⚠️ **Fraudulent** transaction detected "
                f"(Confidence: {prob * 100:.1f}%)"
            )
        else:
            st.success(
                f"✅ Transaction is **Legitimate** "
                f"(Confidence: {(1 - prob) * 100:.1f}%)"
            )
    except Exception as exc:
        st.error(f"❌ Prediction error: {exc}")
```