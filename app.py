```python
import pathlib
from typing import Tuple, List, Optional

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_model(path: str = "rf_fraud_model.pkl"):
    """Load and cache a scikit‑learn model from *path*."""
    return joblib.load(path)


def _validate_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """Return a DataFrame with only *required* columns in the given order."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return df[required]


def get_user_input() -> Optional[pd.DataFrame]:
    """Collect transaction details from the user and return a single‑row DataFrame."""
    st.subheader("📊 Transaction Details")
    with st.form("input_form", clear_on_submit=True):
        scaled_amount = st.number_input("💰 Scaled Transaction Amount", value=0.0, format="%.4f")
        scaled_time = st.number_input("🕒 Scaled Time of Transaction", value=0.0, format="%.4f")
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)

        submitted = st.form_submit_button("🔍 Predict")
        if submitted:
            df = pd.DataFrame(
                {
                    "scaled_amount": [scaled_amount],
                    "scaled_time": [scaled_time],
                    "V1": [v1],
                    "V2": [v2],
                    "V3": [v3],
                    "V4": [v4],
                    "V5": [v5],
                }
            )
            return _validate_columns(df, _REQUIRED_COLUMNS)
    return None


def predict(model, data: pd.DataFrame) -> Tuple[int, float]:
    """Return the predicted class and fraud probability."""
    pred = int(model.predict(data)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(data)[0][1])  # probability of class 1 (fraud)
    else:
        # Fallback to decision_function if predict_proba is unavailable
        prob = float(
            1 / (1 + np.exp(-model.decision_function(data)[0]))
        )
    return pred, prob


# --------------------------------------------------------------------------- #
# Streamlit app layout
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS = [
    "scaled_amount",
    "scaled_time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
]

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "Enter transaction details to predict if it's **Fraudulent** or **Legitimate**."
)

# Sidebar for optional model path
model_path = st.sidebar.text_input(
    "Model file path",
    value="rf_fraud_model.pkl",
    help="Path to the pickled model",
)

# Resolve path and load model
try:
    model = load_model(str(pathlib.Path(model_path).expanduser()))
except Exception as exc:
    st.error(f"❌ Could not load model: {exc}")
    st.stop()

# Get user input
input_df = get_user_input()

# Perform prediction if data is available
if input_df is not None:
    with st.spinner("Predicting…"):
        try:
            pred, prob = predict(model, input_df)
            if pred == 1:
                st.error(
                    f"⚠️ **Fraudulent** transaction detected! (Confidence: {prob:.2%})"
                )
            else:
                st.success(
                    f"✅ **Legitimate** transaction confirmed. (Confidence: {(1 - prob):.2%})"
                )
        except Exception as exc:
            st.error(f"❌ Prediction error: {exc}")
```