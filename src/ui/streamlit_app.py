import os
import json
import streamlit as st
import requests
from dotenv import load_dotenv
from description import FIELD_DESCRIPTIONS

# Load API URL
load_dotenv()
API_URL = os.getenv("PREDICTION_API_URL")

if not API_URL:
    st.error("API URL not found. Please set PREDICTION_API_URL in your .env file.")
    st.stop()

# Load sample payload
sample_payload_path = os.path.join(os.path.dirname(__file__), "sample_payload.json")

if not os.path.exists(sample_payload_path):
    st.error(f"Cannot find sample payload file: {sample_payload_path}")
    st.stop()

with open(sample_payload_path, "r") as f:
    sample_payload = json.dumps(json.load(f), indent=2)

# Tab selection
tab1, tab2 = st.tabs(["📝 Predict by JSON Input", "🔎 Predict by SK_ID_CURR"])

# --------- Tab 1: Predict by editing JSON input ---------
with tab1:
    st.header("Predict by JSON Input")

    with st.expander("📖 Reference: Input Field Descriptions"):
        st.dataframe(FIELD_DESCRIPTIONS)

    user_input = st.text_area(
        "✍️ Paste or edit your prediction input here (format: List[Dict])",
        value=sample_payload,
        height=500,
    )

    if st.button("🚀 Predict from JSON Input"):
        try:
            payload = json.loads(user_input)
            if not isinstance(payload, list):
                raise ValueError("Input must be a list of dictionaries.")
            with st.spinner('Predicting...'):
                response = requests.post(f"{API_URL}/Prediction", json=payload)
                response.raise_for_status()
                st.success("🎯 Prediction Result:")
                st.json(response.json())
        except Exception as e:
            st.error(f"❌ Invalid input or API call failed: {str(e)}")

# --------- Tab 2: Predict by SK_ID_CURR ---------
with tab2:
    st.header("Predict from SK_ID_CURR")
    with st.form("form_predict_by_id"):
        sk_id = st.number_input("Enter SK_ID_CURR", min_value=100000, value=100001, step=1)
        submitted_by_id = st.form_submit_button("Predict by ID")

    if submitted_by_id:
        try:
            with st.spinner('Predicting by ID...'):
                response = requests.post(f"{API_URL}/Prediction-by-id", params={"id": int(sk_id)})
                response.raise_for_status()
                st.success("🎯 Prediction Result:")
                st.json(response.json())
        except Exception as e:
            st.error(f"❌ Error calling prediction-by-id API: {str(e)}")
