
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning based fraud detection system")

st.divider()

st.subheader("Enter Transaction Details")

# Basic transaction information
time = st.number_input(
    "Transaction Time",
    min_value=0.0,
    value=0.0
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

st.subheader("Transaction Features")

features = []

for i in range(1, 29):
    value = st.number_input(
        f"V{i}",
        value=0.0
    )
    features.append(value)

st.divider()

if st.button("🔍 Check Transaction", use_container_width=True):

    transaction = [time] + features + [amount]

    transaction_df = pd.DataFrame(
        [transaction],
        columns=["Time"] +
                [f"V{i}" for i in range(1, 29)] +
                ["Amount"]
    )

    # Scale input
    transaction_scaled = scaler.transform(transaction_df)

    # Prediction
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )
