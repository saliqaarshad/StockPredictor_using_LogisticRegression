import pandas as pd
import streamlit as st
import joblib
import json

# Load model and exact feature names
rf = joblib.load('model.pkl')
feature_names = joblib.load('features.pkl')

st.title("📈 Stock Price Predictor")
st.write("Enter today's stock details to estimate the closing price.")

# Create inputs dynamically from feature names
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", min_value=0.0, format="%.4f")

# Create DataFrame in correct order
input_df = pd.DataFrame([input_data])[feature_names]

# Predict when button is clicked
if st.button("Predict Closing Price"):
    result = rf.predict(input_df)
    st.success(f"Estimated Closing Price: ${result[0]:.3f}")

# Show accuracy
with open('metrics.json', 'r') as f:
    metrics = json.load(f)
st.write(f"Model Accuracy (R²): {metrics['accuracy']:.4f}")
