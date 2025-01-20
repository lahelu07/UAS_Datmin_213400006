import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the trained model
st.title("Shipping Delay Prediction App")

try:
    model = joblib.load('best_decision_tree_model(updated).pkl')
    st.success("✅ Model successfully loaded!")
except FileNotFoundError:
    st.error("🚫 The model file 'best_decision_tree_model(updated).pkl' was not found. Please upload the model.")
    st.stop()

# Helper function to validate dates
def validate_dates(order_date, shipping_date):
    if shipping_date < order_date:
        raise ValueError("Shipping date cannot be earlier than the order date!")

# Collect input features from the user
st.header("Input Features for Prediction")
with st.form("prediction_form"):
    st.write("Enter the following details to predict shipping delay:")

    # Date inputs
    order_date_input = st.date_input("Order Date", datetime(2022, 1, 1))
    shipping_date_input = st.date_input("Shipping Date", datetime(2022, 1, 2))

    # Additional features
    feature_1 = st.number_input("Feature 1 (e.g., shipping distance in km)", value=0.0, step=0.1)
    feature_2 = st.number_input("Feature 2 (e.g., product weight in kg)", value=0.0, step=0.1)

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button:
    try:
        # Validate and process dates
        order_date = pd.Timestamp(order_date_input)
        shipping_date = pd.Timestamp(shipping_date_input)
        validate_dates(order_date, shipping_date)

        # Feature engineering
        days_between = (shipping_date - order_date).days

        # Create input dataframe
        input_data = pd.DataFrame({
            'order_year': [order_date.year],
            'order_month': [order_date.month],
            'order_day': [order_date.day],
            'shipping_year': [shipping_date.year],
            'shipping_month': [shipping_date.month],
            'shipping_day': [shipping_date.day],
            'days_between': [days_between],
            'feature_1': [feature_1],
            'feature_2': [feature_2],
        })

        st.write("### Input Data Preview")
        st.dataframe(input_data)

        # Predict delay
        prediction = model.predict(input_data)
        st.success(f"📦 Predicted Shipping Delay: **{prediction[0]} days**")

        # Visualize prediction against historical data
        st.write("### Historical Shipping Delay Distribution")
        historical_delays = np.random.normal(5, 2, 100)  # Replace with real data if available
        plt.figure(figsize=(8, 4))
        plt.hist(historical_delays, bins=20, alpha=0.7, color='blue', edgecolor='black', label='Historical Delays')
        plt.axvline(prediction[0], color='red', linestyle='--', label='Predicted Delay')
        plt.title("Shipping Delay Distribution")
        plt.xlabel("Delay (days)")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)

    except ValueError as ve:
        st.error(f"❌ {ve}")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
