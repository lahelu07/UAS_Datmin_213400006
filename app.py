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

# Define the expected features
expected_features = [
    'Benefit per order', 'Category Id', 'Customer Segment_Consumer',
    'Customer Segment_Corporate', 'Customer Segment_Home Office', 
    # ... add all other features used during training
]

# Collect input features from the user
st.header("Input Features for Prediction")
with st.form("prediction_form"):
    st.write("Enter the following details to predict shipping delay:")

    # Date inputs
    order_date_input = st.date_input("Order Date", datetime(2022, 1, 1))
    shipping_date_input = st.date_input("Shipping Date", datetime(2022, 1, 2))

    # Additional inputs (replace with actual expected features)
    benefit_per_order = st.number_input("Benefit per Order", value=0.0, step=0.1)
    category_id = st.selectbox("Category ID", [1, 2, 3, 4])  # Example categories
    customer_segment = st.selectbox(
        "Customer Segment", 
        ['Consumer', 'Corporate', 'Home Office']
    )

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button:
    try:
        # Validate and process dates
        order_date = pd.Timestamp(order_date_input)
        shipping_date = pd.Timestamp(shipping_date_input)
        if shipping_date < order_date:
            st.error("❌ Shipping date cannot be earlier than the order date!")
            st.stop()

        # One-hot encoding for customer segment
        segment_features = {
            'Customer Segment_Consumer': 0,
            'Customer Segment_Corporate': 0,
            'Customer Segment_Home Office': 0,
        }
        segment_features[f"Customer Segment_{customer_segment}"] = 1

        # Create input dataframe
        input_data = {
            'Benefit per order': [benefit_per_order],
            'Category Id': [category_id],
        }
        input_data.update(segment_features)  # Add one-hot encoded segment features

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in input_data:
                input_data[feature] = [0]  # Default value for missing features

        input_df = pd.DataFrame(input_data)
        st.write("### Input Data Preview")
        st.dataframe(input_df)

        # Predict delay
        prediction = model.predict(input_df)
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

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
