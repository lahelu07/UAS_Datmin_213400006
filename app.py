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

# Define the full list of features used during training
expected_features = [
    'Benefit per order', 'Category Id', 'Customer Zipcode', 
    'Delivery Status_Late delivery', 
    'Delivery Status_Advance shipping', 
    'Delivery Status_Shipping canceled', 
    'Delivery Status_Shipping on time',
    'Customer Segment_Consumer', 'Customer Segment_Corporate', 'Customer Segment_Home Office',
    'Department Name_Apparel', 'Department Name_Book Shop', 'Department Name_Discs Shop',
    'Department Name_Fan Shop', 'Department Name_Fitness',
]

# Collect input features from the user
st.header("Input Features for Prediction")
with st.form("prediction_form"):
    st.write("Enter the following details to predict shipping delay:")

    # Date inputs
    order_date_input = st.date_input("Order Date", datetime(2022, 1, 1))
    shipping_date_input = st.date_input("Shipping Date", datetime(2022, 1, 2))

    # Additional inputs
    benefit_per_order = st.number_input("Benefit per Order", value=0.0, step=0.1)
    category_id = st.selectbox("Category ID", [1, 2, 3, 4])  # Example categories

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

        # Calculate delay in days
        delay_days = (shipping_date - order_date).days

        # Create input dictionary
        input_data = {
            'Benefit per order': [benefit_per_order],
            'Category Id': [category_id],
            'Customer Zipcode': [0],  # Replace 0 with actual data if required
            'Delivery Status_Late delivery': [1 if delay_days > 0 else 0],
            'Delivery Status_Advance shipping': [1 if delay_days < 0 else 0],
            'Delivery Status_Shipping canceled': [0],  # Assuming no canceled shipping
            'Delivery Status_Shipping on time': [1 if delay_days == 0 else 0],
        }

        # Add missing features with default values (0)
        for feature in expected_features:
            if feature not in input_data:
                input_data[feature] = [0]  # Default value for missing features

        # Create input dataframe
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
