import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Page Title
st.title("Shipping Delay Prediction App")
st.write("Prediksi keterlambatan pengiriman berdasarkan data pesanan Anda.")

# Load the trained model
try:
    model = joblib.load('best_decision_tree_model.pkl')
    st.success("✅ Model successfully loaded!")
except FileNotFoundError:
    st.error("🚫 Model file 'best_decision_tree_model.pkl' was not found.")
    st.stop()

# Define the expected features
expected_features = [
    'Benefit per order', 'Category Id', 'Customer Segment_Consumer',
    'Customer Segment_Corporate', 'Customer Segment_Home Office',
    # Add other features here
]

# Collect user input
st.header("Input Features for Prediction")
with st.form("prediction_form"):
    order_date_input = st.date_input("Order Date", datetime(2022, 1, 1))
    shipping_date_input = st.date_input("Shipping Date", datetime(2022, 1, 2))
    benefit_per_order = st.number_input("Benefit per Order", value=0.0, step=0.1)
    category_id = st.selectbox("Category ID", [1, 2, 3, 4])
    customer_segment = st.selectbox("Customer Segment", ['Consumer', 'Corporate', 'Home Office'])
    submit_button = st.form_submit_button("Predict")

if submit_button:
    try:
        # Validate dates
        order_date = pd.Timestamp(order_date_input)
        shipping_date = pd.Timestamp(shipping_date_input)
        if shipping_date < order_date:
            st.error("❌ Shipping date cannot be earlier than the order date!")
            st.stop()

        # One-hot encoding for customer segment
        segment_features = {f'Customer Segment_{segment}': 0 for segment in ['Consumer', 'Corporate', 'Home Office']}
        segment_features[f"Customer Segment_{customer_segment}"] = 1

        # Prepare input data
        input_data = {
            'Benefit per order': [benefit_per_order],
            'Category Id': [category_id],
        }
        input_data.update(segment_features)

        # Fill missing expected features with default values
        for feature in expected_features:
            if feature not in input_data:
                input_data[feature] = [0]

        input_df = pd.DataFrame(input_data)
        st.write("### Input Data Preview")
        st.dataframe(input_df)

        # Predict delay
        prediction = model.predict(input_df)
        st.success(f"📦 Predicted Shipping Delay: **{prediction[0]} days**")

        # Visualization
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
