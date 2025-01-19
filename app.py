import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('best_decision_tree_model(1).pkl')
    st.success("Model successfully loaded!")
except FileNotFoundError:
    st.error("The model file 'best_decision_tree_model.pkl' was not found. Please upload the model.")

# App Title
st.title("Shipping Delay Prediction App")

# Form for input data
st.header("Input Features for Prediction")
with st.form("prediction_form"):
    st.write("Enter the following details:")

    # Order date inputs
    order_year = st.number_input("Order Year", min_value=2000, max_value=2100, value=2022)
    order_month = st.number_input("Order Month", min_value=1, max_value=12, value=1)
    order_day = st.number_input("Order Day", min_value=1, max_value=31, value=1)

    # Shipping date inputs
    shipping_year = st.number_input("Shipping Year", min_value=2000, max_value=2100, value=2022)
    shipping_month = st.number_input("Shipping Month", min_value=1, max_value=12, value=1)
    shipping_day = st.number_input("Shipping Day", min_value=1, max_value=31, value=1)

    # Additional features
    feature_1 = st.number_input("Feature 1 Value", value=0)
    feature_2 = st.number_input("Feature 2 Value", value=0)

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button:
    try:
        # Validate date inputs
        try:
            order_date = pd.Timestamp(year=order_year, month=order_month, day=order_day)
            shipping_date = pd.Timestamp(year=shipping_year, month=shipping_month, day=shipping_day)
        except ValueError as ve:
            st.error(f"Invalid date input: {ve}")
            st.stop()

        # Create input dataframe
        input_data = pd.DataFrame({
            'order_year': [order_year],
            'order_month': [order_month],
            'order_day': [order_day],
            'shipping_year': [shipping_year],
            'shipping_month': [shipping_month],
            'shipping_day': [shipping_day],
            'feature_1': [feature_1],
            'feature_2': [feature_2],
        })

        # Predict delay
        prediction = model.predict(input_data)
        st.success(f"Predicted Shipping Delay: **{prediction[0]} days**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
