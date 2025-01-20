import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
try:
    model = joblib.load('best_decision_tree_model(updated).pkl')
    st.success("Model successfully loaded!")
except FileNotFoundError:
    st.error("The model file 'best_decision_tree_model(updated).pkl' was not found. Please upload the model.")
    st.stop()

# App Title
st.title("Shipping Delay Prediction App")

# Form for input data
st.header("Input Features for Prediction")
with st.form("prediction_form"):
    st.write("Enter the following details:")

    # Order date inputs
    order_year = st.slider("Order Year", 2000, 2100, 2022)
    order_month = st.slider("Order Month", 1, 12, 1)
    order_day = st.slider("Order Day", 1, 31, 1)

    # Shipping date inputs
    shipping_year = st.slider("Shipping Year", 2000, 2100, 2022)
    shipping_month = st.slider("Shipping Month", 1, 12, 1)
    shipping_day = st.slider("Shipping Day", 1, 31, 1)

    # Additional features
    feature_1 = st.number_input("Feature 1 Value", value=0, step=1)
    feature_2 = st.number_input("Feature 2 Value", value=0, step=1)

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

        if shipping_date < order_date:
            st.error("Shipping date cannot be earlier than order date!")
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

        # Get model's expected features
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            st.error("Model feature names are not accessible. Please check your model.")
            st.stop()

        # Add missing features with default values (0)
        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        # Ensure column order matches the model's feature order
        input_data = input_data[model_features]

        # Preview input data
        st.write("### Input Data Preview")
        st.dataframe(input_data)

        # Predict delay
        prediction = model.predict(input_data)
        st.success(f"Predicted Shipping Delay: **{prediction[0]} days**")

        # Visualize historical delay distribution (dummy example)
        st.write("### Historical Delay Distribution")
        delays = np.random.normal(3, 1, 100)  # Example delays
        plt.figure(figsize=(8, 4))
        plt.hist(delays, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(prediction[0], color='red', linestyle='--', label='Prediction')
        plt.title("Delay Distribution")
        plt.xlabel("Delay (days)")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
