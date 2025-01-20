import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# Sidebar for dataset upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        # Attempt to load dataset with ISO-8859-1 encoding
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success("Dataset loaded successfully with ISO-8859-1 encoding.")

        # Display dataset columns
        st.write("### Dataset Columns")
        st.write(data.columns.tolist())

        # Identify relevant columns
        order_date_column = None
        shipping_date_column = None
        for col in data.columns:
            if 'order' in col.lower() and 'date' in col.lower():
                order_date_column = col
            if 'shipping' in col.lower() and 'date' in col.lower():
                shipping_date_column = col

        # Validate columns
        if order_date_column is None or shipping_date_column is None:
            st.error("Dataset does not contain the required 'Order Date' and 'Shipping Date' columns.")
            st.stop()

        # Rename columns for consistency
        data.rename(columns={
            order_date_column: 'Order Date',
            shipping_date_column: 'Shipping Date'
        }, inplace=True)
        st.success("Date columns successfully identified and renamed!")

        # Parse dates
        data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
        data['Shipping Date'] = pd.to_datetime(data['Shipping Date'], errors='coerce')

        # Drop rows with invalid dates
        data.dropna(subset=['Order Date', 'Shipping Date'], inplace=True)

        # Calculate shipping delay
        data['Delay'] = (data['Shipping Date'] - data['Order Date']).dt.days
        st.write("### Dataset with Calculated Delay")
        st.dataframe(data[['Order Date', 'Shipping Date', 'Delay']])

        # Visualize delay distribution
        st.write("### Delay Distribution")
        plt.figure(figsize=(8, 4))
        plt.hist(data['Delay'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title("Shipping Delay Distribution")
        plt.xlabel("Delay (days)")
        plt.ylabel("Frequency")
        st.pyplot(plt)

        # Make predictions for the dataset
        if st.button("Predict Delays for Dataset"):
            try:
                # Ensure dataset contains necessary features for the model
                model_features = getattr(model, "feature_names_in_", None)
                if model_features is None:
                    st.error("Model feature names are not accessible. Please check your model.")
                    st.stop()

                for feature in model_features:
                    if feature not in data.columns:
                        data[feature] = 0

                input_data = data[model_features]

                # Predict delays
                predictions = model.predict(input_data)
                data['Predicted Delay'] = predictions
                st.write("### Dataset with Predicted Delays")
                st.dataframe(data[['Order Date', 'Shipping Date', 'Delay', 'Predicted Delay']])

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
        st.stop()

# Form for manual input and prediction
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

if submit_button:
    try:
        # Validate and create date inputs
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

        # Ensure column order matches the model's features
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            st.error("Model feature names are not accessible. Please check your model.")
            st.stop()

        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data[model_features]

        # Predict delay
        prediction = model.predict(input_data)
        st.success(f"Predicted Shipping Delay: **{prediction[0]} days**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
