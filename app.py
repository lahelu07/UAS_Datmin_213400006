import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Title and Description
st.title("Shipping Delay Prediction and Analysis App")
st.markdown("This app predicts shipping delays and provides insights into historical delay patterns.")

# File Upload Section
st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load dataset from uploaded file
    try:
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success("Dataset uploaded successfully!")
    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
        st.stop()
else:
    # Default dataset path
    data_path = "DataCoSupplyChainDataset.csv"
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
        st.success("Default dataset loaded successfully!")
    except FileNotFoundError:
        st.error("Default dataset file not found. Please upload a dataset.")
        st.stop()

# Process Dataset
try:
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data['Shipping Date'] = pd.to_datetime(data['Shipping Date'], errors='coerce')
    data['Shipping Delay Days'] = (data['Shipping Date'] - data['Order Date']).dt.days

    # Remove invalid data
    data = data.dropna(subset=['Shipping Delay Days'])
    data = data[data['Shipping Delay Days'] >= 0]
    st.success("Dataset processed successfully!")
except KeyError:
    st.error("Dataset does not contain the required 'Order Date' and 'Shipping Date' columns.")
    st.stop()

# Show Dataset Sample
st.subheader("Dataset Preview")
st.write(data.head())

# Load Model
model_path = 'best_decision_tree_model(updated).pkl'
try:
    model = joblib.load(model_path)
    st.success("Prediction model loaded successfully!")
except FileNotFoundError:
    st.error(f"The model file '{model_path}' was not found. Please upload the model.")
    st.stop()

# Sidebar for Input Parameters
st.sidebar.header("Input Parameters")
order_year = st.sidebar.slider("Order Year", 2000, 2100, 2022)
order_month = st.sidebar.slider("Order Month", 1, 12, 1)
order_day = st.sidebar.slider("Order Day", 1, 31, 1)
shipping_year = st.sidebar.slider("Shipping Year", 2000, 2100, 2022)
shipping_month = st.sidebar.slider("Shipping Month", 1, 12, 1)
shipping_day = st.sidebar.slider("Shipping Day", 1, 31, 1)
feature_1 = st.sidebar.number_input("Feature 1 Value", value=0, step=1)
feature_2 = st.sidebar.number_input("Feature 2 Value", value=0, step=1)

# Validate Dates
try:
    order_date = pd.Timestamp(year=order_year, month=order_month, day=order_day)
    shipping_date = pd.Timestamp(year=shipping_year, month=shipping_month, day=shipping_day)
except ValueError as ve:
    st.error(f"Invalid date input: {ve}")
    st.stop()

if shipping_date < order_date:
    st.error("Shipping date cannot be earlier than order date!")
    st.stop()

# Prepare Input Data for Prediction
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

# Predict Shipping Delay
if st.sidebar.button("Predict"):
    try:
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            st.error("Model feature names are not accessible. Please check your model.")
            st.stop()

        # Ensure input data matches model features
        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data[model_features]

        # Display input data
        st.subheader("Input Data Preview")
        st.dataframe(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        st.success(f"Predicted Shipping Delay: **{prediction[0]} days**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Historical Shipping Delay Analysis
st.subheader("Historical Shipping Delay Analysis")

# Display Descriptive Statistics
st.write("### Delay Statistics")
st.write(data['Shipping Delay Days'].describe())

# Visualize Delay Distribution
st.write("### Delay Distribution")
plt.figure(figsize=(8, 4))
plt.hist(data['Shipping Delay Days'], bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title("Shipping Delay Distribution")
plt.xlabel("Delay (days)")
plt.ylabel("Frequency")
st.pyplot(plt)

# Monthly Delay Trend
st.write("### Monthly Delay Trend")
data['Order Month'] = data['Order Date'].dt.month
delay_by_month = data.groupby('Order Month')['Shipping Delay Days'].mean()
plt.figure(figsize=(8, 4))
delay_by_month.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Average Shipping Delay by Month")
plt.xlabel("Month")
plt.ylabel("Average Delay (days)")
st.pyplot(plt)
