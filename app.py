import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data_path = "E:\Semester 7\DATMIN\UAS_Datmin_213400006\DataCoSupplyChainDataset.csv"
try:
    data = pd.read_csv(data_path)
    st.success("Dataset loaded successfully!")

    # Create a new column for shipping delay (assume columns 'Order Date' and 'Shipping Date' exist)
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data['Shipping Date'] = pd.to_datetime(data['Shipping Date'], errors='coerce')
    data['Shipping Delay Days'] = (data['Shipping Date'] - data['Order Date']).dt.days
    
    # Filter out invalid or missing data
    data = data.dropna(subset=['Shipping Delay Days'])
    data = data[data['Shipping Delay Days'] >= 0]
except FileNotFoundError:
    st.error(f"Dataset file '{data_path}' was not found. Please upload the file.")
    st.stop()

# Load the trained model
model_path = 'E:\Semester 7\DATMIN\UAS_Datmin_213400006\best_decision_tree_model(updated).pkl'
try:
    model = joblib.load(model_path)
    st.success("Model successfully loaded!")
except FileNotFoundError:
    st.error(f"The model file '{model_path}' was not found. Please upload the model.")
    st.stop()

# App Title
st.title("Shipping Delay Prediction and Analysis App")
st.markdown("This app predicts shipping delays and provides insights into historical delay patterns.")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Order date inputs
order_year = st.sidebar.slider("Order Year", 2000, 2100, 2022)
order_month = st.sidebar.slider("Order Month", 1, 12, 1)
order_day = st.sidebar.slider("Order Day", 1, 31, 1)

# Shipping date inputs
shipping_year = st.sidebar.slider("Shipping Year", 2000, 2100, 2022)
shipping_month = st.sidebar.slider("Shipping Month", 1, 12, 1)
shipping_day = st.sidebar.slider("Shipping Day", 1, 31, 1)

# Additional features
feature_1 = st.sidebar.number_input("Feature 1 Value", value=0, step=1)
feature_2 = st.sidebar.number_input("Feature 2 Value", value=0, step=1)

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

# Prepare input data
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

# Model prediction
if st.sidebar.button("Predict"):
    try:
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            st.error("Model feature names are not accessible. Please check your model.")
            st.stop()

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

# Historical delay analysis
st.subheader("Historical Shipping Delay Analysis")
st.markdown("The following analysis provides insights into historical shipping delays:")

# Display basic stats
st.write("### Delay Statistics")
st.write(data['Shipping Delay Days'].describe())

# Visualize delay distribution
st.write("### Delay Distribution")
plt.figure(figsize=(8, 4))
plt.hist(data['Shipping Delay Days'], bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title("Shipping Delay Distribution")
plt.xlabel("Delay (days)")
plt.ylabel("Frequency")
st.pyplot(plt)

# Monthly delay trend
st.write("### Monthly Delay Trend")
data['Order Month'] = data['Order Date'].dt.month
delay_by_month = data.groupby('Order Month')['Shipping Delay Days'].mean()
plt.figure(figsize=(8, 4))
delay_by_month.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Average Shipping Delay by Month")
plt.xlabel("Month")
plt.ylabel("Average Delay (days)")
st.pyplot(plt)
