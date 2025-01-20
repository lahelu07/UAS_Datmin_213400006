import streamlit as st
import pandas as pd
import pickle

# Load model and data
MODEL_PATH = 'best_decision_tree_model.pkl'
DATA_PATH = 'DataCoSupplyChainDataset.csv'

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

def load_data():
    try:
        return pd.read_csv(DATA_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(DATA_PATH, encoding='ISO-8859-1')

# Initialize app
st.title("Shipment Time Prediction App")
st.write("This app predicts shipment delivery times for domestic and international customers.")

# Load resources
model = load_model()
data = load_data()

# Input section
shipment_type = st.selectbox("Select Shipment Type:", ["Domestic", "International"])

# Filter relevant data based on shipment type
data_filtered = data[data['Shipment Mode'] == shipment_type]

destination = st.text_input("Destination:", placeholder="Enter destination (e.g., city)")
product_category = st.selectbox("Product Category:", data_filtered['Product Category'].unique())
warehouse = st.selectbox("Warehouse:", data_filtered['Warehouse'].unique())

# Additional input fields
demand = st.number_input("Demand (units):", min_value=1, step=1)
weight = st.number_input("Weight (kg):", min_value=0.1, step=0.1)
priority = st.selectbox("Shipping Priority:", ["Low", "Medium", "High", "Critical"])

# Predict button
if st.button("Predict Delivery Time"):
    # Prepare input features
    input_features = {
        "Shipment Type": shipment_type,
        "Destination": destination,
        "Product Category": product_category,
        "Warehouse": warehouse,
        "Demand": demand,
        "Weight": weight,
        "Priority": priority
    }
    
    # Convert to DataFrame for model input
    input_df = pd.DataFrame([input_features])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Estimated Delivery Time: {prediction} days")

st.write("\n\n**Note**: Ensure that your input values align with the dataset structure for accurate predictions.")
