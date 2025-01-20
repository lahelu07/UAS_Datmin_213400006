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

def preprocess_data(data):
    # Periksa apakah kolom 'order date (DateOrders)' dan 'shipping date (DateOrders)' ada
    if 'order date (DateOrders)' in data.columns and 'shipping date (DateOrders)' in data.columns:
        # Feature engineering
        data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
        data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
        data['order_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).day
        data['shipping_year'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).year
        data['shipping_month'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).month
        data['shipping_day'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).day

        # Drop unused columns
        data.drop(columns=['order date (DateOrders)', 'shipping date (DateOrders)', 'Category Name'], inplace=True)
    else:
        st.error("Required columns for preprocessing are missing in the dataset.")
        st.stop()

    # One-hot encoding
    data = pd.get_dummies(data)
    return data
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

def preprocess_data(data):
    # Periksa apakah kolom 'order date (DateOrders)' dan 'shipping date (DateOrders)' ada
    if 'order date (DateOrders)' in data.columns and 'shipping date (DateOrders)' in data.columns:
        # Feature engineering
        data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
        data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
        data['order_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).day
        data['shipping_year'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).year
        data['shipping_month'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).month
        data['shipping_day'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).day

        # Drop unused columns
        data.drop(columns=['order date (DateOrders)', 'shipping date (DateOrders)', 'Category Name'], inplace=True)
    else:
        st.error("Required columns for preprocessing are missing in the dataset.")
        st.stop()

    # One-hot encoding
    data = pd.get_dummies(data)
    return data

def create_input_section():
    st.write("### Input Features")
    order_year = st.number_input("Order Year", min_value=2000, max_value=2025, step=1)
    order_month = st.number_input("Order Month", min_value=1, max_value=12, step=1)
    order_day = st.number_input("Order Day", min_value=1, max_value=31, step=1)
    shipping_year = st.number_input("Shipping Year", min_value=2000, max_value=2025, step=1)
    shipping_month = st.number_input("Shipping Month", min_value=1, max_value=12, step=1)
    shipping_day = st.number_input("Shipping Day", min_value=1, max_value=31, step=1)
    demand = st.number_input("Demand (units):", min_value=1, step=1)
    weight = st.number_input("Weight (kg):", min_value=0.1, step=0.1)

    categorical_features = {
        "Product Category": data.filter(like="Product Category").columns.tolist(),
        "Warehouse": data.filter(like="Warehouse").columns.tolist(),
        "Priority": data.filter(like="Priority").columns.tolist()
    }

    input_data = {}

    for feature, options in categorical_features.items():
        selected_option = st.selectbox(f"Select {feature}:", options)
        input_data.update({col: 1 if col == selected_option else 0 for col in options})

    input_data.update({
        "order_year": order_year,
        "order_month": order_month,
        "order_day": order_day,
        "shipping_year": shipping_year,
        "shipping_month": shipping_month,
        "shipping_day": shipping_day,
        "demand": demand,
        "weight": weight
    })

    return input_data

def main():
    st.title("Shipment Time Prediction App")
    st.write("This app predicts shipment delivery times based on provided inputs.")

    model = load_model()
    raw_data = load_data()
    data = preprocess_data(raw_data)

    st.write("Dataset Columns:", raw_data.columns.tolist())

    input_data = create_input_section()

    if st.button("Predict Delivery Time"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Delivery Time: {prediction} days")

if __name__ == "__main__":
    main
st.write("Dataset Columns:", raw_data.columns.tolist())

# Initialize app
st.title("Shipment Time Prediction App")
st.write("This app predicts shipment delivery times based on provided inputs.")

# Load resources
model = load_model()
raw_data = load_data()

# Preprocess dataset
data = preprocess_data(raw_data)

# Input section (based on features in X)
st.write("### Input Features")
order_year = st.number_input("Order Year", min_value=2000, max_value=2025, step=1)
order_month = st.number_input("Order Month", min_value=1, max_value=12, step=1)
order_day = st.number_input("Order Day", min_value=1, max_value=31, step=1)
shipping_year = st.number_input("Shipping Year", min_value=2000, max_value=2025, step=1)
shipping_month = st.number_input("Shipping Month", min_value=1, max_value=12, step=1)
shipping_day = st.number_input("Shipping Day", min_value=1, max_value=31, step=1)
demand = st.number_input("Demand (units):", min_value=1, step=1)
weight = st.number_input("Weight (kg):", min_value=0.1, step=0.1)

# One-hot encoded categorical columns
categorical_features = {
    "Product Category": data.filter(like="Product Category").columns.tolist(),
    "Warehouse": data.filter(like="Warehouse").columns.tolist(),
    "Priority": data.filter(like="Priority").columns.tolist()
}

input_data = {}

for feature, options in categorical_features.items():
    selected_option = st.selectbox(f"Select {feature}:", options)
    input_data.update({col: 1 if col == selected_option else 0 for col in options})

# Add numerical features to input_data
input_data.update({
    "order_year": order_year,
    "order_month": order_month,
    "order_day": order_day,
    "shipping_year": shipping_year,
    "shipping_month": shipping_month,
    "shipping_day": shipping_day,
    "demand": demand,
    "weight": weight
})

# Predict button
if st.button("Predict Delivery Time"):
    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Estimated Delivery Time: {prediction} days")
