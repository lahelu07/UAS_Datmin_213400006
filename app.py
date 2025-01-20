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
    # Daftar kolom yang harus ada
    required_columns = ['order date (DateOrders)', 'shipping date (DateOrders)', 'Category Name']
    
    # Periksa keberadaan kolom
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"The following required columns are missing in the dataset: {missing_columns}")
        st.stop()

    # Pastikan kolom sesuai dengan dataset
    date_order_col = 'order date (DateOrders)' if 'order date (DateOrders)' in data.columns else 'Order Date'
    shipping_date_col = 'shipping date (DateOrders)' if 'shipping date (DateOrders)' in data.columns else 'Shipping Date'

    # Feature engineering
    data['order_year'] = pd.DatetimeIndex(data[date_order_col]).year
    data['order_month'] = pd.DatetimeIndex(data[date_order_col]).month
    data['order_day'] = pd.DatetimeIndex(data[date_order_col]).day
    data['shipping_year'] = pd.DatetimeIndex(data[shipping_date_col]).year
    data['shipping_month'] = pd.DatetimeIndex(data[shipping_date_col]).month
    data['shipping_day'] = pd.DatetimeIndex(data[shipping_date_col]).day

    # Drop unused columns jika ada
    columns_to_drop = [date_order_col, shipping_date_col, 'Category Name']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    # One-hot encoding
    data = pd.get_dummies(data)
    return data

def main():
    # Judul aplikasi
    st.title("Shipment Time Prediction App")
    st.write("This app predicts shipment delivery times based on provided inputs.")

    # Load model dan data
    model = load_model()
    raw_data = load_data()

    # Debugging: Menampilkan kolom dataset
    st.write("Dataset Columns:", raw_data.columns.tolist())

    # Preprocess dataset
    data = preprocess_data(raw_data)
    st.write("Preprocessed Dataset Columns:", data.columns.tolist())

    # Input section (berdasarkan fitur dalam dataset)
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

    # Tambahkan fitur numerik ke input_data
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
        # Konversi input_data ke DataFrame
        input_df = pd.DataFrame([input_data])

        # Prediksi
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Delivery Time: {prediction} days")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
