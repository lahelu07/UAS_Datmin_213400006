import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        with open('best_decision_tree_model.pkl', 'rb') as file:
            model = pickle.load(file)
        if not hasattr(model, "predict"):
            st.error("File .pkl tidak berisi model prediksi yang valid. Pastikan file adalah model Scikit-learn.")
            return None
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Judul aplikasi
st.title("Aplikasi Prediksi Jumlah Hari Keterlambatan Pengiriman")

# Deskripsi aplikasi
st.write("Aplikasi ini memprediksi jumlah hari keterlambatan pengiriman berdasarkan input data.")

# Muat model
model = load_model()
if model is None:
    st.stop()  # Hentikan aplikasi jika model tidak valid

# Fitur yang dibutuhkan model
fitur_model = [
    'days_for_shipping_real',
    'days_for_shipment_scheduled',
    'shipping_mode',
    'customer_segment',
    'order_item_quantity',
    'sales',
    'order_profit_per_order',
    'late_delivery_risk'
]

# Sidebar untuk input data pengguna
st.sidebar.header("Masukkan Data untuk Prediksi")
input_data = {}

# Form input untuk masing-masing fitur
input_data['days_for_shipping_real'] = st.sidebar.number_input(
    "Days for Shipping (Real)", min_value=0, max_value=100, step=1, value=5
)
input_data['days_for_shipment_scheduled'] = st.sidebar.number_input(
    "Days for Shipment (Scheduled)", min_value=0, max_value=100, step=1, value=3
)
input_data['shipping_mode'] = st.sidebar.selectbox(
    "Shipping Mode", options=["Standard Class", "Second Class", "First Class", "Same Day"]
)
input_data['customer_segment'] = st.sidebar.selectbox(
    "Customer Segment", options=["Consumer", "Corporate", "Home Office"]
)
input_data['order_item_quantity'] = st.sidebar.number_input(
    "Order Item Quantity", min_value=1, max_value=100, step=1, value=1
)
input_data['sales'] = st.sidebar.number_input(
    "Sales", min_value=0.0, max_value=10000.0, step=1.0, value=100.0
)
input_data['order_profit_per_order'] = st.sidebar.number_input(
    "Order Profit Per Order", min_value=-500.0, max_value=500.0, step=1.0, value=10.0
)
input_data['late_delivery_risk'] = st.sidebar.selectbox(
    "Late Delivery Risk", options=[0, 1]
)

# Konversi input pengguna ke DataFrame
input_df = pd.DataFrame([input_data])

st.write("### Input Data yang Diberikan:")
st.write(input_df)

# Prediksi berdasarkan input pengguna
if st.button("Prediksi"):
    try:
        # Lakukan prediksi
        prediction = model.predict(input_df)[0]

        # Tampilkan hasil prediksi
        st.write(f"### Hasil Prediksi: Pengiriman kemungkinan akan terlambat selama **{prediction:.2f} hari**.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
