import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        with open('best_decision_tree_model(final).pkl', 'rb') as file:
            model = joblib.load(file)
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
    'order_item_quantity',
    'sales',
    'order_profit_per_order'
]

# Sidebar untuk input data pengguna
st.sidebar.header("Masukkan Data untuk Prediksi")
input_data = {}

# Form input untuk masing-masing fitur
input_data['days_for_shipping_real'] = st.sidebar.number_input(
    "Days for Shipping (Real)", min_value=0, max_value=100, step=1, value=10
)
input_data['days_for_shipment_scheduled'] = st.sidebar.number_input(
    "Days for Shipment (Scheduled)", min_value=0, max_value=100, step=1, value=7
)
input_data['order_item_quantity'] = st.sidebar.number_input(
    "Order Item Quantity", min_value=1, max_value=100, step=1, value=2
)
input_data['sales'] = st.sidebar.number_input(
    "Sales", min_value=0.0, max_value=10000.0, step=1.0, value=90.0
)
input_data['order_profit_per_order'] = st.sidebar.number_input(
    "Order Profit Per Order", min_value=-500.0, max_value=500.0, step=1.0, value=500.0
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

        # Konversi hasil prediksi ke dalam hari, jam, dan menit
        total_hours = prediction * 24  # Konversi hari ke jam
        days = int(total_hours // 24)
        hours = int(total_hours % 24)
        minutes = int((total_hours % 1) * 60)

        # Tampilkan hasil prediksi
        st.write(f"### Hasil Prediksi:")
        st.write(f"Pengiriman kemungkinan akan terlambat selama **{days} hari, {hours} jam, dan {minutes} menit**.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
