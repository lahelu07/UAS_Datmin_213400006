import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    with open('best_decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk memuat dataset
@st.cache_data
def load_dataset():
    return pd.read_csv('DataCoSupplyChainDataset.csv')

# Judul aplikasi
st.title("Aplikasi Prediksi Keterlambatan Pengiriman")

# Deskripsi aplikasi
st.write("Aplikasi ini memprediksi apakah pengiriman akan terlambat berdasarkan input data.")

# Memuat dataset dan model
dataset = load_dataset()
model = load_model()

# Normalisasi nama kolom
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_')
fitur_model = [
    'days_for_shipping_(real)',
    'days_for_shipment_(scheduled)',
    'shipping_mode',
    'customer_segment',
    'order_item_quantity',
    'sales',
    'order_profit_per_order',
    'late_delivery_risk'
]

# Menampilkan nama kolom dalam dataset
st.write("Nama kolom dalam dataset (dengan panjang):")
st.write({col: len(col) for col in dataset.columns})

# Input pengguna berdasarkan fitur yang diperlukan model
st.sidebar.header("Input Data Pengguna")

input_data = {}
for fitur in fitur_model:
    if fitur in dataset.columns:
        if dataset[fitur].dtype == 'object':
            options = dataset[fitur].unique()
            input_data[fitur] = st.sidebar.selectbox(f"{fitur}", options)
        else:
            input_data[fitur] = st.sidebar.number_input(
                f"{fitur}", 
                min_value=float(dataset[fitur].min()), 
                max_value=float(dataset[fitur].max())
            )
    else:
        st.error(f"Kolom '{fitur}' tidak ditemukan dalam dataset. Periksa kembali nama kolom.")

# Konversi input pengguna ke DataFrame
input_df = pd.DataFrame([input_data])

st.write("### Input yang Diberikan:")
st.write(input_df)

# Debugging untuk memastikan kolom input cocok dengan model
if input_df.columns.tolist() != fitur_model:
    st.error("Input data tidak cocok dengan fitur model. Periksa kembali nama kolom.")
else:
    st.write("Kolom input cocok dengan model.")

# Prediksi berdasarkan input pengguna
if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.write("### Hasil Prediksi: Pengiriman akan TERLAMBAT.")
        else:
            st.write("### Hasil Prediksi: Pengiriman TEPAT WAKTU.")

        st.write("Probabilitas Prediksi:")
        st.write(f"Tepat Waktu: {prediction_proba[0]:.2f}, Terlambat: {prediction_proba[1]:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
