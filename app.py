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
    return pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# Judul aplikasi
st.title("Aplikasi Prediksi Keterlambatan Pengiriman")

# Deskripsi aplikasi
st.write("Aplikasi ini memprediksi apakah pengiriman akan terlambat berdasarkan input data.")

# Memuat dataset dan model
dataset = load_dataset()
model = load_model()

# Normalisasi nama kolom dalam dataset
dataset.columns = (
    dataset.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[()]', '', regex=True)
)

# Menampilkan nama kolom dalam dataset
st.write("**Nama kolom dalam dataset (setelah normalisasi):**")
st.write(dataset.columns.tolist())

# Daftar fitur yang diperlukan model (setelah normalisasi)
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

# Periksa apakah semua fitur tersedia dalam dataset
missing_columns = [fitur for fitur in fitur_model if fitur not in dataset.columns]
if missing_columns:
    st.error(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")
    st.stop()  # Hentikan eksekusi jika ada kolom yang hilang

# Input pengguna berdasarkan fitur yang diperlukan model
st.sidebar.header("Input Data Pengguna")

input_data = {}
for fitur in fitur_model:
    if dataset[fitur].dtype == 'object':
        options = dataset[fitur].unique()
        input_data[fitur] = st.sidebar.selectbox(f"{fitur}", options)
    else:
        input_data[fitur] = st.sidebar.number_input(
            f"{fitur}", 
            min_value=float(dataset[fitur].min()), 
            max_value=float(dataset[fitur].max())
        )

# Konversi input pengguna ke DataFrame
input_df = pd.DataFrame([input_data])

st.write("### Input yang Diberikan:")
st.write(input_df)

# Prediksi berdasarkan input pengguna
if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.write("### Hasil Prediksi: Pengiriman akan TERLAMBAT.")
        else:
            st.write("### Hasil Prediksi: Pengiriman TEPAT WAKTU.")

        st.write("**Probabilitas Prediksi:**")
        st.write(f"Tepat Waktu: {prediction_proba[0]:.2f}, Terlambat: {prediction_proba[1]:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
