import streamlit as st
import pandas as pd
import pickle

# Load the trained model
def load_model():
    with open("best_decision_tree_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load and preprocess dataset for display or feature insights
def preprocess_data(file):
    dataset = pd.read_csv(file, encoding='ISO-8859-1', delimiter=';')
    # Assuming specific preprocessing (e.g., dropping unnecessary columns) is required
    # Adjust this section based on your dataset features and model requirements
    return dataset

# Predict function
def make_prediction(model, input_data):
    return model.predict(input_data)

# Streamlit app
def main():
    st.title("Prediksi Pengiriman")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

    if uploaded_file is not None:
        # Display the dataset
        dataset = preprocess_data(uploaded_file)
        st.write("Dataset yang diunggah:", dataset.head())

        # Allow user to select features for prediction
        st.subheader("Masukkan Data untuk Prediksi")

        # Dynamically create input fields for required features
        feature_inputs = {}
        for feature in dataset.columns:
            feature_inputs[feature] = st.text_input(f"{feature}", "")

        # Convert inputs to DataFrame format
        input_df = pd.DataFrame([feature_inputs])

        # Predict button
        if st.button("Prediksi"):
            model = load_model()
            try:
                prediction = make_prediction(model, input_df)
                st.success(f"Hasil Prediksi: {prediction[0]}")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
