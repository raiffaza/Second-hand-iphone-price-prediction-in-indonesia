import streamlit as st
import joblib
import pandas as pd

# Load the tuned XGBoost model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model_path = 'xgboost(tuned).pkl'  # Sesuaikan path jika perlu
model = load_model(model_path)
expected_columns = model.feature_names_in_.tolist()

def format_to_ribuan(value):
    return f"{value:,.0f}"

def make_prediction(
    tahun_pencatatan, model_iphone, kapasitas_gb, warna, kondisi, kategori_pasar, sumber, lokasi
):
    data = {col: 0 for col in expected_columns}
    if 'ID' in data:
        data['ID'] = 0
    data['Tahun_Pencatatan'] = tahun_pencatatan

    for model_name in model_iphone:
        col_name = f'Model_iPhone {model_name.strip()}'
        if col_name in data:
            data[col_name] = 1
    for kapasitas in kapasitas_gb:
        col_name = f'Kapasitas_GB_{kapasitas.strip()}'
        if col_name in data:
            data[col_name] = 1
    for color in warna:
        col_name = f'Warna_{color.strip()}'
        if col_name in data:
            data[col_name] = 1
    for cond in kondisi:
        col_name = f'Kondisi_{cond.strip()}'
        if col_name in data:
            data[col_name] = 1
    for category in kategori_pasar:
        col_name = f'Kategori_Pasar_{category.strip()}'
        if col_name in data:
            data[col_name] = 1
    for source in sumber:
        col_name = f'Sumber_{source.strip()}'
        if col_name in data:
            data[col_name] = 1
    for loc in lokasi:
        col_name = f'Lokasi_{loc.strip()}'
        if col_name in data:
            data[col_name] = 1

    input_df = pd.DataFrame([data], columns=expected_columns)
    prediction = model.predict(input_df)[0]
    return format_to_ribuan(prediction)

# --- Streamlit Interface ---
st.title("Prediksi Harga iPhone Second")

available_iphone_models = [
    "iPhone 11", "iPhone 11 Pro", "iPhone 11 Pro Max", 
    "iPhone 12", "iPhone 12 Pro", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 Pro", "iPhone 13 Pro Max",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
    "iPhone 16", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max"
]

available_kapasitas_gb = ["64", "128", "256", "512", "1TB"]

available_warna = [
    "Alpine Green", "Black", "Black Titanium", "Blue", "Blue Titanium", "Deep Purple", "Desert Titanium",
    "Gold", "Graphite", "Gray Titanium", "Green", "Midnight", "Midnight Green", "Natural Titanium",
    "Pacific Blue", "Pink", "Purple", "Red", "Rose Gold", "Sierra Blue", "Silver", "Space Black",
    "Space Gray", "Starlight", "White", "White Titanium", "Yellow"
]

available_kondisi = [
    "Batangan", "Lecet", "Lecet Halus", "Lecet Pemakaian", "Lecet Ringan", "Like New", "Minus BH 80%", "Mulus"
]

available_kategori_pasar = ["Bea Cukai", "Ex-Inter", "Resmi"]

available_sumber = ["Carousell", "Facebook Marketplace", "Forum Jual Beli", "Kaskus", "OLX"]

available_lokasi = ["Bandung", "Batam", "Jakarta", "Makassar", "Medan", "Semarang", "Surabaya", "Yogyakarta"]

with st.form("prediction_form"):
    tahun_pencatatan = st.number_input("Tahun Pencatatan", min_value=2000, max_value=2025, step=1)
    model_iphone = st.selectbox("Model iPhone", options=available_iphone_models)
    kapasitas_gb = st.selectbox("Kapasitas GB", options=available_kapasitas_gb)
    warna = st.selectbox("Warna", options=available_warna)
    kondisi = st.selectbox("Kondisi", options=available_kondisi)
    kategori_pasar = st.selectbox("Kategori Pasar", options=available_kategori_pasar)
    sumber = st.selectbox("Sumber", options=available_sumber)
    lokasi = st.selectbox("Lokasi", options=available_lokasi)
    submitted = st.form_submit_button("Prediksi Harga")

    if submitted:
        try:
            # Bungkus input selectbox menjadi list agar kompatibel ke fungsi
            harga = make_prediction(
                tahun_pencatatan,
                [model_iphone],
                [kapasitas_gb],
                [warna],
                [kondisi],
                [kategori_pasar],
                [sumber],
                [lokasi]
            )
            st.success(f"Prediksi Harga (IDR): {harga}")
        except Exception as e:
            st.error(f"Terjadi error dalam prediksi: {e}")
