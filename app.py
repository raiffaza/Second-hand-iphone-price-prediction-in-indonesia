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
st.title("Prediksi Harga iPhone Second (2021 - 2025)")

available_iphone_models = [
    "iPhone 11", "iPhone 11 Pro", "iPhone 11 Pro Max",
    "iPhone 12", "iPhone 12 Pro", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 Pro", "iPhone 13 Pro Max", "iPhone 13 mini",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
    "iPhone 16", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max"
]

# Map tahun rilis iPhone untuk validasi
iphone_release_year = {
    "iPhone 13": 2021,
    "iPhone 13 Pro": 2021,
    "iPhone 13 Pro Max": 2021,
    "iPhone 13 mini": 2021,
    "iPhone 14": 2022,
    "iPhone 14 Plus": 2022,
    "iPhone 14 Pro": 2022,
    "iPhone 14 Pro Max": 2022,
    "iPhone 15": 2023,
    "iPhone 15 Plus": 2023,
    "iPhone 15 Pro": 2023,
    "iPhone 15 Pro Max": 2023,
    "iPhone 16": 2024,
    "iPhone 16 Plus": 2024,
    "iPhone 16 Pro": 2024,
    "iPhone 16 Pro Max": 2024,
    # Models before 2021 (you can add if needed)
}

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

    # Validasi segera setelah input
    valid = True
    messages = []

    # Validasi tahun_pencatatan minimal 2021
    if tahun_pencatatan < 2021:
        messages.append("❌ Tahun Pencatatan minimal adalah 2021 dan maksimal 2025.")
        valid = False

    # Validasi tahun rilis iPhone terhadap tahun_pencatatan
    release_year = iphone_release_year.get(model_iphone, None)
    if release_year is None:
        # Model iPhone tidak ada tahun rilis, diasumsikan sudah rilis (atau bisa diperluas mappingnya)
        pass
    else:
        if tahun_pencatatan < release_year:
            messages.append(
                f"❌ iPhone {model_iphone} baru dirilis pada tahun {release_year}. "
                "Tidak bisa prediksi untuk tahun sebelum itu."
            )
            valid = False

    # Validasi semua input wajib diisi (selectbox selalu ada pilihan, jadi yg penting tahun_pencatatan)
    # Tambahan: pastikan model_iphone, kapasitas_gb, warna, kondisi dll tidak None atau empty string
    inputs = {
        "Model iPhone": model_iphone,
        "Kapasitas GB": kapasitas_gb,
        "Warna": warna,
        "Kondisi": kondisi,
        "Kategori Pasar": kategori_pasar,
        "Sumber": sumber,
        "Lokasi": lokasi,
    }

    for label, val in inputs.items():
        if val is None or (isinstance(val, str) and val.strip() == ""):
            messages.append(f"❌ {label} harus dipilih.")
            valid = False

    # Tampilkan pesan peringatan jika ada
    if messages:
        for msg in messages:
            st.warning(msg)

    if submitted:
        if not valid:
            st.error("Prediksi tidak dilakukan. Silakan perbaiki input sesuai pesan di atas.")
        else:
            try:
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
