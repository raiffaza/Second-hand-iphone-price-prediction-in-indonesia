import streamlit as st
import joblib
import pandas as pd

# ================== CONFIG & STYLING ====================
st.set_page_config(page_title="Prediksi Harga iPhone Second", page_icon="📱", layout="centered")
st.markdown("""
    <style>
    .block-container {
        padding-top:2rem;
        padding-bottom:2rem;
        padding-left:3rem;
        padding-right:3rem;
    }
    .main {background-color: #f9f9f9;}
    .stButton button {height:3em;width:100%;font-size:1.2em;}
    </style>
    """, unsafe_allow_html=True)

# ================== LOADER ===================
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model_path = 'xgboost(tuned).pkl'
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

# ================== OPTIONS & DATA ======================
available_iphone_models = [
    "iPhone 11", "iPhone 11 Pro", "iPhone 11 Pro Max",
    "iPhone 12", "iPhone 12 Pro", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 Pro", "iPhone 13 Pro Max", "iPhone 13 mini",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
    "iPhone 16", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max"
]
iphone_release_year = {
    "iPhone 13": 2021, "iPhone 13 Pro": 2021, "iPhone 13 Pro Max": 2021, "iPhone 13 mini": 2021,
    "iPhone 14": 2022, "iPhone 14 Plus": 2022, "iPhone 14 Pro": 2022, "iPhone 14 Pro Max": 2022,
    "iPhone 15": 2023, "iPhone 15 Plus": 2023, "iPhone 15 Pro": 2023, "iPhone 15 Pro Max": 2023,
    "iPhone 16": 2024, "iPhone 16 Plus": 2024, "iPhone 16 Pro": 2024, "iPhone 16 Pro Max": 2024,
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

# ================== HEADER & INTRO ======================
st.markdown("<h1 style='text-align:center;'>📱 Prediksi Harga iPhone Second</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:1.1em;'>"
    "Selamat datang di <b>Prediksi Harga iPhone Second</b>! "
    "Web ini membantu Anda memperkirakan harga pasar iPhone bekas (second) berbagai tipe di Indonesia "
    "berdasarkan tahun, model, kondisi, warna, kapasitas, sumber, dan lokasi.<br><br>"
    "<b>Tujuan:</b> Memberikan gambaran harga yang lebih rasional sebelum Anda membeli atau menjual iPhone bekas sehingga keputusan Anda lebih terinformasi."
    "</p>", unsafe_allow_html=True
)
st.divider()

# ================== FORM INPUT ===========================
with st.form("prediction_form"):
    cols = st.columns([1, 1])
    tahun_pencatatan = cols[0].number_input("Tahun Pencatatan", min_value=2021, max_value=2025, value=2021, step=1)

    model_iphone = st.selectbox(
        "Model iPhone", options=[""] + available_iphone_models, index=0,
        help="Pilih model iPhone yang ingin diprediksi."
    )
    kapasitas_gb = st.selectbox(
        "Kapasitas GB", options=[""] + available_kapasitas_gb, index=0,
        help="Pilih varian kapasitas iPhone."
    )
    warna = st.selectbox(
        "Warna", options=[""] + available_warna, index=0,
        help="Pilih warna unit."
    )
    kondisi = st.selectbox(
        "Kondisi", options=[""] + available_kondisi, index=0,
        help="Pilih kondisi unit."
    )
    kategori_pasar = st.selectbox(
        "Kategori Pasar", options=[""] + available_kategori_pasar, index=0,
        help="Pilih kategori pasar."
    )
    sumber = st.selectbox(
        "Sumber", options=[""] + available_sumber, index=0,
        help="Pilih sumber listing data."
    )
    lokasi = st.selectbox(
        "Lokasi", options=[""] + available_lokasi, index=0,
        help="Pilih lokasi."
    )
    st.caption("Pastikan semua input telah dipilih sebelum melakukan prediksi harga.")
    submitted = st.form_submit_button("Prediksi Harga 💰")

    # ================== VALIDATION ======================
    valid = True
    messages = []

    # Validasi tahun_cacatan
    if tahun_pencatatan < 2021:
        messages.append("❌ Tahun pencatatan minimal adalah 2021.")
        valid = False
    # Validasi model iPhone & tahun rilis
    if not model_iphone or model_iphone.strip() == "":
        messages.append("🚫 Silakan pilih model iPhone.")
        valid = False
    elif model_iphone in iphone_release_year:
        rilis = iphone_release_year[model_iphone]
        if tahun_pencatatan < rilis:
            messages.append(
                f"❌ iPhone {model_iphone} baru rilis tahun {rilis}. "
                f"Tidak dapat memprediksi harga sebelum tahun rilis."
            )
            valid = False
    # Validasi kolom lain tidak boleh kosong
    if not kapasitas_gb or kapasitas_gb.strip() == "":
        messages.append("🚫 Kapasitas GB harus dipilih.")
        valid = False
    if not warna or warna.strip() == "":
        messages.append("🚫 Warna harus dipilih.")
        valid = False
    if not kondisi or kondisi.strip() == "":
        messages.append("🚫 Kondisi harus dipilih.")
        valid = False
    if not kategori_pasar or kategori_pasar.strip() == "":
        messages.append("🚫 Kategori pasar harus dipilih.")
        valid = False
    if not sumber or sumber.strip() == "":
        messages.append("🚫 Sumber harus dipilih.")
        valid = False
    if not lokasi or lokasi.strip() == "":
        messages.append("🚫 Lokasi harus dipilih.")
        valid = False

    # Show validation errors direct & disable prediksi
    if messages:
        for msg in messages:
            st.warning(msg)

    # ================== PREDICTION ======================
    if submitted:
        if not valid:
            st.error("Prediksi tidak dapat dilakukan. Silakan lengkapi/benahi data terlebih dahulu.")
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
