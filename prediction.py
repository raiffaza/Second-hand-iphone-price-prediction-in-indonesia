import joblib
import pandas as pd

# Load the tuned XGBoost model
model_path = r'F:\Kuliah\Bootcamp\Data Analyst\Iphone Second\xgboost(tuned).pkl'
model = joblib.load(model_path)

# Get the exact feature names and order expected by the model
expected_columns = model.feature_names_in_.tolist()

def format_to_ribuan(value):
    return f"{value:,.0f}"

def make_prediction(
    tahun_pencatatan, model_iphone, kapasitas_gb, warna, kondisi, kategori_pasar, sumber, lokasi
):
    # Inisialisasi semua fitur dengan 0
    data = {col: 0 for col in expected_columns}
    # Tambahkan ID dummy jika ada
    if 'ID' in data:
        data['ID'] = 0
    # Tahun pencatatan
    data['Tahun_Pencatatan'] = tahun_pencatatan

    # Model iPhone (pastikan format OHE: Model_iPhone 15 Pro Max)
    for model_name in model_iphone:
        col_name = f'Model_iPhone {model_name.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Kapasitas GB (pastikan format OHE: Kapasitas_GB_1TB)
    for kapasitas in kapasitas_gb:
        col_name = f'Kapasitas_GB_{kapasitas.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Warna (pastikan format OHE: Warna_Black)
    for color in warna:
        col_name = f'Warna_{color.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Kondisi (pastikan format OHE: Kondisi_Mulus, Kondisi_Like New, dst)
    for cond in kondisi:
        col_name = f'Kondisi_{cond.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Kategori Pasar (OHE: Kategori_Pasar_Resmi, Kategori_Pasar_Ex-Inter, dsb)
    for category in kategori_pasar:
        col_name = f'Kategori_Pasar_{category.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Sumber (OHE: Sumber_OLX, Sumber_Facebook Marketplace, dsb)
    for source in sumber:
        col_name = f'Sumber_{source.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Lokasi (OHE: Lokasi_Jakarta, Lokasi_Bandung, dsb)
    for loc in lokasi:
        col_name = f'Lokasi_{loc.strip()}'
        if col_name in data:
            data[col_name] = 1

    # Konversi ke DataFrame dengan urutan kolom yang benar
    input_df = pd.DataFrame([data], columns=expected_columns)
    # Lakukan prediksi
    prediction = model.predict(input_df)[0]
    return format_to_ribuan(prediction)

# --- User Input Section ---
print("Enter the following details to predict the Harga_IDR:")

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

try:
    tahun_pencatatan = int(input("Tahun Pencatatan (e.g., 2022): "))

    print(f"Available iPhone models: {', '.join(available_iphone_models)}")
    model_iphone = input("Enter the iPhone model(s) (comma separated, e.g., 15 Pro Max): ").split(",")
    model_iphone = [m.strip() for m in model_iphone if m.strip() in available_iphone_models]

    print(f"Available GB capacities: {', '.join(available_kapasitas_gb)}")
    kapasitas_gb = input("Enter the GB capacities (comma separated, e.g., 1TB, 256): ").split(",")
    kapasitas_gb = [k.strip() for k in kapasitas_gb if k.strip() in available_kapasitas_gb]

    print(f"Available colors: {', '.join(available_warna)}")
    warna = input("Enter the color(s) (comma separated, e.g., Black, Gold): ").split(",")
    warna = [w.strip() for w in warna if w.strip() in available_warna]

    print(f"Available conditions: {', '.join(available_kondisi)}")
    kondisi = input("Enter the condition(s) (comma separated, e.g., Mulus, Like New): ").split(",")
    kondisi = [c.strip() for c in kondisi if c.strip() in available_kondisi]

    print(f"Available market categories: {', '.join(available_kategori_pasar)}")
    kategori_pasar = input("Enter the market categories (comma separated, e.g., Resmi): ").split(",")
    kategori_pasar = [cat.strip() for cat in kategori_pasar if cat.strip() in available_kategori_pasar]

    print(f"Available sources: {', '.join(available_sumber)}")
    sumber = input("Enter the sources (comma separated, e.g., OLX): ").split(",")
    sumber = [s.strip() for s in sumber if s.strip() in available_sumber]

    print(f"Available locations: {', '.join(available_lokasi)}")
    lokasi = input("Enter the location(s) (comma separated, e.g., Jakarta): ").split(",")
    lokasi = [l.strip() for l in lokasi if l.strip() in available_lokasi]

    predicted_price = make_prediction(
        tahun_pencatatan, model_iphone, kapasitas_gb, warna, kondisi, kategori_pasar, sumber, lokasi
    )

    print(f"\nPredicted Harga (IDR): {predicted_price}")

except Exception as e:
    print("\nAn error occurred during prediction:")
    print(e)
