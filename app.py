import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================
st.set_page_config(
    page_title="Prediksi Biodiversitas",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Prediksi Species Richness")
st.write(
    "Aplikasi prediksi kekayaan spesies berdasarkan "
    "faktor lingkungan menggunakan model Random Forest."
)

# ==========================================
# MEMUAT MODEL
# ==========================================
MODEL_PATH = (
    Path(__file__).parent
    / "model_biodiversitas"
    / "random_forest_biodiversitas.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(f"File model tidak ditemukan: {MODEL_PATH}")
    st.stop()

try:
    bundle = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

model = bundle["model"]
features = bundle["features"]
metrics = bundle.get("metrics", {})

st.success("Model Random Forest berhasil dimuat.")

# ==========================================
# INFORMASI MODEL
# ==========================================
st.subheader("Informasi Model")

col1, col2, col3 = st.columns(3)

col1.metric("Algoritma", bundle.get("model_name", "Random Forest"))
col2.metric("Jumlah fitur", len(features))
col3.metric("Target", str(bundle.get("target", "species_richness")))

if metrics:
    st.subheader("Metrik Evaluasi Model")
    cols = st.columns(len(metrics))

    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name.upper(), f"{value:.4f}")

# ==========================================
# INPUT FAKTOR LINGKUNGAN
# ==========================================
st.subheader("Input Faktor Lingkungan")
st.write("Masukkan nilai sesuai rentang dan satuan data pelatihan.")

with st.form("form_prediksi"):
    input_values = {}

    for feature in features:
        if feature == "temperature":
            input_values[feature] = st.number_input(
                "Suhu (temperature)", value=25.0
            )
        elif feature == "precipitation":
            input_values[feature] = st.number_input(
                "Curah hujan (precipitation)", value=5.0,
                min_value=0.0
            )
        elif feature == "humidity":
            input_values[feature] = st.number_input(
                "Kelembapan (humidity)", value=70.0,
                min_value=0.0, max_value=100.0
            )
        elif feature == "wind_speed":
            input_values[feature] = st.number_input(
                "Kecepatan angin (wind_speed)", value=2.0,
                min_value=0.0
            )
        else:
            input_values[feature] = st.number_input(
                feature, value=0.0
            )

    submitted = st.form_submit_button("Prediksi Species Richness")

# ==========================================
# HASIL PREDIKSI
# ==========================================
if submitted:
    input_df = pd.DataFrame([input_values], columns=features)

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Hasil Prediksi")
        st.metric(
            "Prediksi Species Richness",
            f"{prediction:.2f}"
        )

        st.caption(
            "Hasil merupakan estimasi model, bukan jumlah spesies "
            "yang telah diverifikasi melalui survei lapangan."
        )

        with st.expander("Lihat data input"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediksi gagal: {e}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
if hasattr(model, "feature_importances_"):
    st.subheader("Kontribusi Fitur pada Model")

    importance_df = pd.DataFrame({
        "Fitur": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    st.bar_chart(
        importance_df.set_index("Fitur")["Importance"],
        horizontal=True
    )