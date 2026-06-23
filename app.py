import streamlit as st
import cv2
import numpy as np
from PIL import Image

from backend.face_detection import FaceDetection
from backend.face_alignment import align_face
from backend.preprocess import preprocess_image
from backend.pca_utils import load_pca_model, transform_vector
from backend.similarity import compare_faces
from backend.analysis import analyze_similarity

# ==================================
# KONFIGURASI HALAMAN
# ==================================

st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

h1 {
    text-align: center;
    color: #FFD54F;
    font-size: 50px !important;
    text-shadow: 3px 3px #000;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.pixel-card {
    background-color: #111827;
    border: 3px solid #3B82F6;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 6px 6px 0px #000;
}

.upload-box {
    border: 3px dashed #22C55E;
    padding: 20px;
    border-radius: 10px;
    background-color: #1E293B;
}

.stButton>button {
    width: 100%;
    height: 60px;
    background-color: #FACC15;
    color: black;
    font-size: 22px;
    font-weight: bold;
    border-radius: 8px;
    border: 3px solid black;
}

.stButton>button:hover {
    background-color: #F59E0B;
}

.metric-card {
    background-color: #111827;
    border-radius: 12px;
    padding: 15px;
    border: 2px solid #60A5FA;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Deteksi Kemiripan Wajah PCA",
    page_icon="🧑",
    layout="wide"
)

st.markdown("""
<h1>🎮 DETEKSI KEMIRIPAN WAJAH</h1>
<h3 style='text-align:center;color:white;'>
Foto Masa Kecil vs Foto Masa Dewasa
</h3>

<p style='text-align:center;color:#cbd5e1;font-size:20px'>
Menggunakan PCA, Cosine Similarity, dan Euclidean Distance
</p>
""", unsafe_allow_html=True)

st.sidebar.title("📖 Informasi")

st.sidebar.success("""
Metode yang digunakan:

- PCA
- Cosine Similarity
- Euclidean Distance

Upload dua foto untuk mengetahui tingkat kemiripan wajah.
""")

st.write("""
Upload foto masa kecil dan foto masa dewasa,
kemudian sistem akan menghitung tingkat kemiripan
menggunakan PCA, Cosine Similarity, dan Euclidean Distance.
""")

# ==================================
# LOAD MODEL PCA
# ==================================

try:
    pca_model = load_pca_model()

except Exception as e:

    st.error(
        f"Gagal memuat model PCA:\n\n{e}"
    )

    st.stop()

# ==================================
# FACE DETECTOR
# ==================================

detector = FaceDetection()

# ==================================
# UPLOAD FILE
# ==================================

col1, col2 = st.columns(2)

with col1:

    st.markdown("## 👶 Foto Masa Kecil")

    child_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"],
        key="child"
    )

with col2:

    st.markdown("## 🧑 Foto Masa Dewasa")

    adult_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"],
        key="adult"
    )

# ==================================
# PROSES
# ==================================

if st.button("🎮 BANDINGKAN WAJAH"):

    if child_file is None or adult_file is None:

        st.warning(
            "Silakan upload kedua gambar terlebih dahulu."
        )

        st.stop()

    try:

        # ==================================
        # BACA GAMBAR
        # ==================================

        child_image = np.array(
            Image.open(child_file).convert("RGB")
        )

        adult_image = np.array(
            Image.open(adult_file).convert("RGB")
        )

        child_bgr = cv2.cvtColor(
            child_image,
            cv2.COLOR_RGB2BGR
        )

        adult_bgr = cv2.cvtColor(
            adult_image,
            cv2.COLOR_RGB2BGR
        )

        # ==================================
        # FACE DETECTION
        # ==================================

        child_face = detector.crop_largest_face(
            child_bgr
        )

        adult_face = detector.crop_largest_face(
            adult_bgr
        )

        # ==================================
        # FACE ALIGNMENT
        # ==================================

        child_face = align_face(
            child_face
        )

        adult_face = align_face(
            adult_face
        )

        # ==================================
        # PREPROCESSING
        # ==================================

        child_vector = preprocess_image(
            child_face
        )

        adult_vector = preprocess_image(
            adult_face
        )

        # ==================================
        # PCA TRANSFORM
        # ==================================

        child_pca = transform_vector(
            child_vector,
            pca_model
        )

        adult_pca = transform_vector(
            adult_vector,
            pca_model
        )

        # ==================================
        # SIMILARITY
        # ==================================

        similarity_result = compare_faces(
            child_pca,
            adult_pca
        )

        # ==================================
        # DEBUG
        # ==================================

        st.write("### Debug")

        st.write(
            "Cosine Similarity:",
            similarity_result["cosine_similarity"]
        )

        st.write(
            "Euclidean Distance:",
            similarity_result["euclidean_distance"]
        )

        # ==================================
        # ANALYSIS
        # ==================================

        analysis = analyze_similarity(
            similarity_result["cosine_similarity"],
            similarity_result["euclidean_distance"]
        )

        # ==================================
        # PREVIEW
        # ==================================

        st.divider()

        st.markdown("## 🖼️ Preview Foto yang Diunggah")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                child_image,
                caption="👶 Foto Masa Kecil",
                use_container_width=True
            )

        with c2:
            st.image(
                adult_image,
                caption="🧑 Foto Masa Dewasa",
                use_container_width=True
            )

        # ==================================
        # HASIL CROP WAJAH
        # ==================================

        st.divider()

        st.subheader("🧑 Hasil Crop Wajah")

        c3, c4 = st.columns(2)

        with c3:

            st.image(
                cv2.cvtColor(
                    child_face,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Wajah Anak",
                use_container_width=True
            )

        with c4:

            st.image(
                cv2.cvtColor(
                    adult_face,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Wajah Dewasa",
                use_container_width=True
            )

        # ==================================
        # HASIL ANALISIS
        # ==================================

        st.divider()

        st.subheader("🏆 Hasil Analisis")

        persen = analysis["similarity_percent"]

        progress_value = max(
            0.0,
            min(1.0, persen / 100)
        )

        # Tampilkan 3 metrik dalam satu baris
        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric(
                "🎯 Kemiripan",
                f"{persen:.2f}%"
            )

        with k2:
            st.metric(
                "📐 Cosine Similarity",
                f"{analysis['cosine_similarity']:.4f}"
            )

        with k3:
            st.metric(
                "📏 Euclidean Distance",
                f"{analysis['euclidean_distance']:.4f}"
            )

        # Progress bar
        st.progress(progress_value)

        # ==================================
        # KATEGORI
        # ==================================

        if persen >= 85:

            st.success(
                f"Kategori: {analysis['category']}"
            )

            st.balloons()

        elif persen >= 70:

            st.info(
                f"Kategori: {analysis['category']}"
            )

        elif persen >= 55:

            st.warning(
                f"Kategori: {analysis['category']}"
            )

        else:

            st.error(
                f"Kategori: {analysis['category']}"
            )

        st.info(
            analysis["description"]
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan:\n\n{e}"
        )