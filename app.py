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

try:
    pca_model = load_pca_model()
except Exception as e:
    st.exception(e)
    st.stop()
    
# ==========================
# Konfigurasi Halaman
# ==========================

st.set_page_config(
    page_title="Deteksi Kemiripan Wajah dengan PCA",
    page_icon="🧑",
    layout="wide"
)

st.title("🧑 Deteksi Kemiripan Wajah Menggunakan PCA")
st.write(
    """
Upload **foto masa kecil** dan **foto masa dewasa**,
kemudian sistem akan menghitung tingkat kemiripan
menggunakan PCA, Cosine Similarity, dan Euclidean Distance.
"""
)

# ==========================
# Load Model
# ==========================

try:
    pca_model = load_pca_model()
except Exception as e:
    st.error(f"Gagal memuat model PCA:\n\n{str(e)}")
    st.stop()

detector = FaceDetection()

# ==========================
# Upload File
# ==========================

col1, col2 = st.columns(2)

with col1:
    child_file = st.file_uploader(
        "Upload Foto Masa Kecil",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    adult_file = st.file_uploader(
        "Upload Foto Masa Dewasa",
        type=["jpg", "jpeg", "png"]
    )

# ==========================
# Tombol Proses
# ==========================

if st.button("Bandingkan Wajah"):

    if child_file is None or adult_file is None:
        st.warning("Silakan upload kedua gambar terlebih dahulu.")
        st.stop()

    try:

        # ----------------------
        # Baca gambar
        # ----------------------

        child_image = np.array(Image.open(child_file).convert("RGB"))
        adult_image = np.array(Image.open(adult_file).convert("RGB"))

        # RGB -> BGR (OpenCV)
        child_bgr = cv2.cvtColor(child_image, cv2.COLOR_RGB2BGR)
        adult_bgr = cv2.cvtColor(adult_image, cv2.COLOR_RGB2BGR)

        # ----------------------
        # Face Detection
        # ----------------------

        child_face = detector.crop_largest_face(child_bgr)
        adult_face = detector.crop_largest_face(adult_bgr)

        # ----------------------
        # Face Alignment
        # ----------------------

        child_face = align_face(child_face)
        adult_face = align_face(adult_face)

        # ----------------------
        # Preprocessing
        # ----------------------

        child_vector = preprocess_image(child_face)
        adult_vector = preprocess_image(adult_face)

        # ----------------------
        # PCA Projection
        # ----------------------

        child_pca = transform_vector(
            child_vector,
            pca_model
        )

        adult_pca = transform_vector(
            adult_vector,
            pca_model
        )

        # ----------------------
        # Similarity
        # ----------------------

        similarity_result = compare_faces(
            child_pca,
            adult_pca
        )

        # ----------------------
        # Analysis
        # ----------------------

        analysis = analyze_similarity(
            similarity_result["cosine_similarity"],
            similarity_result["euclidean_distance"]
        )

        # ==========================
        # Tampilkan Preview
        # ==========================

        st.divider()

        st.subheader("Preview Gambar")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                child_image,
                caption="Foto Masa Kecil",
                use_container_width=True
            )

        with c2:
            st.image(
                adult_image,
                caption="Foto Masa Dewasa",
                use_container_width=True
            )

        st.divider()

        st.subheader("Hasil Crop Wajah")

        c3, c4 = st.columns(2)

        with c3:
            st.image(
                cv2.cvtColor(child_face, cv2.COLOR_BGR2RGB),
                caption="Wajah Anak",
                use_container_width=True
            )

        with c4:
            st.image(
                cv2.cvtColor(adult_face, cv2.COLOR_BGR2RGB),
                caption="Wajah Dewasa",
                use_container_width=True
            )

        st.divider()

        st.subheader("Hasil Analisis")

        st.metric(
            "Cosine Similarity",
            f"{analysis['similarity_percent']} %"
        )

        st.metric(
            "Euclidean Distance",
            f"{analysis['euclidean_distance']:.4f}"
        )

        st.success(
            f"Kategori: {analysis['category']}"
        )

        st.info(
            analysis["description"]
        )

    except Exception as e:

        st.error(f"Terjadi kesalahan:\n\n{e}")