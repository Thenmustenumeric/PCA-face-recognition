"""
pca_utils.py

Utilitas untuk memuat model PCA dan melakukan transformasi
vektor wajah ke ruang PCA.
"""

from pathlib import Path
import pickle
import numpy as np

# ==================================================
# Path Project
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

PCA_MODEL_PATH = MODEL_DIR / "pca_model.pkl"
LABELS_PATH = MODEL_DIR / "labels.pkl"
X_PCA_PATH = MODEL_DIR / "x_pca.npy"


# ==================================================
# Load PCA Model
# ==================================================

def load_pca_model():
    """
    Memuat model PCA yang telah dilatih.
    """

    if not PCA_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model PCA tidak ditemukan:\n{PCA_MODEL_PATH}"
        )

    file_size = PCA_MODEL_PATH.stat().st_size

    if file_size == 0:
        raise RuntimeError(
            f"File model kosong:\n{PCA_MODEL_PATH}"
        )

    try:

        with open(PCA_MODEL_PATH, "rb") as file:
            pca = pickle.load(file)

        return pca

    except Exception as e:

        raise RuntimeError(
            f"Gagal memuat model PCA:\n{e}"
        )


# ==================================================
# Load Labels
# ==================================================

def load_labels():
    """
    Memuat label dataset.
    """

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"labels.pkl tidak ditemukan:\n{LABELS_PATH}"
        )

    file_size = LABELS_PATH.stat().st_size

    if file_size == 0:
        raise RuntimeError(
            f"File labels.pkl kosong:\n{LABELS_PATH}"
        )

    with open(LABELS_PATH, "rb") as file:
        labels = pickle.load(file)

    return np.array(labels)


# ==================================================
# Load Dataset Projection
# ==================================================

def load_dataset_projection():
    """
    Memuat hasil proyeksi PCA dataset.
    """

    if not X_PCA_PATH.exists():
        raise FileNotFoundError(
            f"x_pca.npy tidak ditemukan:\n{X_PCA_PATH}"
        )

    return np.load(X_PCA_PATH)


# ==================================================
# Transform Vector
# ==================================================

def transform_vector(vector, pca=None):
    """
    Mengubah vektor wajah menjadi representasi PCA.
    """

    if pca is None:
        pca = load_pca_model()

    vector = np.asarray(vector)

    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    transformed = pca.transform(vector)

    return transformed


# ==================================================
# Model Information
# ==================================================

def get_model_information():
    """
    Mengambil informasi model PCA.
    """

    pca = load_pca_model()

    info = {
        "n_components": int(
            pca.n_components_
        ),
        "n_features": int(
            pca.n_features_in_
        ),
        "explained_variance": float(
            np.sum(
                pca.explained_variance_ratio_
            )
        )
    }

    return info


# ==================================================
# Debug
# ==================================================

if __name__ == "__main__":

    try:

        print("=== DEBUG PCA ===")

        print("BASE_DIR:")
        print(BASE_DIR)

        print("\nMODEL_DIR:")
        print(MODEL_DIR)

        print("\nPCA MODEL:")
        print(PCA_MODEL_PATH)

        print("\nEXISTS:")
        print(PCA_MODEL_PATH.exists())

        if PCA_MODEL_PATH.exists():
            print(
                "SIZE:",
                PCA_MODEL_PATH.stat().st_size,
                "bytes"
            )

        info = get_model_information()

        print("\n=== INFORMASI MODEL ===")

        print(
            "Jumlah Komponen:",
            info["n_components"]
        )

        print(
            "Jumlah Fitur:",
            info["n_features"]
        )

        print(
            "Explained Variance:",
            info["explained_variance"]
        )

    except Exception as e:

        print("\nERROR:")
        print(e)