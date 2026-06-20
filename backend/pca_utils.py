"""
pca_utils.py

Utilitas untuk memuat model PCA dan melakukan transformasi
vektor wajah ke ruang PCA.
"""

import os
import pickle
import numpy as np

# Lokasi penyimpanan model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

PCA_MODEL_PATH = MODEL_DIR / "pca_model.pkl"
LABELS_PATH = MODEL_DIR / "labels.pkl"
X_PCA_PATH = MODEL_DIR / "x_pca.npy"


def load_pca_model():
    """
    Memuat model PCA yang telah dilatih.

    Returns
    -------
    PCA
        Objek PCA dari scikit-learn.
    """

    if not os.path.exists(PCA_MODEL_PATH):
        raise FileNotFoundError(
            "Model PCA tidak ditemukan. "
            "Jalankan train_model.py terlebih dahulu."
        )

    with open(PCA_MODEL_PATH, "rb") as file:
        pca = pickle.load(file)

    return pca


def load_labels():
    """
    Memuat label dataset.

    Returns
    -------
    numpy.ndarray
    """

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "File labels.pkl tidak ditemukan."
        )

    with open(LABELS_PATH, "rb") as file:
        labels = pickle.load(file)

    return np.array(labels)


def load_dataset_projection():
    """
    Memuat hasil proyeksi PCA dari dataset.

    Returns
    -------
    numpy.ndarray
    """

    if not os.path.exists(X_PCA_PATH):
        raise FileNotFoundError(
            "File x_pca.npy tidak ditemukan."
        )

    return np.load(X_PCA_PATH)


def transform_vector(vector, pca=None):
    """
    Mengubah satu vektor wajah menjadi representasi PCA.

    Parameters
    ----------
    vector : numpy.ndarray
        Vektor hasil preprocessing (1 dimensi).

    pca : PCA atau None
        Jika None maka model akan dimuat otomatis.

    Returns
    -------
    numpy.ndarray
        Vektor pada ruang PCA.
    """

    if pca is None:
        pca = load_pca_model()

    vector = np.asarray(vector)

    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    transformed = pca.transform(vector)

    return transformed


def get_model_information():
    """
    Mengambil informasi dasar model PCA.

    Returns
    -------
    dict
    """

    pca = load_pca_model()

    info = {
        "n_components": pca.n_components_,
        "n_features": pca.n_features_in_,
        "explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        )
    }

    return info


if __name__ == "__main__":

    try:

        info = get_model_information()

        print("=== Informasi Model PCA ===")
        print("Jumlah Komponen :", info["n_components"])
        print("Jumlah Fitur    :", info["n_features"])
        print("Explained Var   :", info["explained_variance"])

    except Exception as e:
        print(e)