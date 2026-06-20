"""
preprocess.py

Modul untuk melakukan preprocessing gambar wajah sebelum diproses
menggunakan PCA.

Tahapan:
1. Konversi ke grayscale
2. Resize ke ukuran tetap
3. Normalisasi piksel (0 - 1)
4. Flatten menjadi vektor 1 dimensi
"""

import cv2
import numpy as np

# Ukuran gambar yang akan digunakan untuk PCA
IMG_SIZE = (100, 100)


def preprocess_image(image):
    """
    Melakukan preprocessing pada gambar yang sudah dibaca.

    Parameters
    ----------
    image : numpy.ndarray
        Gambar dalam format OpenCV (BGR atau Grayscale).

    Returns
    -------
    numpy.ndarray
        Vektor 1 dimensi hasil preprocessing.
    """

    if image is None:
        raise ValueError("Input gambar kosong atau tidak valid.")

    # Jika gambar masih berwarna (BGR), ubah menjadi grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize ke ukuran standar
    resized = cv2.resize(gray, IMG_SIZE)

    # Normalisasi ke rentang 0-1
    normalized = resized.astype(np.float32) / 255.0

    # Flatten menjadi vektor 1 dimensi
    vector = normalized.flatten()

    return vector


def preprocess_image_from_path(image_path):
    """
    Membaca gambar dari file kemudian melakukan preprocessing.

    Parameters
    ----------
    image_path : str
        Lokasi file gambar.

    Returns
    -------
    numpy.ndarray
        Vektor hasil preprocessing.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {image_path}"
        )

    return preprocess_image(image)


def preprocess_for_display(image):
    """
    Melakukan preprocessing tanpa flatten.
    Digunakan jika ingin menampilkan hasil preprocessing
    di Streamlit.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Gambar grayscale 100x100 dengan nilai 0-255.
    """

    if image is None:
        raise ValueError("Input gambar kosong.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    resized = cv2.resize(gray, IMG_SIZE)

    return resized


if __name__ == "__main__":
    # Contoh penggunaan
    sample_path = "contoh.jpg"

    try:
        vector = preprocess_image_from_path(sample_path)

        print("Preprocessing berhasil")
        print("Panjang vektor :", len(vector))
        print("Shape :", vector.shape)
        print("Nilai minimum :", vector.min())
        print("Nilai maksimum :", vector.max())

    except Exception as e:
        print("Terjadi kesalahan:", e)