import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

from backend.face_detection import FaceDetection
from backend.face_alignment import align_face
from backend.preprocess import preprocess_image

# Konfigurasi
DATASET_DIR = "dataset/train"
MODEL_DIR = "models"
PCA_MODEL_PATH = os.path.join(MODEL_DIR, "pca_model.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

# Jumlah komponen PCA
N_COMPONENTS = 50


def load_dataset():
    """
    Membaca seluruh dataset dan melakukan preprocessing.

    Struktur dataset:
    dataset/train/
        person1/
            img1.jpg
            img2.jpg
        person2/
            img1.jpg
            img2.jpg
    """

    detector = FaceDetection()

    X = []
    labels = []

    supported_ext = (".jpg", ".jpeg", ".png")

    for person_name in os.listdir(DATASET_DIR):

        person_folder = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):

            if not filename.lower().endswith(supported_ext):
                continue

            image_path = os.path.join(person_folder, filename)

            try:
                import cv2

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Gagal membaca: {image_path}")
                    continue

                # Deteksi wajah
                face = detector.crop_largest_face(image)

                # Alignment
                aligned = align_face(face)

                # Preprocessing
                vector = preprocess_image(aligned)

                X.append(vector)
                labels.append(person_name)

            except Exception as e:
                print(f"Skip {image_path}: {e}")

    return np.array(X), np.array(labels)


def train_pca():

    X, labels = load_dataset()

    if len(X) == 0:
        raise RuntimeError(
            "Dataset kosong atau tidak ada wajah yang berhasil diproses."
        )

    print(f"Jumlah sampel: {len(X)}")
    print(f"Jumlah fitur : {X.shape[1]}")

    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="auto",
        random_state=42
    )

    X_pca = pca.fit_transform(X)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Simpan model PCA
    with open(PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca, f)

    # Simpan label dataset
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    print("Model PCA berhasil disimpan.")
    print("Explained Variance:",
          np.sum(pca.explained_variance_ratio_))

    return X_pca, labels


if __name__ == "__main__":

    train_pca()