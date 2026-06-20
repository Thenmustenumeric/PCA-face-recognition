"""
face_detection.py

Modul untuk mendeteksi dan memotong (crop) area wajah
menggunakan Haar Cascade OpenCV.
"""

import cv2
import numpy as np


class FaceDetection:
    def __init__(self):
        """
        Inisialisasi model Haar Cascade bawaan OpenCV.
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                "Gagal memuat Haar Cascade untuk deteksi wajah."
            )

    def detect_faces(self, image):
        """
        Mendeteksi semua wajah pada gambar.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        list
            List koordinat wajah (x, y, w, h).
        """

        if image is None:
            raise ValueError("Input gambar tidak valid.")

        # Konversi ke grayscale jika masih BGR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        return faces

    def crop_largest_face(self, image):
        """
        Mengambil wajah terbesar pada gambar.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Hasil crop wajah.

        Raises
        ------
        ValueError
            Jika tidak ada wajah terdeteksi.
        """

        faces = self.detect_faces(image)

        if len(faces) == 0:
            raise ValueError("Tidak ditemukan wajah pada gambar.")

        # Pilih wajah dengan area terbesar
        largest = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = largest

        cropped = image[y:y + h, x:x + w]

        return cropped

    def draw_face_box(self, image):
        """
        Menggambar kotak pada semua wajah yang terdeteksi.

        Cocok untuk preview di Streamlit.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Gambar dengan kotak deteksi.
        """

        output = image.copy()

        faces = self.detect_faces(output)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return output


# Contoh penggunaan
if __name__ == "__main__":

    detector = FaceDetection()

    img = cv2.imread("contoh.jpg")

    try:

        face = detector.crop_largest_face(img)

        print("Deteksi wajah berhasil.")
        print("Ukuran hasil crop:", face.shape)

        cv2.imshow("Face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)