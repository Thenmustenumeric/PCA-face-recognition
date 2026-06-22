import cv2
import numpy as np

# ukuran standar wajah setelah alignment
ALIGN_SIZE = (100, 100)


def align_face(face_image):
    """
    Menyelaraskan wajah hasil crop.

    Parameters
    ----------
    face_image : numpy.ndarray
        Gambar wajah hasil deteksi.

    Returns
    -------
    numpy.ndarray
        Gambar wajah yang telah disejajarkan.
    """

    if face_image is None:
        raise ValueError("Input wajah tidak valid.")

    # Resize ke ukuran standar
    aligned = cv2.resize(face_image, ALIGN_SIZE)

    return aligned


def align_face_from_path(image_path, detector):
    """
    Membaca gambar, mendeteksi wajah, lalu melakukan alignment.

    Parameters
    ----------
    image_path : str
        Lokasi file gambar.

    detector : FaceDetection
        Objek dari FaceDetection.

    Returns
    -------
    numpy.ndarray
        Wajah yang telah disejajarkan.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {image_path}"
        )

    # Crop wajah terlebih dahulu
    face = detector.crop_largest_face(image)

    # Alignment sederhana
    aligned = align_face(face)

    return aligned


def show_alignment_info(face_image):
    """
    Mengembalikan informasi ukuran gambar
    sebelum dan sesudah alignment.

    Parameters
    ----------
    face_image : numpy.ndarray

    Returns
    -------
    dict
    """

    aligned = align_face(face_image)

    return {
        "original_shape": face_image.shape,
        "aligned_shape": aligned.shape
    }


if __name__ == "__main__":

    from face_detection import FaceDetection

    detector = FaceDetection()

    try:

        aligned_face = align_face_from_path(
            "contoh.jpg",
            detector
        )

        print("Alignment berhasil.")
        print("Ukuran hasil:", aligned_face.shape)

        cv2.imshow("Aligned Face", aligned_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)