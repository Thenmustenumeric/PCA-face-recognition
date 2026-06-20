"""
analysis.py

Modul untuk menganalisis hasil kemiripan wajah
berdasarkan Cosine Similarity dan Euclidean Distance.
"""


def analyze_similarity(cosine_score, euclidean_distance):
    """
    Menganalisis tingkat kemiripan wajah.

    Parameters
    ----------
    cosine_score : float
        Nilai Cosine Similarity (0 - 1).

    euclidean_distance : float
        Nilai Euclidean Distance.

    Returns
    -------
    dict
        Informasi hasil analisis.
    """

    similarity_percent = round(cosine_score * 100, 2)

    # Penentuan kategori
    if cosine_score >= 0.90 and euclidean_distance <= 10:
        category = "Sangat Mirip"
        description = (
            "Kedua wajah memiliki karakteristik utama yang sangat serupa "
            "berdasarkan representasi PCA."
        )

    elif cosine_score >= 0.80 and euclidean_distance <= 15:
        category = "Mirip"
        description = (
            "Terdapat kemiripan yang kuat pada fitur-fitur utama wajah, "
            "meskipun terdapat beberapa perbedaan."
        )

    elif cosine_score >= 0.65 and euclidean_distance <= 20:
        category = "Cukup Mirip"
        description = (
            "Beberapa karakteristik wajah serupa, namun terdapat "
            "perbedaan yang cukup terlihat."
        )

    else:
        category = "Kurang Mirip"
        description = (
            "Representasi PCA menunjukkan perbedaan yang cukup besar "
            "antara kedua wajah."
        )

    return {
        "similarity_percent": similarity_percent,
        "cosine_similarity": round(cosine_score, 4),
        "euclidean_distance": round(euclidean_distance, 4),
        "category": category,
        "description": description
    }


def generate_report(result):
    """
    Membuat laporan singkat dalam bentuk teks.

    Parameters
    ----------
    result : dict

    Returns
    -------
    str
    """

    report = (
        f"Tingkat Kemiripan : {result['similarity_percent']}%\n"
        f"Cosine Similarity : {result['cosine_similarity']}\n"
        f"Euclidean Distance : {result['euclidean_distance']}\n"
        f"Kategori : {result['category']}\n\n"
        f"Analisis:\n"
        f"{result['description']}"
    )

    return report


if __name__ == "__main__":

    cosine = 0.91
    distance = 8.42

    result = analyze_similarity(
        cosine,
        distance
    )

    print(generate_report(result))