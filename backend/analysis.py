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
        Nilai cosine similarity (-1 sampai 1)

    euclidean_distance : float
        Nilai euclidean distance

    Returns
    -------
    dict
    """

    # Ubah cosine similarity (-1..1)
    # menjadi persentase (0..100)

    similarity_percent = ((cosine_score + 1) / 2) * 100

    # Pastikan selalu 0 - 100
    similarity_percent = max(
        0,
        min(100, similarity_percent)
    )

    similarity_percent = round(
        similarity_percent,
        2
    )

    # ==========================
    # Penentuan Kategori
    # ==========================

    if similarity_percent >= 85:

        category = "Sangat Mirip"

        description = (
            "Kedua wajah memiliki karakteristik yang "
            "sangat mirip berdasarkan representasi PCA."
        )

    elif similarity_percent >= 70:

        category = "Mirip"

        description = (
            "Terdapat banyak kesamaan pada fitur wajah "
            "meskipun masih terdapat beberapa perbedaan."
        )

    elif similarity_percent >= 55:

        category = "Cukup Mirip"

        description = (
            "Beberapa fitur wajah terlihat serupa "
            "namun perbedaannya masih cukup terlihat."
        )

    else:

        category = "Kurang Mirip"

        description = (
            "Representasi PCA menunjukkan perbedaan yang "
            "cukup besar antara kedua wajah."
        )

    return {

        "similarity_percent": similarity_percent,

        "cosine_similarity": round(
            cosine_score,
            4
        ),

        "euclidean_distance": round(
            euclidean_distance,
            4
        ),

        "category": category,

        "description": description
    }


def generate_report(result):
    """
    Membuat laporan hasil analisis.
    """

    report = (
        f"Tingkat Kemiripan : "
        f"{result['similarity_percent']}%\n\n"

        f"Cosine Similarity : "
        f"{result['cosine_similarity']}\n"

        f"Euclidean Distance : "
        f"{result['euclidean_distance']}\n\n"

        f"Kategori : "
        f"{result['category']}\n\n"

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