"""
similarity.py

Modul untuk menghitung kemiripan antara dua vektor PCA.
Menggunakan:
1. Cosine Similarity
2. Euclidean Distance
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_score(vector1, vector2):
    """
    Menghitung Cosine Similarity antara dua vektor.

    Parameters
    ----------
    vector1 : numpy.ndarray
    vector2 : numpy.ndarray

    Returns
    -------
    float
    """

    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)

    if vector1.ndim == 1:
        vector1 = vector1.reshape(1, -1)

    if vector2.ndim == 1:
        vector2 = vector2.reshape(1, -1)

    score = cosine_similarity(vector1, vector2)[0][0]

    return float(score)


def euclidean_distance(vector1, vector2):
    """
    Menghitung Euclidean Distance antara dua vektor.

    Parameters
    ----------
    vector1 : numpy.ndarray
    vector2 : numpy.ndarray

    Returns
    -------
    float
    """

    vector1 = np.asarray(vector1).flatten()
    vector2 = np.asarray(vector2).flatten()

    distance = np.linalg.norm(vector1 - vector2)

    return float(distance)


def compare_faces(vector1, vector2):
    """
    Membandingkan dua wajah sekaligus.

    Parameters
    ----------
    vector1 : numpy.ndarray
    vector2 : numpy.ndarray

    Returns
    -------
    dict
    """

    cosine = cosine_similarity_score(vector1, vector2)
    euclidean = euclidean_distance(vector1, vector2)

    return {
        "cosine_similarity": cosine,
        "euclidean_distance": euclidean
    }


def find_most_similar(query_vector, database_vectors):
    """
    Mencari data paling mirip dalam database PCA.

    Parameters
    ----------
    query_vector : numpy.ndarray
    database_vectors : numpy.ndarray

    Returns
    -------
    tuple
        (
            index_terbaik,
            cosine_similarity,
            euclidean_distance
        )
    """

    best_index = -1
    best_cosine = -1.0
    best_distance = float("inf")

    for i, db_vector in enumerate(database_vectors):

        cosine = cosine_similarity_score(
            query_vector,
            db_vector
        )

        distance = euclidean_distance(
            query_vector,
            db_vector
        )

        if cosine > best_cosine:
            best_cosine = cosine
            best_distance = distance
            best_index = i

    return (
        best_index,
        best_cosine,
        best_distance
    )


if __name__ == "__main__":

    # Contoh sederhana

    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 3, 5])

    result = compare_faces(a, b)

    print(result)