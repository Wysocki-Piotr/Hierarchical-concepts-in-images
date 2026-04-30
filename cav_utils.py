import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_centroid_cav(features_matrix, concept_indices, background_indices=None):
    """Wyznacza CAV jako różnicę średnich i normalizuje wynik."""
    mean_concept = np.mean(features_matrix[concept_indices], axis=0)
    if background_indices is None:
        mean_background = np.mean(features_matrix, axis=0)
    else:
        mean_background = np.mean(features_matrix[background_indices], axis=0)

    cav = mean_concept - mean_background
    norm = np.linalg.norm(cav)
    if norm > 1e-8:
        return cav / norm
    raise ValueError('Norma CAV jest bliska zeru.')


def get_cosine_similarity(vec1, vec2):
    """Zwraca wartość podobieństwa cosinusowego dla dwóch wektorów."""
    return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])


def get_dot_product_similarity(vec1, vec2):
    """Zwraca wartość iloczynu skalarnego dwóch wektorów."""
    return float(np.dot(vec1, vec2))


def calculate_hcep_cav(features_matrix, concept_indices, background_indices, parent_cav=None):
    """Wylicza HCEP CAV poprzez odjęcie składowej równoległej do wektora rodzica."""
    mean_concept = np.mean(features_matrix[concept_indices], axis=0)
    mean_background = np.mean(features_matrix[background_indices], axis=0)
    cav_raw = mean_concept - mean_background

    if parent_cav is not None:
        projection_scalar = np.dot(cav_raw, parent_cav)
        projection_vector = projection_scalar * parent_cav
        cav_raw = cav_raw - projection_vector

    norm = np.linalg.norm(cav_raw)
    if norm > 1e-8:
        return cav_raw / norm
    raise ValueError(
        'Norma HCEP CAV po projekcji jest bliska zeru - pojęcie i rodzic mogą być prawie identyczne.'
    )


def projection_ratio(child_cav_raw, parent_cav):
    """Zwraca kwadrat projekcji surowego CAV dziecka na parent_cav."""
    return float(np.dot(child_cav_raw, parent_cav) ** 2)
