import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm


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


def calculate_filtered_cav(features_matrix, child_pos_indices, child_neg_indices, parent_pos_indices):
    """
    Calculates the Filtered CAV for a child concept (A), training on subset
    where the parent concept (B) is present.
    
    Assumption: Concept A appears only in the context of concept B (A => B).
    
    Parameters:
    - features_matrix: Feature matrix (embeddings)
    - child_pos_indices: Indices where child concept A is present
    - child_neg_indices: Indices where child concept A is NOT present   
    - parent_pos_indices: Indices where parent concept B is PRESENT
    
    Returns: Normalized CAV vector trained on the filtered subset.
    """
    # We use set intersection to get indices with (child and parent) and (parent without this child)
    filtered_pos_idx = np.intersect1d(child_pos_indices, parent_pos_indices)
    filtered_neg_idx = np.intersect1d(child_neg_indices, parent_pos_indices)
    
    # Check if there is anything left to learn after filtering
    if len(filtered_pos_idx) == 0 or len(filtered_neg_idx) == 0:
         raise ValueError(
            'Not enough data after filtering the subset relative to the parent. '
            'Ensure the child concept occurs within the parent concept.'
        )
         
    mean_concept = np.mean(features_matrix[filtered_pos_idx], axis=0)
    mean_background = np.mean(features_matrix[filtered_neg_idx], axis=0)
    
    cav = mean_concept - mean_background
    
    norm = np.linalg.norm(cav)
    if norm > 1e-8:
        return cav / norm
    
    raise ValueError('The CAV norm after filtering is close to zero.')





@DeprecationWarning
def get_concept_distribution_stats(features_train, labels_train, cav_vector):
    """
    Calculates the statistics of the cosine similarity distribution for a specific concept
    on the positive set (only on images that DEFINITELY have this concept).

    Returns: (mu, sigma) - mean and standard deviation.
    """
    pos_features = features_train[labels_train == 1.0]
    
    if len(pos_features) < 2:
        raise ValueError("Too litte positive samples to calculate distribution statistics reliably.")
        
    similarities = cosine_similarity(pos_features, cav_vector.reshape(1, -1)).flatten()
    
    mu = np.mean(similarities)
    sigma = np.std(similarities)
    
    return mu, sigma

PARENTS_THRESHOLDS = {
    "enclosed area": 0.0667,
    "aged/ worn": 0.2222,
    "working": 0.1556,
    "natural light": -0.0444,
    "rock/stone": 0.3333,
    "cold": 0.3000,
    "asphalt": 0.1778,
    "open area": 0.0000
}

def is_obs_in_parent(X_obs: np.ndarray, name_parent: str, parent_cav_vector: np.ndarray) -> bool:
    """
    Checks if the observation X_obs belongs to the parent concept defined by parent_cav_vector
    using a predefined threshold for that parent concept.
    """
    if name_parent not in PARENTS_THRESHOLDS:
        raise ValueError(f"Nie znaleziono progu dla pojęcia nadrzędnego: '{name_parent}'")
    
    threshold = PARENTS_THRESHOLDS[name_parent]
    X_obs = np.asarray(X_obs).flatten()
    parent_cav_vector = np.asarray(parent_cav_vector).flatten()
    
    norm_obs = np.linalg.norm(X_obs)
    norm_parent = np.linalg.norm(parent_cav_vector)
    if norm_obs == 0 or norm_parent == 0: 
        return False
    sim = np.dot(X_obs, parent_cav_vector) / (norm_obs * norm_parent)
    
    return bool(sim >= threshold)


def get_matching_parent(X_obs, parent_cavs_dict, thresholds_dict):
    """
    Sprawdza do jakiego środowiska (rodzica) należy dana obserwacja.
    Zwraca pierwszy dopasowany kontekst: (nazwa_rodzica, wektor_rodzica) 
    lub (None, False), jeśli obserwacja nie należy do żadnego.
    """
    X_obs_flat = np.asarray(X_obs).flatten()
    norm_obs = np.linalg.norm(X_obs_flat)

    if norm_obs == 0:
        return None, False

    # Przeszukujemy dostępnych rodziców
    for parent_name, parent_cav in parent_cavs_dict.items():
        if parent_name not in thresholds_dict:
            continue
            
        threshold = thresholds_dict[parent_name]
        
        cav_flat = np.asarray(parent_cav).flatten()
        norm_cav = np.linalg.norm(cav_flat)
        
        if norm_cav == 0:
            continue
            
        # Szybki kosinus w czystym NumPy
        sim = np.dot(X_obs_flat, cav_flat) / (norm_obs * norm_cav)
        
        # Jeśli podobieństwo przekracza twardy próg, przerywamy i zwracamy sukces
        if sim >= threshold:
            return parent_name, parent_cav
            
    # Zwracamy False, jeśli żaden rodzic nie został dopasowany
    return None, False