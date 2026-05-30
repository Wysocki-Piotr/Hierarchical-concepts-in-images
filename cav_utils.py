import numpy as np
from sklearn.metrics import f1_score
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



def is_obs_in_concept(X_obs: np.ndarray, 
                      features: np.ndarray, labels: np.ndarray, 
                      cav_vector: np.ndarray, num_steps: int = 25) -> bool:
    """
    Determines if a single observation belongs to a specific concept.
    
    This function dynamically calculates the optimal cosine similarity threshold 
    by performing a quick grid search over the training features to maximize 
    the F1-score. It then applies this optimal threshold to evaluate the new observation.
    
    Args:
        X_obs (np.ndarray): The feature vector of the single new observation to be tested.
        features (np.ndarray): The feature matrix of the training set used for grid search.
        labels (np.ndarray): Binary labels (1.0 or 0.0) corresponding to the training set.
        cav_vector (np.ndarray): The Concept Activation Vector (CAV) for the concept.
        num_steps (int, optional): Number of threshold steps to search between -0.1 and 1.0. Defaults to 25.
        
    Returns bool
    """
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    cav_norm = np.asarray(cav_vector).flatten() / (np.linalg.norm(cav_vector) + 1e-10)
    
    similarities = np.dot(features_norm, cav_norm)
    
    thresholds = np.linspace(-0.1, 1.0, num_steps)
    
    best_t = 0.0
    best_f1 = 0.0
    
    # Grid search loop
    for t in thresholds:
        predictions = (similarities >= t).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    # 2. INFERENCE FOR X_obs (Pure NumPy optimization)
    X_obs_flat = np.asarray(X_obs).flatten()
    norm_obs = np.linalg.norm(X_obs_flat)
    
    if norm_obs == 0:
        return False
        
    # Since cav_norm is already normalized, we only divide by norm_obs
    sim = np.dot(X_obs_flat, cav_norm) / norm_obs

    return bool(sim >= best_t)



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

def is_obs_in_parent_preevaluted(X_obs: np.ndarray, name_parent: str, parent_cav_vector: np.ndarray) -> bool:
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

def find_optimal_threshold_vectorized(features, labels, cav_vector, num_steps=25):
    """
    Ekstrakcja logiki Grid Search z funkcji 'is_obs_in_concept'.
    Błyskawicznie wylicza optymalny próg na całym zbiorze treningowym.
    """
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    cav_norm = np.asarray(cav_vector).flatten() / (np.linalg.norm(cav_vector) + 1e-10)
    
    similarities = np.dot(features_norm, cav_norm)
    thresholds = np.linspace(-0.1, 1.0, num_steps)
    
    best_t, best_f1 = 0.0, 0.0
    for t in thresholds:
        predictions = (similarities >= t).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            
    return best_t




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