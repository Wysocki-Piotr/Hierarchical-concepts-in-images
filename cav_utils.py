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


def is_obs_in_concept(X_obs, cav_parent, mu_parent, sigma_parent, 
                      cav_child, mu_child, sigma_child, prob_z_thresh=0.95):
    """
    Cascade inference for single new observation X_obs.
    Returns a bool tuple (exists_parent_B, exists_child_A)


    similarity_score >= (mu - z_threshold * sigma)
    Checks if the similarity score falls 
    within the concept distribution using one sided Z-Score: (x - mu)/ sigma >= -z_threshold
    """
    if sigma_parent == 0:
        return False, False

    X_obs = np.asarray(X_obs).flatten()
    cav_parent = np.asarray(cav_parent).flatten()
    cav_child = np.asarray(cav_child).flatten()
    
    norm_obs = np.linalg.norm(X_obs)
    norm_parent = np.linalg.norm(cav_parent)
    norm_child = np.linalg.norm(cav_child)

    if norm_obs == 0 or norm_parent == 0 or norm_child == 0: 
        return False, False

    Z_SCORES = {
    0.75: 0.67449, 
    0.80: 0.84162,
    0.85: 1.03643,
    0.90: 1.28155,
    0.95: 1.64485,
    0.975: 1.95996,
    0.99: 2.32634,
    0.999: 3.09024
    }

    z_thresh = Z_SCORES.get(prob_z_thresh, 1.64485) # default to 1.64485 for 95% if not found
    
    sim_b = np.dot(X_obs, cav_parent) / (norm_obs * norm_parent)
    #sim_b = cosine_similarity(X_obs, cav_parent.reshape(1, -1)).flatten()[0]
    exists_b = bool( sim_b >= (mu_parent - z_thresh * sigma_parent) )

    if not exists_b:
        return False, False
    if sigma_child == 0:
        return True, False
    
    # sim_a = cosine_similarity(X_obs, cav_child.reshape(1, -1)).flatten()[0]
    sim_a = np.dot(X_obs, cav_child) / (norm_obs * norm_child)
    exists_a = bool( sim_a >= (mu_child - z_thresh * sigma_child) )
    
    return True, exists_a