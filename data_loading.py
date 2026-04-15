"""
Data loading utilities for SUN and CUB datasets.
Handles loading and preprocessing of raw data files.
"""

import pandas as pd
import numpy as np
import scipy.io
import os


def load_sun_data(base_path='data/SUN/SUNAttributeDB', threshold=0.5):
    """
    Load SUN Attributes dataset from .mat files.
    
    Args:
        base_path: Path to SUNAttributeDB directory
        threshold: Threshold for binarizing continuous attribute labels
        
    Returns:
        df_sun: DataFrame with image paths and binary attribute labels
        attributes_list: List of attribute names
        attr_cols: List of attribute column names in DataFrame
    """
    # Wczytanie .mat plików
    mat_attrs  = scipy.io.loadmat(os.path.join(base_path, 'attributes.mat'))
    mat_images = scipy.io.loadmat(os.path.join(base_path, 'images.mat'))
    mat_labels = scipy.io.loadmat(os.path.join(base_path, 'attributeLabels_continuous.mat'))

    # Ekstrakcja nazw atrybutów i ścieżek obrazów
    attributes_list = [item[0][0] for item in mat_attrs['attributes']]
    images_list     = [item[0][0] for item in mat_images['images']]
    labels_matrix   = mat_labels['labels_cv']

    # Binaryzacja etykiet
    binary_labels  = (labels_matrix > threshold).astype(float)

    # Stworzenie DataFramu
    df_sun = pd.DataFrame(binary_labels, columns=attributes_list)
    df_sun.insert(0, 'image_id',  range(1, len(images_list) + 1))
    df_sun.insert(1, 'file_path', images_list)
    df_sun.insert(2, 'class_id',  df_sun['file_path'].apply(lambda x: x.split('/')[0] if '/' in x else x))

    # Kolumny atrybutów
    non_attr_cols = ['image_id', 'file_path', 'class_id']
    attr_cols     = [c for c in df_sun.columns if c not in non_attr_cols]

    return df_sun, attributes_list, attr_cols


def load_cub_data(base_path='data/CUB', threshold=0.5):
    """
    Load Caltech-UCSD Birds (CUB) dataset.
    
    Args:
        base_path: Path to CUB directory
        threshold: Threshold for binarizing attribute labels
        
    Returns:
        df_final: DataFrame with image paths and attribute labels
        attr_cols: List of attribute column names
        attr_mapping: Dictionary mapping attribute IDs to names
    """
    # Wczytanie metadanych
    images = pd.read_csv(os.path.join(base_path, 'images.txt'), 
                         sep=' ', header=None, names=['image_id', 'file_path'])
    split = pd.read_csv(os.path.join(base_path, 'train_test_split.txt'), 
                        sep=' ', header=None, names=['image_id', 'is_training'])
    classes = pd.read_csv(os.path.join(base_path, 'image_class_labels.txt'), 
                          sep=' ', header=None, names=['image_id', 'class_id'])
    
    metadata = images.merge(split, on='image_id').merge(classes, on='image_id')
    
    # Wczytanie atrybutów
    col_names = ['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time', 'extra']
    img_attrs = pd.read_csv(os.path.join(base_path, 'attributes/image_attribute_labels.txt'), 
                            sep=r'\s+',  
                            header=None, 
                            names=col_names)
    img_attrs = img_attrs[['image_id', 'attribute_id', 'is_present', 'certainty_id']]
    
    # Stworzenie macierzy atrybutów
    attr_matrix = img_attrs.pivot(index='image_id', columns='attribute_id', values='is_present').fillna(0)
    
    # Połączenie metadanych z atrybutami
    df_final = metadata.merge(attr_matrix, on='image_id')
    
    # Mapowanie nazw atrybutów
    attributes_df = pd.read_csv(os.path.join(base_path, 'attributes.txt'), 
                                sep=r'\s+', header=None, names=['attr_id', 'attr_name'])
    attr_mapping = dict(zip(attributes_df['attr_id'], attributes_df['attr_name']))
    
    # Kolumny atrybutów
    non_attr_cols = ['image_id', 'file_path', 'is_training', 'class_id']
    attr_cols = [c for c in df_final.columns if c not in non_attr_cols]
    
    return df_final, attr_cols, attr_mapping


def get_attribute_stats(attr_matrix, attr_mapping=None):
    """
    Calculate co-occurrence statistics for attributes.
    
    Args:
        attr_matrix: Attribute matrix (observations x attributes)
        attr_mapping: Optional dictionary mapping attribute IDs to names
        
    Returns:
        co_occurrence: Co-occurrence matrix
        p_B_given_A: Conditional probability matrix P(B|A)
    """
    co_occurrence = attr_matrix.T.dot(attr_matrix)
    sum_A = attr_matrix.sum(axis=0)
    p_B_given_A = co_occurrence.divide(sum_A, axis=0)
    np.fill_diagonal(p_B_given_A.values, 0)
    
    return co_occurrence, p_B_given_A
