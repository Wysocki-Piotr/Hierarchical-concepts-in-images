"""
Dataloaders for SUN and CUB datasets.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class SUNDataset(Dataset):
    """Dataset class for SUN Attributes dataset."""
    
    def __init__(self, df, img_dir, transform, attr_cols):
        """
        Args:
            df: DataFrame containing image paths and attribute labels
            img_dir: Path to directory containing images
            transform: Image transformation to apply
            attr_cols: List of attribute column names
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.attr_cols = attr_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['file_path']))
        
        # Wczytanie obrazu
        img = Image.open(img_path).convert('RGB')
        
        # Transformacja (np. centre_crop lub clip_preprocess)
        tensor_img = self.transform(img)
        
        # Etykiety atrybutów
        labels = row[self.attr_cols].values.astype(np.float32)
        
        # Zwracamy obraz, etykiety oraz ścieżkę
        return tensor_img, labels, img_path


class CUBDataset(Dataset):
    """Dataset class for Caltech-UCSD Birds (CUB) dataset."""
    
    def __init__(self, df, img_dir, transform, attr_cols):
        """
        Args:
            df: DataFrame containing image paths and attribute labels
            img_dir: Path to directory containing images
            transform: Image transformation to apply
            attr_cols: List of attribute column names
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.attr_cols = attr_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['file_path']))
        
        # Wczytanie obrazu
        img = Image.open(img_path).convert('RGB')
        
        # Transformacja
        tensor_img = self.transform(img)
        
        # Etykiety atrybutów
        labels = row[self.attr_cols].values.astype(np.float32)
        
        # Zwracamy obraz, etykiety oraz ścieżkę
        return tensor_img, labels, img_path


def create_sun_dataloader(df, img_dir, transform, attr_cols, batch_size=16, num_workers=0, shuffle=False):
    """
    Create a DataLoader for SUN dataset.
    
    Args:
        df: DataFrame containing SUN data
        img_dir: Path to SUN images directory
        transform: Image transformation pipeline
        attr_cols: List of attribute column names
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = SUNDataset(df=df, img_dir=img_dir, transform=transform, attr_cols=attr_cols)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True  # Szybsza transmisja danych na GPU
    )
    return dataloader


def create_cub_dataloader(df, img_dir, transform, attr_cols, batch_size=16, num_workers=0, shuffle=False):
    """
    Create a DataLoader for CUB dataset.
    
    Args:
        df: DataFrame containing CUB data
        img_dir: Path to CUB images directory
        transform: Image transformation pipeline
        attr_cols: List of attribute column names
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = CUBDataset(df=df, img_dir=img_dir, transform=transform, attr_cols=attr_cols)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
