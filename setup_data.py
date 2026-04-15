#!/usr/bin/env python3
"""
Setup script to download and organize datasets.

Usage:
    python setup_data.py --sun         # Download SUN dataset
    python setup_data.py --cub         # Download CUB dataset
    python setup_data.py --all         # Download all datasets
    python setup_data.py --check       # Check if data is present
"""

import os
import sys
import argparse
from pathlib import Path


def check_sun_dataset():
    """Check if SUN dataset is present."""
    sun_path = Path('data/SUN/SUNAttributeDB')
    required_files = [
        'attributes.mat',
        'images.mat',
        'attributeLabels_continuous.mat',
        'images'  # Directory
    ]
    
    if not sun_path.exists():
        return False
    
    for file in required_files:
        if not (sun_path / file).exists():
            return False
    
    print("✓ SUN dataset found")
    return True


def check_cub_dataset():
    """Check if CUB dataset is present."""
    cub_path = Path('data/CUB')
    required_files = [
        'images.txt',
        'image_class_labels.txt',
        'train_test_split.txt',
        'attributes.txt',
        'attributes/image_attribute_labels.txt',
        'images'  # Directory
    ]
    
    if not cub_path.exists():
        return False
    
    for file in required_files:
        if not (cub_path / file).exists():
            return False
    
    print("✓ CUB dataset found")
    return True


def check_models():
    """Check if pre-trained models are present."""
    models_path = Path('models')
    required_models = [
        'resnet18_places365.pth'
    ]
    
    found = True
    for model in required_models:
        if (models_path / model).exists():
            print(f"✓ {model} found")
        else:
            print(f"✗ {model} NOT found - will be loaded from torchvision/CLIP")
            found = False
    
    return found


def create_directories():
    """Create necessary directories."""
    dirs = [
        'data/SUN/SUNAttributeDB',
        'data/CUB/attributes',
        'models'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def setup_sun():
    """Setup SUN dataset - manual download required."""
    print("\n" + "="*60)
    print("SUN ATTRIBUTES DATASET SETUP")
    print("="*60)
    print("""
    The SUN Attributes dataset must be downloaded manually:
    
    1. Visit: http://cs.mit.edu/~kadir/sunattributes/
    2. Download the dataset
    3. Extract to: data/SUN/SUNAttributeDB/
    
    Directory structure should be:
    data/SUN/SUNAttributeDB/
        ├── attributes.mat
        ├── images.mat
        ├── attributeLabels_continuous.mat
        └── images/
            ├── abbey/
            ├── airplane_cabin/
            └── ... (more image folders)
    """)


def setup_cub():
    """Setup CUB dataset - manual download required."""
    print("\n" + "="*60)
    print("CUB (CALTECH-UCSD BIRDS) DATASET SETUP")
    print("="*60)
    print("""
    The CUB dataset must be downloaded manually:
    
    1. Visit: http://www.vision.caltech.edu/datasets/
    2. Download CUB-200-2011 dataset
    3. Extract to: data/CUB/
    
    Directory structure should be:
    data/CUB/
        ├── images.txt
        ├── image_class_labels.txt
        ├── train_test_split.txt
        ├── attributes.txt
        ├── attributes/
        │   ├── image_attribute_labels.txt
        │   └── class_attribute_labels_continuous.txt
        └── images/
            ├── 001.Black_footed_Albatross/
            ├── 002.Laysan_Albatross/
            └── ... (more bird species)
    """)


def check_all():
    """Check if all datasets are present."""
    print("\n" + "="*60)
    print("CHECKING DATA STATUS")
    print("="*60 + "\n")
    
    sun_ok = check_sun_dataset() if Path('data/SUN').exists() else False
    cub_ok = check_cub_dataset() if Path('data/CUB').exists() else False
    
    if not sun_ok:
        print("✗ SUN dataset NOT found")
    if not cub_ok:
        print("✗ CUB dataset NOT found")
    
    print("\nChecking models:")
    check_models()
    
    if sun_ok and cub_ok:
        print("\n✓ All datasets ready!")
        return True
    else:
        print("\n✗ Some datasets missing. Run:")
        print("  python setup_data.py --sun   (for SUN dataset)")
        print("  python setup_data.py --cub   (for CUB dataset)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Setup datasets for Hierarchical-concepts-in-images project'
    )
    parser.add_argument('--sun', action='store_true', help='Setup SUN dataset')
    parser.add_argument('--cub', action='store_true', help='Setup CUB dataset')
    parser.add_argument('--all', action='store_true', help='Setup all datasets')
    parser.add_argument('--check', action='store_true', help='Check dataset status')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # If no arguments, show help and check status
    if not (args.sun or args.cub or args.all or args.check):
        check_all()
        return
    
    # Check status
    if args.check:
        check_all()
        return
    
    # Setup specific datasets
    if args.sun or args.all:
        setup_sun()
    
    if args.cub or args.all:
        setup_cub()
    
    print("\n" + "="*60)
    print("After downloading and extracting datasets, run:")
    print("  python setup_data.py --check")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
