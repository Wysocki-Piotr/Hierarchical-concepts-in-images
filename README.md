# Hierarchical-concepts-in-images

Badanie hierarchicznych koncepcji w obrazach z użyciem modeli pre-trained (ResNet18, CLIP).

## Quick Start

```bash
# 1. Klonuj repozytorium
git clone <repo-url>
cd Hierarchical-concepts-in-images

# 2. Pobierz i sprawdź dane
python setup_data.py --check

# 3. Uruchom notebookami
jupyter notebook
```

## Wymagania

- Python 3.8+
- PyTorch
- CLIP
- scikit-learn
- pandas, numpy
- Jupyter


## Pobieranie danych

### Automatyczne sprawdzenie
```bash
# Sprawdź status wszystkich danych
python setup_data.py --check

# Utwórz foldery dla danych i modeli
python setup_data.py
```

### Pobieranie zbiorów danych

Projekt używa dwóch zbiorów danych:

#### 1. SUN Attributes Dataset
```bash
# Wyświetl instrukcje pobierania
python setup_data.py --sun

# Alternatywnie ręcznie:
# 1. Pobierz: http://cs.mit.edu/~kadir/sunattributes/
# 2. Rozpakuj do: data/SUN/SUNAttributeDB/
```

Struktura po rozpakowaniu:
```
data/SUN/SUNAttributeDB/
├── attributes.mat
├── images.mat
├── attributeLabels_continuous.mat
└── images/
    ├── abbey/
    ├── airplane_cabin/
    └── ... (więcej folderów z obrazami)
```

#### 2. CUB (Caltech-UCSD Birds) Dataset
```bash
# Wyświetl instrukcje pobierania
python setup_data.py --cub

# Alternatywnie ręcznie:
# 1. Pobierz: http://www.vision.caltech.edu/datasets/
# 2. Rozpakuj do: data/CUB/
```

Struktura po rozpakowaniu:
```
data/CUB/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── attributes.txt
├── attributes/
│   ├── image_attribute_labels.txt
│   ├── class_attribute_labels_continuous.txt
│   └── certainties.txt
└── images/
    ├── 001.Black_footed_Albatross/
    ├── 002.Laysan_Albatross/
    └── ... (więcej gatunków ptaków)
```

#### 3. Modele pre-trained

Model ResNet18 (Places365) znajduje się w:
```
models/resnet18_places365.pth
```

Model CLIP (ViT-B/32) będzie pobrany automatycznie przy pierwszym uruchomieniu notebooka.

## Struktura projektu

```
.
├── data/                          # Dane (w .gitignore - zbyt duże!)
│   ├── CUB/
│   └── SUN/
├── models/                        # Pre-trained modele (w .gitignore)
│   └── resnet18_places365.pth
├── dataloaders.py                 # PyTorch Dataset klasy
├── data_loading.py                # Funkcje ładowania danych
├── utils.py                       # Utility funkcje
├── setup_data.py                  # Skrypt do zarządzania danymi
├── requirements.txt               # Zależności Python
├── datasets_tests.ipynb           # Analiza zbiorów danych
├── model_evaluation_sun.ipynb     # Ewaluacja modeli na SUN
└── README.md
```

## Dlaczego dane są w .gitignore?

**Dane i duże modele są ignorowane w Git** z kilku powodów:

1. **Rozmiar plików** — SUN i CUB to gigabajty danych. Git nie powinien przechowywać takich zbiorów.
2. **Przepustowość** — Każdy klonujący by musiał ściągać wszystkie dane (wolne).
3. **Dedykowane repozytoria** — Dane powinny być dostępne via linki do źródeł (университет, badacze).
4. **Best practices** — Kod → GitHub, Dane → osobne serwery (S3, Zenodo, itp.)

**Rozwiązanie**: Instrukcje pobierania w README + skrypt `setup_data.py` ✓

## Użycie

### Analiza zbiorów danych
```bash
jupyter notebook datasets_tests.ipynb
```

### Ewaluacja modeli
```bash
jupyter notebook model_evaluation_sun.ipynb
```

## Troubleshooting

### Problem: "FileNotFoundError: data/SUN/images/ not found"
**Rozwiązanie:**
```bash
python setup_data.py --check  # Sprawdzić status danych
python setup_data.py --sun    # Pobrać instrukcje
```

### Problem: CUDA out of memory
**Rozwiązanie:** Zmniejsz `batch_size` w kodzie:
```python
dataloader = create_sun_dataloader(
    ...,
    batch_size=8  # zmienić z 16/64
)
```

### Problem: CLIP model nie jest pobierany
**Rozwiązanie:** Upewnij się że masz internet i `clip` zainstalowany:
```bash
pip install --upgrade clip-by-openai
```

## Reference

- **SUN Attributes Database**: [http://cs.mit.edu/~kadir/sunattributes/](http://cs.mit.edu/~kadir/sunattributes/)
- **CUB Dataset**: [http://www.vision.caltech.edu/datasets/cub_200_2011/](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- **CLIP**: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
- **Places365**: [http://places2.csail.mit.edu/](http://places2.csail.mit.edu/)


**Created:** 2026-04-15  
**Last Updated:** 2026-04-15
