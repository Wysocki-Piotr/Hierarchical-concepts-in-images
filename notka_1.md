W pierwszym etapie projektu wybraliśmy odpowiednie zbiory danych: CUB (ptaki) oraz SUN (sceny i miejsca).

W zbiorze CUB badano empiryczne współwystępowanie cech wizualnych ptaków, jednak nie zauważyliśmy tam wielu ciekawych relacji.
Znacznie większy potencjał wykazał zbiór SUN, zawierający sceny wewnętrzne i zewnętrzne wraz z bogatymi atrybutami semantycznymi. Umożliwił on identyfikację potencjalnych, ciekawych zależności, takich jak np. camping → open area czy medical activity → enclosed area, a także analizę relacji semantycznych.
Plik w jakim to sprawdzano: `datasets_tests.ipynb`.

Dwa modele jakie badano to:

Model konwolucyjny: ResNet18 Places365 (pretrenowany na scenach),
Model SSL: CLIP ViT-B/32 (zarówno linear probe, jak i zero-shot).

Przeprowadziliśmyz wstępną ewaluację modeli na zbiorze SUN przy użyciu metryk takich jak Accuracy, F1-score i ROC-AUC. Plik w jakim to sprawdzano: `model_evaluation_sun.ipynb`.