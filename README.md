# Projet de débruitage LDCT

## Données à télécharger

1. Ouvrir le [portail Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/explore/).
2. Sélectionner la collection **LDCT-and-Projection-data** et choisir **10 patients**.
3. Télécharger le fichier manifest généré par le portail.
4. Installer ou mettre à jour l’outil de téléchargement :

   ```bash
   pip install --upgrade idc-index
   ```

5. Depuis le dossier contenant le manifest, télécharger les données :

   ```bash
   idc download <manifest filename>
   ```

## Installation

Installer les bibliothèques nécessaires au script :

```bash
pip install torch torchvision numpy pillow pydicom
```

## Tri des données

Depuis la racine du projet, lancer le tri des fichiers DICOM :

```bash
python tri.py
```

Les fichiers sont répartis dans `dataset_trie_patients/train` et `dataset_trie_patients/test`, chacun séparé en `low_dose` et `full_dose`.

## Entraînement

Le tri produit des fichiers DICOM, tandis que l’entraînement attend des images PNG ou JPG dans `dataset_final_prêt/train/low_dose` et `dataset_final_prêt/train/full_dose`. Il faut donc préparer ces dossiers d’images avant de lancer l’entraînement :

```bash
python "train _allgan.py"
```

Le script entraîne plusieurs modèles et les enregistre dans `saved_models`.

## Lancement de l’inférence

Depuis la racine du projet, lancer :

```bash
python test_final.py
```

L’inférence utilise les dossiers `dataset_final_prêt/test/low_dose` et `dataset_final_prêt/test/full_dose`, ainsi que le modèle `saved_models/generator_ldct.pth`. Ce modèle n’est pas créé par `train _allgan.py`, qui enregistre ses modèles sous d’autres noms.
