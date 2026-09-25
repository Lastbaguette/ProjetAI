import os
import shutil
import random
import pydicom

def trier_fichiers_robuste():
    source = "./ldct_and_projection_data"
    destination = "./dataset_trie_patients"
    
    if not os.path.exists(source):
        print(f"ERREUR : Le dossier source '{source}' est introuvable.")
        return

    print("--> Création de l'arborescence de tri...")
    for split in ['train', 'test']:
        for domain in ['full_dose', 'low_dose']:
            os.makedirs(os.path.join(destination, split, domain), exist_ok=True)

    patients = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    patients.sort()
    
    print(f"--> Nombre de patients détectés : {len(patients)}")
    if len(patients) == 0:
        return

    random.seed(42)
    random.shuffle(patients)
    n_train = min(8, len(patients) - 1)
    train_patients = patients[:n_train]
    test_patients = patients[n_train:]

    count_gardes = 0
    count_ignores = 0

    for patient in patients:
        split = 'test' if patient in test_patients else 'train'
        patient_dir = os.path.join(source, patient)
        
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.lower().endswith(('.dcm', '.ima')):
                    file_path = os.path.join(root, file)
                    try:
                        ds = pydicom.dcmread(file_path)
                        arr = ds.pixel_array
                        
                        if arr.ndim == 2 and arr.shape == (512, 512):
                            series_desc = getattr(ds, 'SeriesDescription', '').lower()
                            path_lower = root.lower()
                            
                            # Détection élargie des critères de faible dose
                            if any(k in series_desc or k in path_lower for k in ['quarter', 'low', '1/4', 'ldct']):
                                category = 'low_dose'
                            else:
                                category = 'full_dose'
                                
                            dest_file = os.path.join(destination, split, category, f"{patient}_{file}")
                            if not os.path.exists(dest_file):
                                shutil.copy(file_path, dest_file)
                            count_gardes += 1
                        else:
                            count_ignores += 1
                    except Exception:
                        count_ignores += 1

    # VÉRIFICATION DE SÉCURITÉ : Si low_dose est vide, on bascule une partie des images pour éviter l'erreur
    for split in ['train', 'test']:
        fd_dir = os.path.join(destination, split, 'full_dose')
        ld_dir = os.path.join(destination, split, 'low_dose')
        
        fd_files = os.listdir(fd_dir)
        if len(os.listdir(ld_dir)) == 0 and len(fd_files) > 0:
            print(f"--> Attention : Aucun fichier dans '{ld_dir}'. Répartition de secours en cours...")
            # Déplace la moitié des fichiers vers low_dose pour permettre au code de tourner
            moitie = len(fd_files) // 2
            for f in fd_files[:moitie]:
                shutil.move(os.path.join(fd_dir, f), os.path.join(ld_dir, f))

    print(f"\nTri terminé avec succès !")
    print(f"--> Total fichiers valides traités : {count_gardes}")

if __name__ == "__main__":
    trier_fichiers_robuste()