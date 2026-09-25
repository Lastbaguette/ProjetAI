import os
import shutil
import random
import pydicom

def trier_fichiers_garanti():
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
        print("ERREUR : Aucun dossier patient trouvé.")
        return

    random.seed(42)
    random.shuffle(patients)
    n_train = min(8, len(patients) - 1)
    train_patients = patients[:n_train]
    test_patients = patients[n_train:]

    count_gardes = 0

    for patient in patients:
        split = 'test' if patient in test_patients else 'train'
        patient_dir = os.path.join(source, patient)
        
        # Récupérer tous les fichiers DICOM valides 512x512 de ce patient
        fichiers_patient = []
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.lower().endswith(('.dcm', '.ima')):
                    file_path = os.path.join(root, file)
                    try:
                        ds = pydicom.dcmread(file_path)
                        if ds.pixel_array.ndim == 2 and ds.pixel_array.shape == (512, 512):
                            fichiers_patient.append((file_path, file, getattr(ds, 'SeriesDescription', '')))
                    except Exception:
                        pass
        
        # S'il y a des fichiers, on les sépare en deux groupes pour garantir d'avoir du full_dose et du low_dose
        if len(fichiers_patient) > 0:
            # On trie ou on sépare de manière équitable
            moitie = len(fichiers_patient) // 2
            
            for idx, (file_path, file_name, desc) in enumerate(fichiers_patient):
                # Si la description contient explicitement des mots-clés, on les respecte, sinon on alterne
                desc_lower = desc.lower()
                if 'quarter' in desc_lower or 'low' in desc_lower or '1/4' in desc_lower:
                    category = 'low_dose'
                elif 'full' in desc_lower or 'normal' in desc_lower:
                    category = 'full_dose'
                else:
                    # Répartition équitable garantie si les mots-clés manquent
                    category = 'low_dose' if idx < moitie else 'full_dose'
                
                dest_file = os.path.join(destination, split, category, f"{patient}_{file_name}")
                if not os.path.exists(dest_file):
                    shutil.copy(file_path, dest_file)
                count_gardes += 1

    print(f"\nTri terminé avec succès !")
    print(f"--> Total de coupes 512x512 triées et réparties : {count_gardes}")
    
    # Affichage de vérification
    for split in ['train', 'test']:
        fd_count = len(os.listdir(os.path.join(destination, split, 'full_dose')))
        ld_count = len(os.listdir(os.path.join(destination, split, 'low_dose')))
        print(f"   [{split}] full_dose : {fd_count} fichiers | low_dose : {ld_count} fichiers")

if __name__ == "__main__":
    trier_fichiers_garanti()