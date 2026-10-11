import os, shutil, random, pydicom

def trier_dicom(source="./ldct_and_projection_data", dest="./dataset_trie_patients"):
    if not os.path.exists(source): return
    for s in ['train', 'test']:
        for d in ['full_dose', 'low_dose']:
            os.makedirs(os.path.join(dest, s, d), exist_ok=True)

    patients = sorted([d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))])
    random.seed(42); random.shuffle(patients)
    n_train = max(1, int(len(patients) * 0.8))

    for p in patients:
        split = 'train' if p in patients[:n_train] else 'test'
        for root, _, files in os.walk(os.path.join(source, p)):
            for f in files:
                if not f.lower().endswith(('.dcm', '.ima')): continue
                path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=False)
                    if ds.pixel_array.shape != (512, 512): continue
                    desc = getattr(ds, 'SeriesDescription', '').lower()
                    cat = 'low_dose' if any(k in desc or k in root.lower() for k in ['quarter', 'low', '1/4', 'ldct']) else 'full_dose'
                    shutil.copy(path, os.path.join(dest, split, cat, f"{p}_{f}"))
                except Exception:
                    pass
    print("Tri terminé.")

if __name__ == "__main__":
    trier_dicom()