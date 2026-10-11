import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ========================================================
# 1. CONFIGURATION & DISPOSITIF
# ========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Dispositif d'inférence : {device}")

# ========================================================
# 2. ARCHITECTURE DU GÉNÉRATEUR
# ========================================================
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# ========================================================
# 3. CHARGEMENT DIRECT DES POIDS (.PTH)
# ========================================================
def charger_modele(chemin_poids):
    modele = SimpleGenerator().to(device)
    if os.path.exists(chemin_poids):
        poids = torch.load(chemin_poids, map_location=device, weights_only=True)
        if isinstance(poids, dict) and "state_dict" in poids:
            modele.load_state_dict(poids["state_dict"])
        else:
            modele.load_state_dict(poids)
        print(f"[OK] Chargé : {chemin_poids}")
    else:
        print(f"[ERREUR] Fichier introuvable : {chemin_poids}")
    modele.eval()
    return modele

# Les 12 variantes de modèles (sans doublons et avec CycleGAN 10/20)
modeles = {
    # 5 époques
    "GAN 5":          charger_modele(r".\saved_models\generator_gan-5epoque.pth"),
    "CycleGAN 5":     charger_modele(r".\saved_models\generator_cyclegan-5epoque.pth"),
    "IdentityGAN 5":  charger_modele(r".\saved_models\generator_identitygan-5epoque.pth"),
    "GAN-CIRCLE 5":   charger_modele(r".\saved_models\generator_gancircle-5epoque.pth"),
    
    # 10 époques
    "GAN 10":         charger_modele(r".\saved_models\generator_gan-10epoque.pth"),
    "CycleGAN 10":    charger_modele(r".\saved_models\generator_cyclegan-10epoque.pth"),
    "IdentityGAN 10": charger_modele(r".\saved_models\generator_identitygan-10epoque.pth"),
    "GAN-CIRCLE 10":  charger_modele(r".\saved_models\generator_gancircle-10epoque.pth"),
    
    # 20 époques
    "GAN 20":         charger_modele(r".\saved_models\generator_gan-20epoque.pth"),
    "CycleGAN 20":    charger_modele(r".\saved_models\generator_cyclegan-20epoque.pth"),
    "IdentityGAN 20": charger_modele(r".\saved_models\generator_identitygan-20epoque.pth"),
    "GAN-CIRCLE 20":  charger_modele(r".\saved_models\generator_gancircle-20epoque.pth"),
}

# ========================================================
# 4. MÉTRIQUES PSNR / SSIM (Formules complètes)
# ========================================================
def calcul_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(1.0 / np.sqrt(mse)))

def calcul_ssim(img1, img2):
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.mean((img1 - mu1) ** 2)
    sigma2_sq = np.mean((img2 - mu2) ** 2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerateur = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominateur = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(numerateur / denominateur)

# ========================================================
# 5. TEST ET SAUVEGARDE DES PLANCHES
# ========================================================
test_ld_dir = r".\dataset_final_prêt\test\low_dose"
test_fd_dir = r".\dataset_final_prêt\test\full_dose"
dossier_sortie = r".\resultats_comparaison_completes"
os.makedirs(dossier_sortie, exist_ok=True)

transform_in = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

fichiers = [f for f in os.listdir(test_ld_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.ima', '.bmp'))]

if not fichiers:
    print(f"Aucune image trouvée dans '{test_ld_dir}'.")
    exit()

fichiers_selectionnes = sorted(fichiers)[:4]
print(f"\n--> Comparaison lancée sur les 4 images : {fichiers_selectionnes}")

for idx_img, nom_fichier in enumerate(fichiers_selectionnes):
    chemin_ld = os.path.join(test_ld_dir, nom_fichier)
    chemin_fd = os.path.join(test_fd_dir, nom_fichier)
    
    img_ld_pil = Image.open(chemin_ld).convert('L')
    tensor_ld = transform_in(img_ld_pil).unsqueeze(0).to(device)
    
    a_verite_terrain = os.path.exists(chemin_fd)
    if a_verite_terrain:
        img_fd_pil = Image.open(chemin_fd).convert('L').resize((256, 256))
        img_fd_np = np.array(img_fd_pil).astype(np.float32) / 255.0

    predictions = {}
    with torch.no_grad():
        for nom_modele, net in modeles.items():
            pred = net(tensor_ld)
            pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
            predictions[nom_modele] = pred.squeeze().cpu().numpy()

    # Calcul dynamique : 1 input + 12 modèles + 1 ground truth optionnelle = 13 ou 14 colonnes
    nb_colonnes = 1 + len(predictions) + (1 if a_verite_terrain else 0)
    fig, axes = plt.subplots(1, nb_colonnes, figsize=(2.8 * nb_colonnes, 4.5))

    plt.suptitle(f"Évaluation comparative — Image : {nom_fichier}", fontsize=13, fontweight='bold', y=0.98)

    # 1. Colonne Entrée (Low-Dose)
    img_ld_np = np.array(img_ld_pil.resize((256, 256))).astype(np.float32) / 255.0
    axes[0].imshow(img_ld_np, cmap='gray')
    axes[0].set_title(f"Entrée : Low-Dose\n({nom_fichier[:18]}...)", fontsize=8, fontweight='bold')
    axes[0].axis('off')

    # 2. Colonnes des modèles
    col_idx = 1
    for nom_modele, img_pred in predictions.items():
        axes[col_idx].imshow(img_pred, cmap='gray')
        
        titre = nom_modele
        if a_verite_terrain:
            p_val = calcul_psnr(img_pred, img_fd_np)
            s_val = calcul_ssim(img_pred, img_fd_np)
            titre += f"\nPSNR: {p_val:.2f} dB\nSSIM: {s_val:.4f}"
        
        couleur_titre = 'darkgreen' if 'GAN-CIRCLE' in nom_modele else 'black'
        poids_texte = 'bold' if 'GAN-CIRCLE' in nom_modele else 'normal'
        axes[col_idx].set_title(titre, fontsize=7.5, color=couleur_titre, fontweight=poids_texte)
        axes[col_idx].axis('off')
        col_idx += 1

    # 3. Colonne Vérité Terrain (Full-Dose)
    if a_verite_terrain:
        axes[col_idx].imshow(img_fd_np, cmap='gray')
        axes[col_idx].set_title("Vérité terrain\n(Full-Dose)", fontsize=8, fontweight='bold', color='navy')
        axes[col_idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    base_nom = os.path.splitext(nom_fichier)[0]
    chemin_sauvegarde = os.path.join(dossier_sortie, f"comparaison_{idx_img+1}_{base_nom}.png")
    plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Planche {idx_img+1}/4 enregistrée : {chemin_sauvegarde}")

print(f"\n--> Terminé ! Les planches étiquetées sont disponibles dans : {dossier_sortie}")