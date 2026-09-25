import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

# 1. Configuration du périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {device}")

# 2. Définition de la structure du Générateur (adaptée au checkpoint)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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

def lancer_inference(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Chargement du modèle entraîné
    model_path = "./saved_models/generator_ldct.pth"
    net_G = Generator().to(device)
    net_G.load_state_dict(torch.load(model_path, map_location=device))
    net_G.eval()
    print(f"Modèle chargé avec succès depuis {model_path}")

    # Transformations (redimensionnement et normalisation)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if not os.path.exists(input_dir):
        print(f"Erreur : Le dossier d'entrée '{input_dir}' est introuvable.")
        return

    fichiers = sorted(os.listdir(input_dir))
    if not fichiers:
        print(f"Aucune image trouvée dans {input_dir}")
        return

    print(f"Début du débruitage de {len(fichiers)} images dans {input_dir}...")
    
    with torch.no_grad():
        for f in fichiers:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.ima')):
                continue
                
            img_path = os.path.join(input_dir, f)
            
            try:
                # Lecture directe en tant qu'image (puisque vos fichiers d'entrée sont des .png)
                img = Image.open(img_path).convert('L')
                
                # Transformation et transfert sur le GPU
                tensor_img = transform(img).unsqueeze(0).to(device)
                
                # Passage dans le générateur (Débruitage LDCT -> FDCT)
                fake_fdct = net_G(tensor_img)
                
                # Post-traitement pour sauvegarder l'image générée
                fake_fdct = fake_fdct.squeeze(0).cpu()
                fake_fdct = (fake_fdct * 0.5 + 0.5) * 255 
                fake_fdct = fake_fdct.byte().numpy()[0]
                
                output_img = Image.fromarray(fake_fdct)
                
                # On nettoie le nom de fichier si l'extension y est déjà en double, ou on sauvegarde proprement
                base_name = os.path.splitext(f)[0]
                output_path = os.path.join(output_dir, f"denoised_{base_name}.png")
                output_img.save(output_path)
                
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {f} : {e}")

    print(f"Débruitage terminé ! Les images propres sont enregistrées dans '{output_dir}'.")

if __name__ == '__main__':
    lancer_inference("./dataset_final_prêt/test/low_dose", "./results_denoised/low_dose")
    lancer_inference("./dataset_final_prêt/test/full_dose","./results_denoised/full_dose")