import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 0. CONFIGURATION & MATÉRIEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./saved_models", exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 64
LR = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 5.0
LAMBDA_GP = 10.0

# ==========================================
# 1. ARCHITECTURES COMMUNES (GÉNÉRATEUR & CRITIQUE)
# ==========================================
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

class Discriminator(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. DATASET NON APPARIÉ
# ==========================================
class UnpairedCTDataset(Dataset):
    def __init__(self, low_dose_dir, full_dose_dir, transform=None):
        self.ld_paths = [os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith(('.png', '.jpg'))]
        self.fd_paths = [os.path.join(full_dose_dir, f) for f in os.listdir(full_dose_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return max(len(self.ld_paths), len(self.fd_paths))

    def __getitem__(self, idx):
        ld_img = Image.open(self.ld_paths[idx % len(self.ld_paths)]).convert('L')
        fd_img = Image.open(self.fd_paths[idx % len(self.fd_paths)]).convert('L')
        if self.transform:
            ld_img = self.transform(ld_img)
            fd_img = self.transform(fd_img)
        return ld_img, fd_img

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = UnpairedCTDataset("./dataset_final_prêt/train/low_dose", "./dataset_final_prêt/train/full_dose", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 3. CALCUL DE LA PÉNALITÉ DE GRADIENT (WGAN-GP)
# ==========================================
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake_target = torch.ones(d_interpolates.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_target,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ==========================================
# 4. FONCTIONS D'ENTRAÎNEMENT SPÉCIFIQUES
# ==========================================

def train_gan_standard():
    print("\n--- 1/4 Entraînement GAN Standard (Supervisé) ---")
    G = SimpleGenerator().to(device)
    D = Discriminator(use_sigmoid=True).to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    for epoch in range(EPOCHS):
        for ld, fd in dataloader:
            ld, fd = ld.to(device), fd.to(device)

            # Entraînement D
            opt_D.zero_grad()
            pred_real = D(fd)
            loss_d_real = criterion_bce(pred_real, torch.ones_like(pred_real))
            fake_fd = G(ld)
            pred_fake = D(fake_fd.detach())
            loss_d_fake = criterion_bce(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_d_real + loss_d_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # Entraînement G (Adversarial + reconstruction L1)
            opt_G.zero_grad()
            pred_fake = D(fake_fd)
            loss_adv = criterion_bce(pred_fake, torch.ones_like(pred_fake))
            loss_pixel = criterion_l1(fake_fd, fd)
            loss_G = loss_adv + 10.0 * loss_pixel
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] achevée.")
    torch.save(G.state_dict(), "./saved_models/generator_gan.pth")
    print("-> Sauvegardé : ./saved_models/generator_gan.pth")

def train_cyclegan_family(variant="cyclegan"):
    """Entraîne CycleGAN ou IdentityGAN selon le paramètre variant."""
    print(f"\n--- Entraînement {variant.upper()} ---")
    G_AB = SimpleGenerator().to(device)
    G_BA = SimpleGenerator().to(device)
    D_A = Discriminator(use_sigmoid=True).to(device)
    D_B = Discriminator(use_sigmoid=True).to(device)

    opt_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    for epoch in range(EPOCHS):
        for ld, fd in dataloader:
            ld, fd = ld.to(device), fd.to(device)

            # Entraînement G_AB et G_BA
            opt_G.zero_grad()
            fake_B = G_AB(ld)
            fake_A = G_BA(fd)

            loss_gan_ab = criterion_gan(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            loss_gan_ba = criterion_gan(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            loss_cycle = (criterion_cycle(G_BA(fake_B), ld) + criterion_cycle(G_AB(fake_A), fd)) * LAMBDA_CYCLE

            loss_G = loss_gan_ab + loss_gan_ba + loss_cycle

            if variant == "identitygan":
                # Ajout de la contrainte d'identité
                loss_id_A = criterion_identity(G_BA(ld), ld)
                loss_id_B = criterion_identity(G_AB(fd), fd)
                loss_G += (loss_id_A + loss_id_B) * LAMBDA_IDENTITY

            loss_G.backward()
            opt_G.step()

            # Entraînement D_A
            opt_D_A.zero_grad()
            loss_d_a = 0.5 * (criterion_gan(D_A(ld), torch.ones_like(D_A(ld))) +
                              criterion_gan(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach()))))
            loss_d_a.backward()
            opt_D_A.step()

            # Entraînement D_B
            opt_D_B.zero_grad()
            loss_d_b = 0.5 * (criterion_gan(D_B(fd), torch.ones_like(D_B(fd))) +
                              criterion_gan(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach()))))
            loss_d_b.backward()
            opt_D_B.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] achevée.")

    save_path = f"./saved_models/generator_{variant}.pth"
    torch.save(G_AB.state_dict(), save_path)
    print(f"-> Sauvegardé : {save_path}")

def train_gancircle():
    print("\n--- 4/4 Entraînement GAN-CIRCLE (WGAN-GP + Cycle + Identité) ---")
    G_AB = SimpleGenerator().to(device)
    G_BA = SimpleGenerator().to(device)
    # WGAN-GP n'utilise pas de Sigmoid
    D_A = Discriminator(use_sigmoid=False).to(device)
    D_B = Discriminator(use_sigmoid=False).to(device)

    opt_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.9))
    opt_D = optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=LR, betas=(0.5, 0.9))

    criterion_l1 = nn.L1Loss()

    for epoch in range(EPOCHS):
        for ld, fd in dataloader:
            ld, fd = ld.to(device), fd.to(device)

            # 1. Mise à jour des Critiques (WGAN-GP)
            opt_D.zero_grad()
            fake_B = G_AB(ld).detach()
            fake_A = G_BA(fd).detach()

            gp_A = compute_gradient_penalty(D_A, ld, fake_A)
            gp_B = compute_gradient_penalty(D_B, fd, fake_B)

            # Wasserstein loss = D(fake) - D(real) + GP
            loss_D_A = D_A(fake_A).mean() - D_A(ld).mean() + LAMBDA_GP * gp_A
            loss_D_B = D_B(fake_B).mean() - D_B(fd).mean() + LAMBDA_GP * gp_B
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            # 2. Mise à jour des Générateurs
            opt_G.zero_grad()
            gen_fake_B = G_AB(ld)
            gen_fake_A = G_BA(fd)

            # Perte Wasserstein générateurs
            loss_wgan = - D_B(gen_fake_B).mean() - D_A(gen_fake_A).mean()

            # Cycle + Identité
            loss_cycle = (criterion_l1(G_BA(gen_fake_B), ld) + criterion_l1(G_AB(gen_fake_A), fd)) * LAMBDA_CYCLE
            loss_id = (criterion_l1(G_BA(ld), ld) + criterion_l1(G_AB(fd), fd)) * LAMBDA_IDENTITY

            loss_G = loss_wgan + loss_cycle + loss_id
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] achevée.")

    torch.save(G_AB.state_dict(), "./saved_models/generator_gancircle.pth")
    print("-> Sauvegardé : ./saved_models/generator_gancircle.pth")

# ==========================================
# 5. EXÉCUTION COMPLÈTE
# ==========================================
if __name__ == "__main__":
    train_gan_standard()
    train_cyclegan_family("cyclegan")
    train_cyclegan_family("identitygan")
    train_gancircle()
    print("\nTous les modèles ont été entraînés et exportés dans ./saved_models/")