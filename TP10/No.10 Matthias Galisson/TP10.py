import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NOISE_STD = 0.3

loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64,
    shuffle=True
)

def add_gaussian_noise(images, mean=0.0, std=NOISE_STD):
    noise = torch.randn_like(images) * std + mean
    noisy_images = torch.clamp(images + noise, 0.0, 1.0)
    return noisy_images

def add_salt_and_pepper_noise(images, salt_prob=0.5, pepper_prob=0.5):
    noisy_images = images.clone()
    salt_mask = torch.rand_like(images) < salt_prob
    noisy_images[salt_mask] = 1.0
    pepper_mask = torch.rand_like(images) < pepper_prob
    noisy_images[pepper_mask] = 0.0
    return noisy_images

class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        return self.net(x)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())
    def forward(self, x):
        feat = self.enc(x)
        return self.dec(torch.cat([feat, x.expand_as(feat)], dim=1))

models = {
    'FCNN': FCNN().to(device),
    'Autoencoder': AE().to(device),
    'U-Net': UNet().to(device)
}

crit = nn.MSELoss()
for name, m in models.items():
    print(f"Entraînement de {name} (Bruit std={NOISE_STD})...")
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for x, _ in loader:
        x = x.to(device)
        noisy = add_gaussian_noise(x, std=NOISE_STD)
        loss = crit(m(noisy), x)
        opt.zero_grad()
        loss.backward()
        opt.step()

imgs, _ = next(iter(loader))
clean = imgs[:4].to(device)
noisy = add_gaussian_noise(clean, std=NOISE_STD)

cols = ['Bruitée', 'Clean'] + list(models.keys())
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

for i in range(4):
    axes[i, 0].imshow(noisy[i, 0].cpu(), cmap='gray')
    axes[i, 1].imshow(clean[i, 0].cpu(), cmap='gray')
    for j, (name, m) in enumerate(models.items()):
        with torch.no_grad():
            pred = m(noisy[i:i+1])[0, 0].cpu()
        axes[i, j + 2].imshow(pred, cmap='gray')

for ax, col in zip(axes[0], cols):
    ax.set_title(col, fontsize=11, fontweight='bold')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()