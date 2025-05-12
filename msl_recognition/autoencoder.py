"""
Convolutional Autoencoder definition for feature compression and reconstruction.
"""
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 32 * 32),
            nn.Unflatten(1, (32, 32, 32)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out