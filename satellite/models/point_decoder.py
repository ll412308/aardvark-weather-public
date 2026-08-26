import torch
from torch import nn


class PointDecoder(nn.Module):
    """Decode latent features conditioned on known query metadata."""

    def __init__(self, latent_dim=128, metadata_dim=32, n_channels=15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + metadata_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, n_channels),
        )

    def forward(self, latent, metadata):
        return self.mlp(torch.cat([latent, metadata], dim=-1))
