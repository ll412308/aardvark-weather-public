import torch
from torch import nn

from .metadata_encoder import MetadataEncoder


class PointEncoder(nn.Module):
    def __init__(self, n_channels=15, point_dim=128, metadata_dim=32,
                 num_satellite_embeddings=256, embedding_dim=16,
                 include_angles=True):
        super().__init__()
        self.radiance = nn.Sequential(
            nn.Linear(2 * n_channels, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.metadata = MetadataEncoder(metadata_dim, num_satellite_embeddings,
                                        embedding_dim, include_delta_time=True,
                                        include_angles=include_angles)
        self.fusion = nn.Sequential(
            nn.Linear(128 + metadata_dim, 128), nn.SiLU(), nn.Linear(128, point_dim)
        )

    def forward(self, bt, valid, **metadata):
        # Invalid values are zero-filled, while the explicit mask tells the MLP why.
        # Also treat an unexpected NaN/Inf as invalid; torch.where(valid, bt, 0)
        # alone would retain a non-finite value when its saved valid flag is True.
        finite_valid = valid & torch.isfinite(bt)
        bt = torch.where(finite_valid, bt, torch.zeros_like(bt))
        radiance = self.radiance(
            torch.cat([bt, finite_valid.to(bt.dtype)], dim=-1)
        )  # B, N, 128
        metadata_output = self.metadata(**metadata)  # B, N, metadata_dim
        return self.fusion(torch.cat([radiance, metadata_output], dim=-1))  # B, N, point_dim
    
