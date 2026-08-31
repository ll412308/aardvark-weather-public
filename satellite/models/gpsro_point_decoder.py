from torch import nn
import torch
from .gpsro_point_encoder import GPSROMetadataEncoder, build_mlp


class GPSROPointDecoder(nn.Module):
    """Decode standardized log-refractivity at arbitrary 3-D query points."""

    def __init__(self, latent_dim=64, metadata_dim=32,
                 num_satellite_embeddings=256, embedding_dim=16,
                 metadata_hidden_dims=(64,), decoder_hidden_dims=(64, 64)):
        super().__init__()
        self.metadata = GPSROMetadataEncoder(
            metadata_dim, num_satellite_embeddings, embedding_dim,
            include_delta_time=False,
            hidden_dims=metadata_hidden_dims,
        )
        self.mlp = build_mlp(
            latent_dim + metadata_dim,
            decoder_hidden_dims,
            1,
        )

    def forward(self, query_latent, satellite_id, is_land, sample_time):
        metadata = self.metadata(
            satellite_id=satellite_id,
            is_land=is_land,
            sample_time=sample_time,
        )
        return self.mlp(torch.cat([query_latent, metadata], dim=-1))
