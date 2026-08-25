from dataclasses import dataclass


@dataclass
class BAMUAConfig:
    n_channels: int = 15
    point_dim: int = 128
    latent_dim: int = 128
    metadata_dim: int = 32
    satellite_embedding_dim: int = 16
    # Raw categorical IDs are hashed into this embedding table. Increase this if needed.
    num_satellite_embeddings: int = 256
    grid_resolution_deg: float = 2.0
    local_radius: int = 1
    init_lengthscale_deg: float = 2.0
    n_context: int = 65_536
    n_target: int = 16_384
    target_overlap: float = 0.5
    eps: float = 1.0e-6

    @property
    def grid_height(self):
        return round(180.0 / self.grid_resolution_deg) + 1

    @property
    def grid_width(self):
        return round(360.0 / self.grid_resolution_deg)
