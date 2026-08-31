from dataclasses import dataclass


@dataclass
class GPSROConfig:
    point_dim: int = 64
    latent_dim: int = 64
    metadata_dim: int = 32
    value_encoder_hidden_dims: tuple = (64, 64)
    metadata_hidden_dims: tuple = (64,)
    fusion_hidden_dims: tuple = (64,)
    decoder_hidden_dims: tuple = (64, 64)
    satellite_embedding_dim: int = 16
    num_satellite_embeddings: int = 256
    grid_resolution_deg: float = 5.0
    vertical_min_m: float = -1_000.0
    vertical_max_m: float = 64_000.0
    vertical_resolution_m: float = 5_000.0
    horizontal_radius: int = 1
    vertical_radius: int = 1
    init_horizontal_lengthscale_km: float = 550.0
    init_vertical_lengthscale_m: float = 1_000.0
    use_latent_processor: bool = True
    latent_processor_dim: int = 128
    latent_patch_size: int = 0
    latent_patch_min_height: int = 64
    latent_patch_min_width: int = 128
    latent_processor_depth: int = 2
    latent_processor_heads: int = 4
    latent_window_size: int = 5
    setconv_chunk_size: int = 8_192
    decode_chunk_size: int = 8_192
    n_context: int = 65_536
    n_target: int = 16_384
    target_overlap: float = 0.5
    eps: float = 1.0e-6

    @property
    def grid_depth(self):
        return round(
            (self.vertical_max_m - self.vertical_min_m)
            / self.vertical_resolution_m
        ) + 1

    @property
    def grid_height(self):
        return round(180.0 / self.grid_resolution_deg) + 1

    @property
    def grid_width(self):
        return round(360.0 / self.grid_resolution_deg)
