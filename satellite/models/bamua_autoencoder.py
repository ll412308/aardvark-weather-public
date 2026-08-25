from torch import nn

from satellite.config import BAMUAConfig
from .latent_processor import LatentGridProcessor
from .metadata_encoder import MetadataEncoder
from .point_decoder import PointDecoder
from .point_encoder import PointEncoder
from .setconv import SetConvOffToOn, SetConvOnToOff


class BAMUAAutoEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or BAMUAConfig()
        c = self.config
        self.point_encoder = PointEncoder(c.n_channels, c.point_dim, c.metadata_dim,
                                          c.num_satellite_embeddings, c.satellite_embedding_dim)
        self.point_to_grid = SetConvOffToOn(c.grid_resolution_deg, c.local_radius,
                                             c.init_lengthscale_deg, c.eps)
        self.latent_processor = LatentGridProcessor(
            in_dim=c.point_dim,
            latent_dim=c.latent_dim,
            grid_height=c.grid_height,
            grid_width=c.grid_width,
            processor_dim=c.latent_processor_dim,
            patch_size=c.latent_patch_size,
            patch_min_height=c.latent_patch_min_height,
            patch_min_width=c.latent_patch_min_width,
            depth=c.latent_processor_depth,
            num_heads=c.latent_processor_heads,
            window_size=c.latent_window_size,
            enabled=c.use_latent_processor,
        )
        self.grid_to_point = SetConvOnToOff(c.grid_resolution_deg, c.local_radius,
                                            c.init_lengthscale_deg, c.eps)
        self.density_to_point = SetConvOnToOff(c.grid_resolution_deg, c.local_radius,
                                               c.init_lengthscale_deg, c.eps)
        self.query_metadata = MetadataEncoder(c.metadata_dim, c.num_satellite_embeddings,
                                              c.satellite_embedding_dim,
                                              include_delta_time=False,
                                              include_angles=False)
        self.point_decoder = PointDecoder(c.latent_dim, c.metadata_dim, c.n_channels)

    @staticmethod
    def _metadata(satellite_id, is_land, sample_time, obs_time=None,
                  zenith=None, azimuth=None):
        return dict(satellite_id=satellite_id, is_land=is_land, obs_time=obs_time,
                    sample_time=sample_time, zenith=zenith, azimuth=azimuth)

    def encode(self, bt, valid, lon, lat, satellite_id, is_land, obs_time,
               sample_time, zenith, azimuth):
        feat = self.point_encoder(bt, valid, **self._metadata(
            satellite_id=satellite_id, is_land=is_land, sample_time=sample_time,
            obs_time=obs_time, zenith=zenith, azimuth=azimuth))
        latent, density = self.point_to_grid(feat, lon, lat)
        return self.latent_processor(latent), density

    def encode_chunked(self, bt, valid, lon, lat, satellite_id, is_land,
                       obs_time, sample_time, zenith, azimuth, chunk_size):
        """Encode a very large point set without making one huge local tensor."""
        n_points = bt.shape[1]
        latent_sum = None
        density_sum = None
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            feat = self.point_encoder(
                bt[:, start:end], valid[:, start:end],
                **self._metadata(
                    satellite_id=satellite_id[:, start:end],
                    is_land=is_land[:, start:end],
                    sample_time=sample_time,
                    obs_time=obs_time[:, start:end],
                    zenith=zenith[:, start:end],
                    azimuth=azimuth[:, start:end],
                ),
            )
            chunk_sum, chunk_density = self.point_to_grid.aggregate(
                feat, lon[:, start:end], lat[:, start:end]
            )
            latent_sum = chunk_sum if latent_sum is None else latent_sum + chunk_sum
            density_sum = (
                chunk_density if density_sum is None
                else density_sum + chunk_density
            )
        latent = latent_sum / density_sum.clamp_min(self.config.eps)
        return self.latent_processor(latent), density_sum

    def decode(self, latent, density, lon, lat, satellite_id, is_land,
               sample_time, obs_time=None, zenith=None, azimuth=None):
        query_latent = self.grid_to_point(latent, lon, lat)
        query_density = self.density_to_point(density, lon, lat)
        query_meta = self.query_metadata(**self._metadata(
            satellite_id=satellite_id, is_land=is_land, sample_time=sample_time))
        return self.point_decoder(query_latent, query_density, query_meta)

    def forward(self, context, target):
        latent, density = self.encode(**context)
        pred = self.decode(latent=latent, density=density, **target)
        return pred, latent, density
