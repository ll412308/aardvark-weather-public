import torch
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
                                          c.num_satellite_embeddings,
                                          c.satellite_embedding_dim,
                                          include_angles=c.include_angles)
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
        # Decoder metadata only contains quantities known at generation time:
        # satellite identity, sample time, and the land/sea flag.
        self.query_metadata = MetadataEncoder(
            c.metadata_dim,
            c.num_satellite_embeddings,
            c.satellite_embedding_dim,
            include_delta_time=False,
            include_angles=False,
        )
        self.point_decoder = PointDecoder(
            c.latent_dim, c.metadata_dim, c.n_channels
        )

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

    def decode(self, latent, lon, lat, satellite_id, is_land, sample_time):
        """Predict BT using latent features and metadata known at query time."""
        query_latent = self.grid_to_point(latent, lon, lat)
        query_meta = self.query_metadata(
            satellite_id=satellite_id,
            is_land=is_land,
            obs_time=None,
            sample_time=sample_time,
        )
        return self.point_decoder(query_latent, query_meta)

    def forward(self, context, target):
        latent, density = self.encode(**context)
        pred = self.decode(
            latent=latent,
            lon=target["lon"],
            lat=target["lat"],
            satellite_id=target["satellite_id"],
            is_land=target["is_land"],
            sample_time=target["sample_time"],
        )
        return pred, latent, density


def _print_shape(name, value):
    """Print a tensor shape in a compact form for the local smoke test."""
    print(f"{name:<28} shape={tuple(value.shape)} dtype={value.dtype}")


def _synthetic_shape_test():
    """Run one small synthetic sample through every stage of the BAMUA AE."""
    torch.manual_seed(0)

    config = BAMUAConfig()
    model = BAMUAAutoEncoder(config).eval()

    batch_size = 1
    n_context = 100
    n_target = 20
    n_channels = config.n_channels
    sample_time = torch.tensor(
        [1_700_000_000_000_000_000], dtype=torch.long
    )

    context_valid = torch.rand(batch_size, n_context, n_channels) > 0.1
    context_bt = torch.randn(batch_size, n_context, n_channels)
    # The real Zarr also uses zero filling, but validity is always carried
    # separately because zero is a valid z-score value.
    context_bt = torch.where(
        context_valid, context_bt, torch.zeros_like(context_bt)
    )
    context = {
        "bt": context_bt,
        "valid": context_valid,
        "lon": torch.rand(batch_size, n_context) * 360.0 - 180.0,
        "lat": torch.rand(batch_size, n_context) * 180.0 - 90.0,
        "satellite_id": torch.full(
            (batch_size, n_context), 1, dtype=torch.long
        ),
        "is_land": torch.rand(batch_size, n_context) > 0.5,
        # The observations lie within +/-3 hours of the 6-hour sample time.
        "obs_time": sample_time[:, None] + torch.randint(
            -3 * 3_600_000_000_000,
            3 * 3_600_000_000_000 + 1,
            (batch_size, n_context),
            dtype=torch.long,
        ),
        "sample_time": sample_time,
        "zenith": torch.rand(batch_size, n_context) * 70.0,
        "azimuth": torch.rand(batch_size, n_context) * 360.0,
    }
    target = {
        "lon": torch.rand(batch_size, n_target) * 360.0 - 180.0,
        "lat": torch.rand(batch_size, n_target) * 180.0 - 90.0,
        "satellite_id": torch.full(
            (batch_size, n_target), 1, dtype=torch.long
        ),
        "is_land": torch.rand(batch_size, n_target) > 0.5,
        "sample_time": sample_time,
    }

    print("=== Synthetic 1BAMUA AutoEncoder shape test ===")
    print(
        f"grid: H={config.grid_height}, W={config.grid_width}, "
        f"resolution={config.grid_resolution_deg} deg"
    )
    print(
        f"local SetConv neighbours per point="
        f"{(2 * config.local_radius + 1) ** 2}, "
        f"lengthscale={model.point_to_grid.lengthscale.item():.4f} deg"
    )
    print(f"encoder include_angles={config.include_angles}")
    _print_shape("context bt", context["bt"])
    _print_shape("context valid", context["valid"])
    _print_shape("context lon/lat", context["lon"])
    _print_shape("context sample_time", context["sample_time"])
    _print_shape("context zenith", context["zenith"])
    _print_shape("context azimuth", context["azimuth"])
    _print_shape("target lon/lat", target["lon"])
    print(
        f"zenith range              min={context['zenith'].min().item():.2f} "
        f"max={context['zenith'].max().item():.2f} deg"
    )
    print(
        f"azimuth range             min={context['azimuth'].min().item():.2f} "
        f"max={context['azimuth'].max().item():.2f} deg"
    )
    print(f"invalid BT entries          count={(~context_valid).sum().item()}")

    with torch.no_grad():
        point_feature = model.point_encoder(
            context["bt"],
            context["valid"],
            **model._metadata(
                satellite_id=context["satellite_id"],
                is_land=context["is_land"],
                sample_time=context["sample_time"],
                obs_time=context["obs_time"],
                zenith=context["zenith"],
                azimuth=context["azimuth"],
            ),
        )
        latent_sum, density = model.point_to_grid.aggregate(
            point_feature, context["lon"], context["lat"]
        )
        latent_before_processor = latent_sum / density.clamp_min(config.eps)
        latent = model.latent_processor(latent_before_processor)
        query_latent = model.grid_to_point(
            latent, target["lon"], target["lat"]
        )
        query_metadata = model.query_metadata(
            satellite_id=target["satellite_id"],
            is_land=target["is_land"],
            obs_time=None,
            sample_time=target["sample_time"],
        )
        prediction = model.point_decoder(query_latent, query_metadata)

    print("\n--- Step-by-step model shapes ---")
    _print_shape("point feature", point_feature)
    _print_shape("OffToOn weighted sum", latent_sum)
    _print_shape("OffToOn density", density)
    _print_shape("density-normalized grid", latent_before_processor)
    _print_shape("processed latent", latent)
    _print_shape("OnToOff query latent", query_latent)
    _print_shape("query metadata", query_metadata)
    _print_shape("predicted BT", prediction)
    print(
        f"density range              min={density.min().item():.6f} "
        f"max={density.max().item():.6f}"
    )
    print("\nExpected final shape: [B, N_target, 15]")
    print(f"Actual final shape:   {list(prediction.shape)}")


if __name__ == "__main__":
    _synthetic_shape_test()
