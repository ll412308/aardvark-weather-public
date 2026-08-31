import torch
import torch.nn.functional as F
from torch import nn

from satellite.gpsro_config import GPSROConfig
from .gpsro_latent_processor import GPSROLatentProcessor
from .gpsro_point_decoder import GPSROPointDecoder
from .gpsro_point_encoder import GPSROPointEncoder
from .setconv import SetConv3DOffToOn, SetConv3DOnToOff


class GPSROAutoEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GPSROConfig()
        c = self.config
        self.point_encoder = GPSROPointEncoder(
            c.point_dim, c.metadata_dim,
            c.num_satellite_embeddings, c.satellite_embedding_dim,
            value_hidden_dims=c.value_encoder_hidden_dims,
            metadata_hidden_dims=c.metadata_hidden_dims,
            fusion_hidden_dims=c.fusion_hidden_dims,
        )
        setconv_args = dict(
            grid_resolution_deg=c.grid_resolution_deg,
            vertical_min_m=c.vertical_min_m,
            vertical_max_m=c.vertical_max_m,
            vertical_resolution_m=c.vertical_resolution_m,
            horizontal_radius=c.horizontal_radius,
            vertical_radius=c.vertical_radius,
            init_horizontal_lengthscale_km=c.init_horizontal_lengthscale_km,
            init_vertical_lengthscale_m=c.init_vertical_lengthscale_m,
            eps=c.eps,
        )
        self.point_to_grid = SetConv3DOffToOn(**setconv_args)
        self.latent_processor = GPSROLatentProcessor(
            in_dim=c.point_dim,
            latent_dim=c.latent_dim,
            grid_depth=c.grid_depth,
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
        self.grid_to_point = SetConv3DOnToOff(**setconv_args)
        self.point_decoder = GPSROPointDecoder(
            c.latent_dim, c.metadata_dim,
            c.num_satellite_embeddings, c.satellite_embedding_dim,
            metadata_hidden_dims=c.metadata_hidden_dims,
            decoder_hidden_dims=c.decoder_hidden_dims,
        )

    def encode(self, refractivity, valid, lon, lat, height, satellite_id,
               is_land, obs_time, sample_time, chunk_size=None):
        chunk_size = int(chunk_size or self.config.setconv_chunk_size)
        latent_sum = density_sum = None
        for start in range(0, refractivity.shape[1], chunk_size):
            end = min(start + chunk_size, refractivity.shape[1])
            feature = self.point_encoder(
                refractivity[:, start:end], valid[:, start:end],
                satellite_id[:, start:end], is_land[:, start:end],
                obs_time[:, start:end], sample_time,
            )
            # print('feature shape:', feature.shape)
            chunk_sum, chunk_density = self.point_to_grid.aggregate(
                feature, lon[:, start:end], lat[:, start:end],
                height[:, start:end],
                vertical_type="altitude", vertical_unit="m",
                point_mask=valid[:, start:end, 0],
            )
            # print('chunk_sum shape:', chunk_sum.shape)
            latent_sum = chunk_sum if latent_sum is None else latent_sum + chunk_sum
            density_sum = (
                chunk_density if density_sum is None
                else density_sum + chunk_density
            )
        latent = latent_sum / density_sum.clamp_min(self.config.eps)
        # print('latent shape:', latent.shape)
        return self.latent_processor(latent), density_sum

    def decode(self, latent, lon, lat, height, satellite_id, is_land,
               sample_time, chunk_size=None):
        chunk_size = int(chunk_size or self.config.decode_chunk_size)
        native_size = (self.config.grid_height, self.config.grid_width)
        if latent.shape[-2:] != native_size:
            # Fusion may operate on an explicitly aligned instrument grid
            # (for example GPSRO exported from 1 degree to BAMUA's 2 degrees).
            # Restore the AE's native lon/lat grid before rebuilding height.
            latent = F.interpolate(
                latent, size=native_size, mode="bilinear", align_corners=False
            )
        latent_3d = self.latent_processor.restore_3d(latent)
        # print('latent_3d shape:', latent_3d.shape)
        outputs = []
        for start in range(0, lon.shape[1], chunk_size):
            end = min(start + chunk_size, lon.shape[1])
            query_latent = self.grid_to_point(
                latent_3d, lon[:, start:end], lat[:, start:end],
                height[:, start:end],
                vertical_type="altitude", vertical_unit="m",
            )
            outputs.append(self.point_decoder(
                query_latent, satellite_id[:, start:end],
                is_land[:, start:end], sample_time,
            ))
        return torch.cat(outputs, dim=1)

    def forward(self, context, target):
        latent, density = self.encode(**context)
        # print('encoded latent shape:', latent.shape)
        prediction = self.decode(
            latent=latent,
            lon=target["lon"], lat=target["lat"],
            height=target["height"],
            satellite_id=target["satellite_id"],
            is_land=target["is_land"],
            sample_time=target["sample_time"],
        )
        # print('prediction shape:', prediction.shape)
        return prediction, latent, density


def _synthetic_test():
    torch.manual_seed(0)
    config = GPSROConfig(
        grid_resolution_deg=30.0,
        vertical_min_m=0.0,
        vertical_max_m=10_000.0,
        vertical_resolution_m=5_000.0,
        point_dim=16, latent_dim=16, metadata_dim=8,
        latent_processor_dim=32, latent_processor_depth=1,
        latent_processor_heads=4, latent_window_size=3,
        setconv_chunk_size=50,
        decode_chunk_size=20,
    )
    model = GPSROAutoEncoder(config)
    batch, n_context, n_target = 1, 100, 20
    sample_time = torch.tensor([1_700_000_000_000_000_000])
    context = {
        "refractivity": torch.randn(batch, n_context, 1),
        "valid": torch.ones(batch, n_context, 1, dtype=torch.bool),
        "lon": torch.rand(batch, n_context) * 360 - 180,
        "lat": torch.rand(batch, n_context) * 180 - 90,
        "height": torch.rand(batch, n_context) * 10_000,
        "satellite_id": torch.full((batch, n_context), 4, dtype=torch.long),
        "is_land": torch.rand(batch, n_context) > 0.5,
        "obs_time": sample_time[:, None] + torch.randint(
            -10_800_000_000_000, 10_800_000_000_000,
            (batch, n_context), dtype=torch.long,
        ),
        "sample_time": sample_time,
    }
    target = {
        "lon": torch.rand(batch, n_target) * 360 - 180,
        "lat": torch.rand(batch, n_target) * 180 - 90,
        "height": torch.rand(batch, n_target) * 10_000,
        "satellite_id": torch.full((batch, n_target), 4, dtype=torch.long),
        "is_land": torch.rand(batch, n_target) > 0.5,
        "obs_time": sample_time[:, None].expand(batch, n_target),
        "sample_time": sample_time,
    }
    prediction, latent, density = model(context, target)
    coarse_latent = F.interpolate(
        latent, scale_factor=0.5, mode="bilinear", align_corners=False
    )
    coarse_prediction = model.decode(
        latent=coarse_latent,
        lon=target["lon"], lat=target["lat"], height=target["height"],
        satellite_id=target["satellite_id"], is_land=target["is_land"],
        sample_time=target["sample_time"],
    )
    print(f"prediction: {tuple(prediction.shape)}")
    print(f"latent:     {tuple(latent.shape)}")
    print(f"density:    {tuple(density.shape)}")
    print("restored:   ", tuple(model.latent_processor.restore_3d(latent).shape))
    print(f"coarse decode: {tuple(coarse_prediction.shape)}")
    prediction.square().mean().backward()
    print("offtoon horizontal grad:", model.point_to_grid.log_horizontal_lengthscale.grad)
    print("offtoon vertical grad:  ", model.point_to_grid.log_vertical_lengthscale.grad)
    print("ontooff horizontal grad:", model.grid_to_point.log_horizontal_lengthscale.grad)
    print("ontooff vertical grad:  ", model.grid_to_point.log_vertical_lengthscale.grad)


if __name__ == "__main__":
    _synthetic_test()
