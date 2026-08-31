import torch
from torch import nn


def build_mlp(input_dim, hidden_dims, output_dim, activate_output=False):
    """Build a configurable SiLU MLP while keeping the model definition small."""
    layers = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.SiLU()])
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(output_dim)))
    if activate_output:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class GPSROMetadataEncoder(nn.Module):
    """Satellite and GraphCast-style calendar features for GPSRO points."""

    def __init__(self, output_dim=32, num_satellite_embeddings=256,
                 embedding_dim=16, include_delta_time=True,
                 time_window_hours=6.0, hidden_dims=(64,)):
        super().__init__()
        self.num_satellite_embeddings = int(num_satellite_embeddings)
        self.include_delta_time = bool(include_delta_time)
        self.time_window_hours = float(time_window_hours)
        self.satellite_embedding = nn.Embedding(
            self.num_satellite_embeddings, embedding_dim
        )
        # is_land is a known binary property of the query coordinate. It is
        # kept as 0/1 instead of using another embedding table.
        numeric_dim = 12 + 1 + (5 if self.include_delta_time else 0)
        self.mlp = build_mlp(
            embedding_dim + numeric_dim, hidden_dims, output_dim
        )

    def _sample_time_encoding(self, sample_time):
        hours = sample_time.to(torch.float64) / 3.6e12
        offsets = torch.tensor(
            [-self.time_window_hours, 0.0, self.time_window_hours],
            device=sample_time.device, dtype=torch.float64,
        )
        hours = hours.unsqueeze(-1) + offsets
        hour_phase = 2.0 * torch.pi * torch.remainder(hours, 24.0) / 24.0
        year_phase = 2.0 * torch.pi * torch.remainder(
            hours, 24.0 * 366.0
        ) / (24.0 * 366.0)
        return torch.cat([
            torch.sin(year_phase), torch.cos(year_phase),
            torch.sin(hour_phase), torch.cos(hour_phase),
        ], dim=-1)

    def forward(self, satellite_id, is_land, sample_time, obs_time=None):
        satellite_id = torch.remainder(
            satellite_id.long(), self.num_satellite_embeddings
        )
        satellite = self.satellite_embedding(satellite_id)
        while sample_time.ndim < satellite_id.ndim:
            sample_time = sample_time.unsqueeze(-1)
        calendar = self._sample_time_encoding(sample_time).expand(
            *satellite_id.shape, -1
        )
        pieces = [calendar, is_land.to(torch.float64).unsqueeze(-1)]
        if self.include_delta_time:
            if obs_time is None:
                raise ValueError("obs_time is required by the GPSRO point encoder")
            delta_hours = (
                obs_time.to(torch.float64) - sample_time.to(torch.float64)
            ) / 3.6e12
            normalized = (
                delta_hours / (0.5 * self.time_window_hours)
            ).clamp(-2.0, 2.0)
            phase = torch.pi * normalized
            pieces.append(torch.stack([
                normalized,
                torch.sin(phase), torch.cos(phase),
                torch.sin(2.0 * phase), torch.cos(2.0 * phase),
            ], dim=-1))
        numeric = torch.cat(pieces, dim=-1).to(satellite.dtype)
        return self.mlp(torch.cat([satellite, numeric], dim=-1))


class GPSROPointEncoder(nn.Module):
    def __init__(self, point_dim=64, metadata_dim=32,
                 num_satellite_embeddings=256, embedding_dim=16,
                 value_hidden_dims=(64, 64), metadata_hidden_dims=(64,),
                 fusion_hidden_dims=(64,)):
        super().__init__()
        if not value_hidden_dims:
            raise ValueError("value_encoder_hidden_dims must contain at least one size")
        # Two inputs are [standardized refractivity, validity indicator].
        # The final activation matches the original value encoder behaviour.
        value_feature_dim = int(value_hidden_dims[-1])
        self.value_encoder = build_mlp(
            2, value_hidden_dims[:-1], value_feature_dim,
            activate_output=True,
        )
        self.metadata = GPSROMetadataEncoder(
            metadata_dim, num_satellite_embeddings, embedding_dim,
            include_delta_time=True,
            hidden_dims=metadata_hidden_dims,
        )
        self.fusion = build_mlp(
            value_feature_dim + metadata_dim,
            fusion_hidden_dims,
            point_dim,
        )

    def forward(self, refractivity, valid, satellite_id, is_land,
                obs_time, sample_time):
        finite_valid = valid & torch.isfinite(refractivity)
        value = torch.where(
            finite_valid, refractivity, torch.zeros_like(refractivity)
        )
        value_feature = self.value_encoder(torch.cat([
            value, finite_valid.to(value.dtype)
        ], dim=-1))
        metadata = self.metadata(
            satellite_id=satellite_id,
            is_land=is_land,
            obs_time=obs_time,
            sample_time=sample_time,
        )
        return self.fusion(torch.cat([value_feature, metadata], dim=-1))
