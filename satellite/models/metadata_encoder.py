import torch
from torch import nn


class MetadataEncoder(nn.Module):
    def __init__(self, output_dim=32, num_satellite_embeddings=256,
                 embedding_dim=16, include_delta_time=True,
                 include_angles=True, time_window_hours=6.0):
        super().__init__()
        self.num_satellite_embeddings = num_satellite_embeddings
        self.include_delta_time = bool(include_delta_time)
        self.include_angles = bool(include_angles)
        self.time_window_hours = float(time_window_hours)
        self.satellite_embedding = nn.Embedding(num_satellite_embeddings, embedding_dim)
        # sample_time uses GraphCast-style calendar Fourier features at
        # t-6h, t, t+6h. Delta time and viewing angles are encoder-only.
        numeric_dim = 12 + 1
        if self.include_delta_time:
            numeric_dim += 5
        if self.include_angles:
            numeric_dim += 4
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + numeric_dim, 64), nn.SiLU(),
            nn.Linear(64, output_dim)
        )

    def _sample_time_encoding(self, sample_time):
        sample_hours = sample_time.to(torch.float64) / 3.6e12
        offsets = torch.tensor(
            [-self.time_window_hours, 0.0, self.time_window_hours],
            dtype=torch.float64,
            device=sample_time.device,
        )
        hours = sample_hours.unsqueeze(-1) + offsets
        hour_phase = 2.0 * torch.pi * torch.remainder(hours, 24.0) / 24.0
        year_phase = 2.0 * torch.pi * torch.remainder(hours, 24.0 * 366.0) / (
            24.0 * 366.0
        )
        return torch.cat([
            torch.sin(year_phase), torch.cos(year_phase),
            torch.sin(hour_phase), torch.cos(hour_phase),
        ], dim=-1)

    def forward(self, satellite_id, is_land, obs_time, sample_time,
                zenith=None, azimuth=None):
        satellite_id = torch.remainder(satellite_id.long(), self.num_satellite_embeddings)
        sat = self.satellite_embedding(satellite_id)  # B, N, embedding_dim
        while sample_time.ndim < satellite_id.ndim:
            sample_time = sample_time.unsqueeze(-1)  # B, 1
        dtype = sat.dtype
        sample_time_features = self._sample_time_encoding(sample_time)  # B,1,12
        sample_time_features = sample_time_features.expand(*satellite_id.shape, -1)  # B, N, 12
        pieces = [sample_time_features, is_land.to(torch.float64).unsqueeze(-1)]
        if self.include_delta_time:
            if obs_time is None:
                raise ValueError("obs_time is required when include_delta_time=True")
            delta_t = (obs_time.to(torch.float64) - sample_time.to(torch.float64)) / 3.6e12
            delta_t_norm = (delta_t / (0.5 * self.time_window_hours)).clamp(-2.0, 2.0)
            delta_phase = torch.pi * delta_t_norm
            pieces.append(torch.stack([
                delta_t_norm,
                torch.sin(delta_phase),
                torch.cos(delta_phase),
                torch.sin(2.0 * delta_phase),
                torch.cos(2.0 * delta_phase),
            ], dim=-1))  # STACK: B, N, 5
        if self.include_angles:
            if zenith is None or azimuth is None:
                raise ValueError("zenith and azimuth are required when include_angles=True")
            # A single non-finite angle would pass through sin/cos and then
            # spread from one point into many grid cells through SetConv and
            # latent attention. Missing viewing angles have no separate mask in
            # this checkpoint, so use a finite neutral fallback of 0 degrees.
            azimuth = torch.nan_to_num(
                azimuth.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            zenith = torch.nan_to_num(
                zenith.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            azimuth_rad = torch.deg2rad(azimuth)  # B,N
            zenith_rad = torch.deg2rad(zenith)
            pieces.append(torch.stack([
                torch.sin(azimuth_rad),
                torch.cos(azimuth_rad),
                zenith / 90.0,
                torch.cos(zenith_rad),
            ], dim=-1))  # STACK: B, N, 4
        numeric = torch.cat(pieces, dim=-1).to(dtype)  # B, N, numeric_dim(12 + 1 or + 5 + 4)

        output = self.mlp(torch.cat([sat, numeric], dim=-1))  # B, N, output_dim
        return output
