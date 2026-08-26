from __future__ import annotations

import math

import torch
from torch import nn


class VerticalEncoding(nn.Module):
    """Vertical coordinate encoding for the common atmospheric latent."""

    def __init__(self, channels: int, levels: int, pressure_levels_hpa=None, num_freqs=8):
        super().__init__()
        self.channels = int(channels)
        self.levels = int(levels)
        if pressure_levels_hpa is None:
            self.learned = nn.Parameter(torch.zeros(1, channels, levels, 1, 1))
            nn.init.trunc_normal_(self.learned, std=0.02)
            self.register_buffer("pressure", None, persistent=False)
            self.proj = None
        else:
            if len(pressure_levels_hpa) != levels:
                raise ValueError("pressure_levels_hpa length must equal latent_levels")
            p = torch.tensor(pressure_levels_hpa, dtype=torch.float32)
            self.register_buffer("pressure", p, persistent=True)
            self.learned = None
            self.num_freqs = int(num_freqs)
            self.proj = nn.Sequential(
                nn.Linear(2 * self.num_freqs, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )

    def forward(self, batch_size: int, device=None, dtype=None):
        if self.learned is not None:
            return self.learned.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1, -1)
        logp = torch.log(self.pressure.clamp_min(1.0))
        logp = (logp - logp.mean()) / logp.std().clamp_min(1.0e-6)
        freq = torch.arange(1, self.num_freqs + 1, device=logp.device, dtype=logp.dtype)
        phase = logp[:, None] * freq[None, :] * math.pi
        enc = torch.cat([phase.sin(), phase.cos()], dim=-1)
        enc = self.proj(enc).transpose(0, 1).view(1, self.channels, self.levels, 1, 1)
        return enc.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1, -1)


class InstrumentToAtmosphereAdapter(nn.Module):
    """Map one instrument's 2-D latent field into a common 3-D atmospheric latent."""

    def __init__(self, in_dim: int, atmosphere_dim: int, latent_levels: int,
                 pressure_levels_hpa=None):
        super().__init__()
        self.atmosphere_dim = int(atmosphere_dim)
        self.latent_levels = int(latent_levels)
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, atmosphere_dim * latent_levels, kernel_size=1),
            nn.SiLU(),
        )
        self.confidence = nn.Conv2d(1, latent_levels, kernel_size=1)
        self.vertical = VerticalEncoding(
            atmosphere_dim, latent_levels, pressure_levels_hpa=pressure_levels_hpa
        )

    def forward(self, latent, density, available):
        if latent.ndim != 4:
            raise ValueError(f"Expected instrument latent [B,D,H,W], got {latent.shape}")
        b, _, h, w = latent.shape
        x = self.project(latent).view(
            b, self.atmosphere_dim, self.latent_levels, h, w
        )
        x = x + self.vertical(b, device=x.device, dtype=x.dtype)
        log_density = torch.log1p(density.clamp_min(0.0))
        confidence = torch.sigmoid(self.confidence(log_density)).unsqueeze(1)
        avail = available.to(x.dtype).view(b, 1, 1, 1, 1)
        confidence = confidence * avail
        x = x * avail
        return x, confidence


class AtmosphereToInstrumentHead(nn.Module):
    """Decode the common 3-D atmosphere back to one instrument's 2-D latent."""

    def __init__(self, atmosphere_dim: int, latent_levels: int, out_dim: int):
        super().__init__()
        kernel = (int(latent_levels), 1, 1)
        self.latent_head = nn.Sequential(
            nn.Conv3d(atmosphere_dim, atmosphere_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(atmosphere_dim, out_dim, kernel_size=kernel),
        )

    def forward(self, atmosphere):
        return self.latent_head(atmosphere).squeeze(2)
