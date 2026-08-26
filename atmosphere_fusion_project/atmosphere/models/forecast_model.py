from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .adapters import AtmosphereToInstrumentHead, InstrumentToAtmosphereAdapter
from .fusion import InstrumentFusion
from .swin3d import SwinForecastProcessor3D


class AtmosphereFusionForecastModel(nn.Module):
    """Instrument latents -> common 3-D atmosphere -> Swin forecast -> instrument latents."""

    def __init__(
        self,
        instrument_dims: Mapping[str, int],
        atmosphere_dim: int = 96,
        latent_levels: int = 4,
        pressure_levels_hpa=None,
        fusion_refine_blocks: int = 2,
        swin_depth: int = 6,
        swin_num_heads: int = 6,
        swin_window_size=(2, 7, 6),
        swin_mlp_ratio: float = 4.0,
        swin_drop: float = 0.0,
        swin_attn_drop: float = 0.0,
        swin_drop_path: float = 0.1,
    ):
        super().__init__()
        self.instrument_names = tuple(instrument_dims.keys())
        self.instrument_dims = dict(instrument_dims)
        self.atmosphere_dim = int(atmosphere_dim)
        self.latent_levels = int(latent_levels)
        self.adapters = nn.ModuleDict({
            name: InstrumentToAtmosphereAdapter(
                in_dim=dim,
                atmosphere_dim=atmosphere_dim,
                latent_levels=latent_levels,
                pressure_levels_hpa=pressure_levels_hpa,
            )
            for name, dim in instrument_dims.items()
        })
        self.fusion = InstrumentFusion(
            self.instrument_names, atmosphere_dim, refine_blocks=fusion_refine_blocks
        )
        self.processor = SwinForecastProcessor3D(
            dim=atmosphere_dim,
            depth=swin_depth,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            drop=swin_drop,
            attn_drop=swin_attn_drop,
            drop_path=swin_drop_path,
            periodic_longitude=True,
        )
        self.instrument_heads = nn.ModuleDict({
            name: AtmosphereToInstrumentHead(
                atmosphere_dim, latent_levels, out_dim=dim
            )
            for name, dim in instrument_dims.items()
        })

    def fuse(self, latents, densities, available):
        tokens, confidences = {}, {}
        for name in self.instrument_names:
            tokens[name], confidences[name] = self.adapters[name](
                latents[name], densities[name], available[name]
            )
        return self.fusion(tokens, confidences, available)

    def decode_state(self, state):
        outputs = {}
        for name in self.instrument_names:
            latent, log_density = self.instrument_heads[name](state)
            outputs[name] = {"latent": latent, "log_density": log_density}
        return outputs

    def forecast_state(self, state):
        return self.processor(state)

    def forward(self, latents, densities, available, steps=1):
        state, fusion_weights = self.fuse(latents, densities, available)
        current = self.decode_state(state)
        future = []
        for _ in range(int(steps)):
            state = self.forecast_state(state)
            future.append(self.decode_state(state))
        return {
            "state": state,
            "fusion_weights": fusion_weights,
            "current": current,
            "future": future,
        }
