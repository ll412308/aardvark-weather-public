from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
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
        swin_window_size=(2, 9, 9),
        swin_mlp_ratio: float = 4.0,
        swin_drop: float = 0.0,
        swin_attn_drop: float = 0.0,
        swin_drop_path: float = 0.1,
        spatial_multiple: int = 9,
    ):
        super().__init__()
        self.instrument_names = tuple(instrument_dims.keys())
        self.instrument_dims = dict(instrument_dims)
        self.atmosphere_dim = int(atmosphere_dim)
        self.latent_levels = int(latent_levels)
        self.spatial_multiple = int(spatial_multiple)
        if self.spatial_multiple < 1:
            raise ValueError("spatial_multiple must be at least 1")
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

    def spatial_shapes(self, latents):
        """Return each instrument's original (H, W) output shape."""
        return {name: tuple(latents[name].shape[-2:]) for name in self.instrument_names}

    def _work_size(self, latents):
        shapes = self.spatial_shapes(latents)
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(
                "Instrument latent grids must be spatially aligned before fusion; "
                f"got {shapes}"
            )
        height, width = next(iter(unique_shapes))
        multiple = self.spatial_multiple
        work_height = height - height % multiple
        work_width = width - width % multiple
        if work_height < multiple or work_width < multiple:
            raise ValueError(
                f"Spatial grid {(height, width)} is too small for multiple={multiple}"
            )
        return work_height, work_width

    @staticmethod
    def _resize_grid(x, size):
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def fuse(self, latents, densities, available):
        work_size = self._work_size(latents)
        tokens, confidences = {}, {}
        for name in self.instrument_names:
            latent = self._resize_grid(latents[name], work_size)
            density = self._resize_grid(densities[name], work_size)
            tokens[name], confidences[name] = self.adapters[name](
                latent, density, available[name]
            )
        return self.fusion(tokens, confidences, available)

    def decode_state(self, state, output_shapes=None):
        outputs = {}
        for name in self.instrument_names:
            latent = self.instrument_heads[name](state)
            if output_shapes is not None:
                latent = self._resize_grid(latent, output_shapes[name])
            outputs[name] = {"latent": latent}
        return outputs

    def forecast_state(self, state):
        return self.processor(state)

    def forward(self, latents, densities, available, steps=1):
        output_shapes = self.spatial_shapes(latents)
        state, fusion_weights = self.fuse(latents, densities, available)
        current = self.decode_state(state, output_shapes)
        future = []
        for _ in range(int(steps)):
            state = self.forecast_state(state)
            future.append(self.decode_state(state, output_shapes))
        return {
            "state": state,
            "fusion_weights": fusion_weights,
            "current": current,
            "future": future,
        }
