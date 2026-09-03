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


def _print_shape(name, value):
    """Pretty shape output used only by the synthetic demo below."""
    print(f"{name:<42} shape={tuple(value.shape)} dtype={value.dtype}")


def _synthetic_shape_demo():
    """Run every model stage on small fake multi-instrument latent grids."""
    torch.manual_seed(0)

    # Two instruments may have different latent channel counts D, but their
    # latitude/longitude grids must be aligned before fusion.
    batch_size = 2
    original_height, original_width = 19, 36
    instrument_dims = {"1bamua": 6, "mhs": 4}
    model = AtmosphereFusionForecastModel(
        instrument_dims=instrument_dims,
        atmosphere_dim=8,
        latent_levels=2,
        fusion_refine_blocks=1,
        swin_depth=2,
        swin_num_heads=2,
        swin_window_size=(2, 3, 3),
        swin_drop_path=0.0,
        spatial_multiple=3,
    ).eval()

    latents = {
        name: torch.randn(batch_size, dim, original_height, original_width)
        for name, dim in instrument_dims.items()
    }
    densities = {
        name: torch.rand(batch_size, 1, original_height, original_width) * 5.0
        for name in instrument_dims
    }
    available = {
        "1bamua": torch.tensor([True, True]),
        # The second batch sample has no MHS, but it still has 1BAMUA.
        "mhs": torch.tensor([True, False]),
    }

    print("\n=== 1. Synthetic instrument inputs ===")
    for name in model.instrument_names:
        _print_shape(f"{name} latent [B,D,H,W]", latents[name])
        _print_shape(f"{name} density [B,1,H,W]", densities[name])
        print(f"{name + ' available':<42} values={available[name].tolist()}")

    output_shapes = model.spatial_shapes(latents)
    work_size = model._work_size(latents)
    print("\n=== 2. Grid sizes ===")
    print(f"original output shapes                    {output_shapes}")
    print(f"fusion/Swin work size                     {work_size}")

    print("\n=== 3. Resize and instrument adapters ===")
    tokens, confidences = {}, {}
    with torch.no_grad():
        for name in model.instrument_names:
            resized_latent = model._resize_grid(latents[name], work_size)
            resized_density = model._resize_grid(densities[name], work_size)
            tokens[name], confidences[name] = model.adapters[name](
                resized_latent, resized_density, available[name]
            )
            _print_shape(f"{name} resized latent", resized_latent)
            _print_shape(f"{name} resized density", resized_density)
            _print_shape(f"{name} atmosphere token [B,C,L,H,W]", tokens[name])
            _print_shape(f"{name} confidence [B,1,L,H,W]", confidences[name])

        print("\n=== 4. Instrument fusion ===")
        fused, fusion_weights = model.fusion(
            tokens, confidences, available
        )
        _print_shape("fused atmosphere state", fused)
        for name in model.instrument_names:
            _print_shape(f"{name} fusion weight", fusion_weights[name])
        weight_sum = torch.stack(
            [fusion_weights[name] for name in model.instrument_names], dim=1
        ).sum(dim=1)
        _print_shape("sum of weights over instruments", weight_sum)
        print(
            "weight sum min/max                       "
            f"{weight_sum.min().item():.6f} / {weight_sum.max().item():.6f}"
        )

        print("\n=== 5. Decode the current fused state ===")
        current_work_grid = model.decode_state(fused)
        current_original_grid = model.decode_state(fused, output_shapes)
        for name in model.instrument_names:
            _print_shape(
                f"{name} head output on work grid",
                current_work_grid[name]["latent"],
            )
            _print_shape(
                f"{name} restored original grid",
                current_original_grid[name]["latent"],
            )

        print("\n=== 6. One forecast step and decode ===")
        future_state = model.forecast_state(fused)
        _print_shape("future atmosphere state", future_state)
        future = model.decode_state(future_state, output_shapes)
        for name in model.instrument_names:
            _print_shape(
                f"{name} future restored latent",
                future[name]["latent"],
            )

        print("\n=== 7. Complete forward API ===")
        result = model(latents, densities, available, steps=2)
        _print_shape("result['state'] after 2 forecast steps", result["state"])
        for name in model.instrument_names:
            _print_shape(
                f"result current {name}", result["current"][name]["latent"]
            )
            for lead, prediction in enumerate(result["future"], start=1):
                _print_shape(
                    f"result lead {lead} {name}",
                    prediction[name]["latent"],
                )


if __name__ == "__main__":
    # Run from atmosphere_fusion_project so package-relative imports work:
    # python -m atmosphere.models.forecast_model
    _synthetic_shape_demo()
