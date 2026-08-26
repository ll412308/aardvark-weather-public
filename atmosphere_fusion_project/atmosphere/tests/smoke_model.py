import torch

from atmosphere.models import AtmosphereFusionForecastModel
from atmosphere.losses import latent_coverage_mse


def main():
    torch.manual_seed(0)
    b, h, w = 2, 16, 24
    dims = {"1bamua": 12, "mhs": 8}
    model = AtmosphereFusionForecastModel(
        instrument_dims=dims,
        atmosphere_dim=32,
        latent_levels=4,
        fusion_refine_blocks=1,
        swin_depth=2,
        swin_num_heads=4,
        swin_window_size=(2, 4, 4),
        swin_drop_path=0.0,
    )
    latents = {n: torch.randn(b, d, h, w) for n, d in dims.items()}
    densities = {n: torch.rand(b, 1, h, w) * 5 for n in dims}
    available = {"1bamua": torch.tensor([True, True]), "mhs": torch.tensor([True, False])}
    state, weights = model.fuse(latents, densities, available)
    assert state.shape == (b, 32, 4, h, w)
    next_state = model.forecast_state(state)
    decoded = model.decode_state(next_state)
    loss = 0.0
    for name, dim in dims.items():
        assert decoded[name]["latent"].shape == (b, dim, h, w)
        assert decoded[name]["log_density"].shape == (b, 1, h, w)
        loss = loss + latent_coverage_mse(
            decoded[name]["latent"], latents[name], densities[name], available[name]
        )
    loss.backward()
    grad = model.processor.blocks[0].attn.qkv.weight.grad
    assert grad is not None
    print("smoke OK")
    print("state", tuple(state.shape), "loss", float(loss.detach()))
    print("fusion weight 1bamua", tuple(weights["1bamua"].shape))


if __name__ == "__main__":
    main()
