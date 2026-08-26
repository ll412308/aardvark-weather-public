import torch

from atmosphere.models import AtmosphereFusionForecastModel
from atmosphere.losses import latent_mse


def main():
    torch.manual_seed(0)
    b, h, w = 2, 19, 27
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
        spatial_multiple=9,
    )
    latents = {n: torch.randn(b, d, h, w) for n, d in dims.items()}
    densities = {n: torch.rand(b, 1, h, w) * 5 for n in dims}
    available = {"1bamua": torch.tensor([True, True]), "mhs": torch.tensor([True, False])}
    state, weights = model.fuse(latents, densities, available)
    assert state.shape == (b, 32, 4, 18, 27)
    next_state = model.forecast_state(state)
    decoded = model.decode_state(next_state, model.spatial_shapes(latents))
    loss = 0.0
    for name, dim in dims.items():
        assert decoded[name]["latent"].shape == (b, dim, h, w)
        loss = loss + latent_mse(
            decoded[name]["latent"], latents[name], available[name],
            density=densities[name], use_density_mask=False,
        )
        masked_loss = latent_mse(
            decoded[name]["latent"], latents[name], available[name],
            density=densities[name], use_density_mask=True,
            density_threshold=2.5,
        )
        assert torch.isfinite(masked_loss)
    loss.backward()
    grad = model.processor.blocks[0].attn.qkv.weight.grad
    assert grad is not None
    print("smoke OK")
    print("state", tuple(state.shape), "loss", float(loss.detach()))
    print("fusion weight 1bamua", tuple(weights["1bamua"].shape))


if __name__ == "__main__":
    main()
