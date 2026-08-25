"""Synthetic gradient/mask test and optional small real-Zarr smoke test."""

import argparse

import torch

from satellite.config import BAMUAConfig
from satellite.loss import masked_mse_loss
from satellite.models import BAMUAAutoEncoder


def synthetic_batch(batch=1, n_context=100, n_target=20, channels=15):
    def metadata(n):
        sample = torch.full((batch,), 1_700_000_000_000_000_000, dtype=torch.long)
        return dict(
            lon=torch.rand(batch, n) * 360 - 180,
            lat=torch.rand(batch, n) * 180 - 90,
            satellite_id=torch.randint(0, 4, (batch, n)),
            is_land=torch.randint(0, 2, (batch, n)).bool(),
            obs_time=sample[:, None] + torch.randint(-10_800, 10_800, (batch, n)) * 10**9,
            sample_time=sample,
            zenith=torch.rand(batch, n) * 70,
            azimuth=torch.rand(batch, n) * 360,
        )
    context = metadata(n_context)
    context.update(bt=torch.randn(batch, n_context, channels),
                   valid=torch.rand(batch, n_context, channels) > 0.15)
    target = metadata(n_target)
    return context, target, torch.randn(batch, n_target, channels), \
        (torch.rand(batch, n_target, channels) > 0.15)


def run_batch(model, context, target, target_bt, target_valid):
    pred, latent, density = model(context, target)
    query_without_optional = {
        key: target[key] for key in ("lon", "lat", "satellite_id", "is_land", "sample_time")
    }
    pred_without_optional = model.decode(latent, density, **query_without_optional)
    assert pred_without_optional.shape == target_bt.shape
    assert pred.shape == target_bt.shape
    assert latent.shape[1:] == (model.config.latent_dim,
                                model.config.grid_height, model.config.grid_width)
    assert density.shape[1:] == (1, model.config.grid_height, model.config.grid_width)
    loss = masked_mse_loss(pred, target_bt, target_valid)
    loss.backward()
    grads = {
        "off_to_on": model.point_to_grid.log_lengthscale.grad,
        "on_to_off": model.grid_to_point.log_lengthscale.grad,
    }
    assert all(value is not None and torch.isfinite(value) for value in grads.values())
    # A masked target may change arbitrarily without changing the loss.
    changed = target_bt.clone()
    changed[~target_valid] += 12345.0
    assert torch.allclose(loss.detach(), masked_mse_loss(pred.detach(), changed, target_valid))
    print({"pred": tuple(pred.shape), "latent": tuple(latent.shape),
           "density": tuple(density.shape), "loss": float(loss),
           "lengthscale_grad": {k: float(v) for k, v in grads.items()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", help="Optional path for one real-bin sampled smoke test",nargs="?",default=r"F:\lyh_data\data_zarr_no_provider_filter\1bamua.zarr")
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--target-overlap", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="Use 10 degrees for a lightweight smoke test")
    args = parser.parse_args()
    config = BAMUAConfig(grid_resolution_deg=args.resolution,
                         n_context=args.context, n_target=args.target,
                         target_overlap=args.target_overlap)
    model = BAMUAAutoEncoder(config)
    if args.zarr:
        from satellite.datasets import BAMUAZarrDataset
        item = BAMUAZarrDataset(args.zarr, args.context, args.target,
                                target_overlap=args.target_overlap)[0]
        context = {k: v.unsqueeze(0) for k, v in item["context"].items()}
        target = {k: v.unsqueeze(0) for k, v in item["target"].items()}
        run_batch(model, context, target, item["target_bt"].unsqueeze(0),
                  item["target_valid"].unsqueeze(0))
    else:
        run_batch(model, *synthetic_batch(n_context=args.context, n_target=args.target))


if __name__ == "__main__":
    main()
