"""Small synthetic and real-Zarr forward/backward tests for GPSRO AE."""

import argparse

import torch

from satellite.datasets import GPSROZarrDataset
from satellite.gpsro_config import GPSROConfig
from satellite.loss import masked_mse_loss
from satellite.models import GPSROAutoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr", default=r"F:\lyh_data\gps_zarr_no_provider\gpsro.zarr"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    config = GPSROConfig(
        point_dim=16, latent_dim=64, metadata_dim=8,
        grid_resolution_deg=30.0,
        vertical_min_m=-1_000.0,
        vertical_max_m=64_000.0,
        vertical_resolution_m=5_000.0,
        latent_processor_dim=32,
        latent_processor_depth=1,
        latent_processor_heads=4,
        latent_window_size=3,
        setconv_chunk_size=100,
        decode_chunk_size=20,
        n_context=100,
        n_target=20,
    )
    dataset = GPSROZarrDataset(
        args.zarr, config.n_context, config.n_target,
        config.target_overlap, seed=0,
    )
    item = dataset[args.sample_index]
    context = {
        name: value.unsqueeze(0) for name, value in item["context"].items()
    }
    target = {
        name: value.unsqueeze(0) for name, value in item["target"].items()
    }
    target_value = item["target_refractivity"].unsqueeze(0)
    target_valid = item["target_valid"].unsqueeze(0)
    model = GPSROAutoEncoder(config)
    prediction, latent, density = model(context, target)
    loss = masked_mse_loss(prediction, target_value, target_valid)
    loss.backward()
    print(f"source_sample_index={item['source_sample_index'].item()}")
    print(f"prediction={tuple(prediction.shape)}")
    print(f"latent={tuple(latent.shape)}")
    print(f"density={tuple(density.shape)}")
    print(f"restored_3d={tuple(model.latent_processor.restore_3d(latent).shape)}")
    print(f"loss={loss.item():.6f}")
    gradients = {
        "offtoon_horizontal": model.point_to_grid.log_horizontal_lengthscale.grad,
        "offtoon_vertical": model.point_to_grid.log_vertical_lengthscale.grad,
        "ontooff_horizontal": model.grid_to_point.log_horizontal_lengthscale.grad,
        "ontooff_vertical": model.grid_to_point.log_vertical_lengthscale.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not torch.isfinite(gradient):
            raise AssertionError(f"Missing/non-finite gradient: {name}")
        print(f"{name}_grad={gradient.item():.6e}")
    print("GPSRO real-sample forward/backward test passed")


if __name__ == "__main__":
    main()
