"""Small CPU tests for the three-dimensional local SetConv operators."""

import torch

from satellite.models import (
    SetConv3DOffToOn,
    SetConv3DOnToOff,
    VerticalCoordinate,
)


def _operator(cls):
    return cls(
        grid_resolution_deg=30.0,
        vertical_min_m=0.0,
        vertical_max_m=2_000.0,
        vertical_resolution_m=1_000.0,
        horizontal_radius=1,
        vertical_radius=1,
        init_horizontal_lengthscale_km=1_000.0,
        init_vertical_lengthscale_m=1_000.0,
    )


def test_vertical_coordinate_equivalences():
    geopotential_height = torch.tensor([0.0, 1_000.0, 2_000.0])
    geopotential = geopotential_height * VerticalCoordinate.STANDARD_GRAVITY
    converted = VerticalCoordinate.to_geopotential_height(
        geopotential, "geopotential", "m2 s-2"
    )
    assert torch.allclose(converted, geopotential_height)

    pressure = 101_325.0 * torch.exp(-geopotential_height / 7_000.0)
    converted = VerticalCoordinate.to_geopotential_height(
        pressure, "pressure", "Pa"
    )
    assert torch.allclose(converted, geopotential_height, atol=1.0e-3)

    altitude = torch.tensor([1_000.0])
    converted = VerticalCoordinate.to_geopotential_height(
        altitude, "altitude", "m"
    )
    assert 999.0 < converted.item() < 1_000.0


def test_3d_shapes_values_and_gradients():
    torch.manual_seed(0)
    encoder = _operator(SetConv3DOffToOn)
    decoder = _operator(SetConv3DOnToOff)
    features = torch.randn(2, 7, 3, requires_grad=True)
    lon = torch.rand(2, 7) * 360.0 - 180.0
    lat = torch.rand(2, 7) * 180.0 - 90.0
    height = torch.rand(2, 7) * 2_000.0

    grid, density = encoder(features, lon, lat, height)
    assert grid.shape == (2, 3, encoder.depth, encoder.height, encoder.width)
    assert density.shape == (2, 1, encoder.depth, encoder.height, encoder.width)
    query = decoder(grid, lon, lat, height)
    assert query.shape == features.shape
    assert torch.isfinite(query).all()
    query.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert encoder.log_horizontal_lengthscale.grad is not None
    assert encoder.log_vertical_lengthscale.grad is not None
    assert decoder.log_horizontal_lengthscale.grad is not None
    assert decoder.log_vertical_lengthscale.grad is not None


def test_invalid_pressure_contributes_no_density():
    encoder = _operator(SetConv3DOffToOn)
    features = torch.ones(1, 2, 1)
    lon = torch.zeros(1, 2)
    lat = torch.zeros(1, 2)
    pressure = torch.tensor([[101_325.0, -1.0]])
    _, density_both = encoder(
        features, lon, lat, pressure, vertical_type="pressure", vertical_unit="Pa"
    )
    _, density_one = encoder(
        features[:, :1], lon[:, :1], lat[:, :1], pressure[:, :1],
        vertical_type="pressure", vertical_unit="Pa",
    )
    assert torch.allclose(density_both, density_one)


if __name__ == "__main__":
    test_vertical_coordinate_equivalences()
    test_3d_shapes_values_and_gradients()
    test_invalid_pressure_contributes_no_density()
    print("SetConv3D tests passed")
