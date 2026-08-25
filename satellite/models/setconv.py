"""Local SetConv for paired irregular longitude/latitude observations.

The density-normalised Gaussian aggregation follows Aardvark's ``convDeepSet``.
Unlike Aardvark's separable dense einsums, this implementation preserves each
paired (lon_i, lat_i) and scatter-adds only to a fixed local grid neighbourhood.
"""

import math

import torch
from torch import nn


class _LocalGaussian(nn.Module):
    def __init__(self, grid_resolution_deg=2.0, local_radius=1,
                 init_lengthscale_deg=2.0, eps=1.0e-6):
        super().__init__()
        self.resolution = float(grid_resolution_deg)
        self.local_radius = int(local_radius)
        self.eps = float(eps)
        self.height = round(180.0 / self.resolution) + 1
        self.width = round(360.0 / self.resolution)
        self.log_lengthscale = nn.Parameter(
            torch.tensor(float(init_lengthscale_deg)).log()
        )
        offsets = torch.arange(-self.local_radius, self.local_radius + 1)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        self.register_buffer("offset_x", dx.reshape(-1), persistent=False)
        self.register_buffer("offset_y", dy.reshape(-1), persistent=False)

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp().clamp_min(1.0e-4)

    @staticmethod
    def _wrap_delta_lon(delta):
        return torch.remainder(delta + 180.0, 360.0) - 180.0

    def _neighbours(self, lon, lat):
        lon = torch.remainder(lon + 180.0, 360.0) - 180.0
        lat = lat.clamp(-90.0, 90.0)
        gx = (lon + 180.0) / self.resolution
        gy = (lat + 90.0) / self.resolution
        ix = torch.floor(gx).long().unsqueeze(-1) + self.offset_x
        iy = torch.floor(gy).long().unsqueeze(-1) + self.offset_y
        valid = (iy >= 0) & (iy < self.height)
        ix = torch.remainder(ix, self.width)
        iy_safe = iy.clamp(0, self.height - 1)
        node_lon = -180.0 + ix.to(lon.dtype) * self.resolution
        node_lat = -90.0 + iy_safe.to(lat.dtype) * self.resolution
        dlon = self._wrap_delta_lon(lon.unsqueeze(-1) - node_lon)
        mid_lat = 0.5 * (lat.unsqueeze(-1) + node_lat)
        dx = dlon * torch.cos(torch.deg2rad(mid_lat))
        dy = lat.unsqueeze(-1) - node_lat
        weight = torch.exp(-0.5 * (dx.square() + dy.square()) /
                           self.lengthscale.square()) * valid.to(lon.dtype)  # 后期可以改成大圆距离
        return iy_safe * self.width + ix, weight


class SetConvOffToOn(_LocalGaussian):
    """[B,N,D] paired points -> density-normalised [B,D,H,W]."""

    def aggregate(self, features, lon, lat, point_mask=None):
        """Return unnormalised weighted feature sums and density."""
        if features.ndim != 3 or lon.shape != features.shape[:2] or lat.shape != lon.shape:
            raise ValueError("Expected features [B,N,D] and lon/lat [B,N]")
        index, weight = self._neighbours(lon, lat)
        if point_mask is not None:
            weight = weight * point_mask.unsqueeze(-1).to(weight.dtype)
        b, n, d = features.shape
        k = index.shape[-1]
        flat_size = self.height * self.width
        density = features.new_zeros((b, flat_size))
        density.scatter_add_(1, index.reshape(b, n * k), weight.reshape(b, n * k))
        weighted = (features.unsqueeze(2) * weight.unsqueeze(-1)).reshape(b, n * k, d)
        latent_sum = features.new_zeros((b, flat_size, d))
        latent_sum.scatter_add_(1, index.reshape(b, n * k, 1).expand(-1, -1, d), weighted)
        latent_sum = latent_sum.transpose(1, 2).reshape(
            b, d, self.height, self.width
        )
        density = density.reshape(b, 1, self.height, self.width)
        return latent_sum, density

    def forward(self, features, lon, lat, point_mask=None):
        latent_sum, density = self.aggregate(
            features, lon, lat, point_mask=point_mask
        )
        return latent_sum / density.clamp_min(self.eps), density


class SetConvOnToOff(_LocalGaussian):
    """[B,D,H,W] -> Gaussian-interpolated features at arbitrary [B,Nq] points."""

    def forward(self, grid, lon, lat):
        if grid.ndim != 4 or grid.shape[-2:] != (self.height, self.width):
            raise ValueError(f"Expected grid [B,D,{self.height},{self.width}]")
        index, weight = self._neighbours(lon, lat)
        b, d, _, _ = grid.shape
        flat_size = self.height * self.width
        flat = grid.reshape(b, d, flat_size).transpose(1, 2).reshape(b * flat_size, d)
        batch_offset = torch.arange(b, device=grid.device).view(b, 1, 1) * flat_size
        gathered = flat[index + batch_offset]
        return (gathered * weight.unsqueeze(-1)).sum(2) / weight.sum(2, keepdim=True).clamp_min(self.eps)
