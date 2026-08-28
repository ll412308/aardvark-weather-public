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
        # Under AMP the point features can be float16/bfloat16 while the
        # coordinate-derived kernel weights stay float32. scatter_add_ requires
        # matching dtypes, and density sums are more stable in float32 anyway.
        accum_dtype = (
            torch.float32
            if features.dtype in (torch.float16, torch.bfloat16)
            else features.dtype
        )
        features = features.to(accum_dtype)
        weight = weight.to(accum_dtype)
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
        accum_dtype = (
            torch.float32
            if grid.dtype in (torch.float16, torch.bfloat16)
            else grid.dtype
        )
        grid = grid.to(accum_dtype)
        weight = weight.to(accum_dtype)
        b, d, _, _ = grid.shape
        flat_size = self.height * self.width
        flat = grid.reshape(b, d, flat_size).transpose(1, 2).reshape(b * flat_size, d)
        batch_offset = torch.arange(b, device=grid.device).view(b, 1, 1) * flat_size
        gathered = flat[index + batch_offset]
        return (gathered * weight.unsqueeze(-1)).sum(2) / weight.sum(2, keepdim=True).clamp_min(self.eps)


class VerticalCoordinate:
    """Convert common atmospheric vertical coordinates to geopotential height.

    Geopotential height (metres) is used as the common coordinate. Pressure has
    no unique height without temperature/humidity profiles, so its conversion
    deliberately uses the common log-pressure approximation
    ``H = -scale_height * log(p / reference_pressure)``.
    """

    STANDARD_GRAVITY = 9.80665  # m s-2
    EARTH_RADIUS_M = 6_371_000.0

    _ALIASES = {
        "altitude": "altitude",
        "elevation": "altitude",
        "geometric_height": "altitude",
        "geopotential": "geopotential",
        "geopotential_height": "geopotential_height",
        "height": "geopotential_height",
        "pressure": "pressure",
    }

    @classmethod
    def to_geopotential_height(cls, value, vertical_type,
                               unit=None, reference_pressure_pa=101_325.0,
                               pressure_scale_height_m=7_000.0):
        """Return a tensor in geopotential metres for one metadata convention."""
        key = str(vertical_type).strip().lower().replace(" ", "_")
        try:
            key = cls._ALIASES[key]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._ALIASES))
            raise ValueError(
                f"Unsupported vertical_type={vertical_type!r}; use one of {supported}"
            ) from exc

        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        if not value.is_floating_point():
            value = value.float()
        unit = (unit or {
            "altitude": "m",
            "geopotential": "m2 s-2",
            "geopotential_height": "m",
            "pressure": "pa",
        }[key]).strip().lower().replace("²", "2").replace("^", "")

        if key in ("altitude", "geopotential_height"):
            if unit in ("km", "kilometer", "kilometre"):
                value = value * 1_000.0
            elif unit not in ("m", "meter", "metre"):
                raise ValueError(f"Unsupported {key} unit: {unit!r}")
        elif key == "geopotential":
            if unit not in ("m2 s-2", "m2/s2", "j/kg"):
                raise ValueError(f"Unsupported geopotential unit: {unit!r}")
            value = value / cls.STANDARD_GRAVITY
        else:
            if unit in ("hpa", "mbar", "mb"):
                value = value * 100.0
            elif unit not in ("pa", "pascal"):
                raise ValueError(f"Unsupported pressure unit: {unit!r}")
            valid_pressure = value > 0
            value = torch.where(
                valid_pressure,
                -float(pressure_scale_height_m) * torch.log(
                    value / float(reference_pressure_pa)
                ),
                torch.full_like(value, torch.nan),
            )

        if key == "altitude":
            # Convert geometric altitude to geopotential height. This matters
            # increasingly in the stratosphere and keeps all height-like inputs
            # on the same physical coordinate.
            radius = cls.EARTH_RADIUS_M
            valid_altitude = value > -radius
            value = torch.where(
                valid_altitude,
                radius * value / (radius + value),
                torch.full_like(value, torch.nan),
            )
        return value


class _LocalGaussian3D(nn.Module):
    """Shared local anisotropic Gaussian kernel on a lon/lat/vertical grid."""

    def __init__(self, grid_resolution_deg=2.0, vertical_min_m=0.0,
                 vertical_max_m=20_000.0, vertical_resolution_m=1_000.0,
                 horizontal_radius=1, vertical_radius=1,
                 init_horizontal_lengthscale_km=220.0,
                 init_vertical_lengthscale_m=1_000.0, eps=1.0e-6,
                 reference_pressure_pa=101_325.0,
                 pressure_scale_height_m=7_000.0):
        super().__init__()
        if grid_resolution_deg <= 0 or vertical_resolution_m <= 0:
            raise ValueError("Grid resolutions must be positive")
        if vertical_max_m <= vertical_min_m:
            raise ValueError("vertical_max_m must be greater than vertical_min_m")
        self.resolution = float(grid_resolution_deg)
        self.vertical_min_m = float(vertical_min_m)
        self.vertical_max_m = float(vertical_max_m)
        self.vertical_resolution_m = float(vertical_resolution_m)
        self.horizontal_radius = int(horizontal_radius)
        self.vertical_radius = int(vertical_radius)
        self.eps = float(eps)
        self.reference_pressure_pa = float(reference_pressure_pa)
        self.pressure_scale_height_m = float(pressure_scale_height_m)
        self.depth = round(
            (self.vertical_max_m - self.vertical_min_m) /
            self.vertical_resolution_m
        ) + 1
        self.height = round(180.0 / self.resolution) + 1
        self.width = round(360.0 / self.resolution)
        self.log_horizontal_lengthscale = nn.Parameter(
            torch.tensor(float(init_horizontal_lengthscale_km)).log()
        )
        self.log_vertical_lengthscale = nn.Parameter(
            torch.tensor(float(init_vertical_lengthscale_m)).log()
        )
        oz = torch.arange(-self.vertical_radius, self.vertical_radius + 1)
        oy = torch.arange(-self.horizontal_radius, self.horizontal_radius + 1)
        ox = torch.arange(-self.horizontal_radius, self.horizontal_radius + 1)
        dz, dy, dx = torch.meshgrid(oz, oy, ox, indexing="ij")
        self.register_buffer("offset_x", dx.reshape(-1), persistent=False)
        self.register_buffer("offset_y", dy.reshape(-1), persistent=False)
        self.register_buffer("offset_z", dz.reshape(-1), persistent=False)

    @property
    def horizontal_lengthscale(self):
        return self.log_horizontal_lengthscale.exp().clamp_min(1.0e-4)

    @property
    def vertical_lengthscale(self):
        return self.log_vertical_lengthscale.exp().clamp_min(1.0e-4)

    @staticmethod
    def _wrap_delta_lon(delta):
        return torch.remainder(delta + 180.0, 360.0) - 180.0

    def vertical_to_metres(self, vertical, vertical_type, vertical_unit=None):
        return VerticalCoordinate.to_geopotential_height(
            vertical, vertical_type, vertical_unit,
            reference_pressure_pa=self.reference_pressure_pa,
            pressure_scale_height_m=self.pressure_scale_height_m,
        )

    def _neighbours(self, lon, lat, vertical, vertical_type,
                    vertical_unit=None):
        z = self.vertical_to_metres(vertical, vertical_type, vertical_unit)
        coordinate_valid = torch.isfinite(lon) & torch.isfinite(lat) & torch.isfinite(z)
        lon = torch.nan_to_num(lon).remainder(360.0)
        lon = torch.remainder(lon + 180.0, 360.0) - 180.0
        lat = torch.nan_to_num(lat).clamp(-90.0, 90.0)
        z = torch.nan_to_num(z).clamp(self.vertical_min_m, self.vertical_max_m)

        gx = (lon + 180.0) / self.resolution
        gy = (lat + 90.0) / self.resolution
        gz = (z - self.vertical_min_m) / self.vertical_resolution_m
        ix = torch.floor(gx).long().unsqueeze(-1) + self.offset_x
        iy = torch.floor(gy).long().unsqueeze(-1) + self.offset_y
        iz = torch.floor(gz).long().unsqueeze(-1) + self.offset_z
        valid = ((iy >= 0) & (iy < self.height) &
                 (iz >= 0) & (iz < self.depth) & coordinate_valid.unsqueeze(-1))
        ix = torch.remainder(ix, self.width)
        iy_safe = iy.clamp(0, self.height - 1)
        iz_safe = iz.clamp(0, self.depth - 1)

        node_lon = -180.0 + ix.to(lon.dtype) * self.resolution
        node_lat = -90.0 + iy_safe.to(lat.dtype) * self.resolution
        node_z = self.vertical_min_m + iz_safe.to(z.dtype) * self.vertical_resolution_m
        dlon = self._wrap_delta_lon(lon.unsqueeze(-1) - node_lon)
        mid_lat = 0.5 * (lat.unsqueeze(-1) + node_lat)
        dx_km = dlon * torch.cos(torch.deg2rad(mid_lat)) * 111.195
        dy_km = (lat.unsqueeze(-1) - node_lat) * 111.195
        dz_m = z.unsqueeze(-1) - node_z
        exponent = ((dx_km.square() + dy_km.square()) /
                    self.horizontal_lengthscale.square() +
                    dz_m.square() / self.vertical_lengthscale.square())
        weight = torch.exp(-0.5 * exponent) * valid.to(lon.dtype)
        index = ((iz_safe * self.height + iy_safe) * self.width + ix)
        return index, weight


class SetConv3DOffToOn(_LocalGaussian3D):
    """Map paired irregular points to a density-normalised ``[B,D,Z,H,W]`` grid."""

    def aggregate(self, features, lon, lat, vertical,
                  vertical_type="geopotential_height", vertical_unit=None,
                  point_mask=None):
        if (features.ndim != 3 or lon.shape != features.shape[:2] or
                lat.shape != lon.shape or vertical.shape != lon.shape):
            raise ValueError(
                "Expected features [B,N,D] and lon/lat/vertical [B,N]"
            )
        index, weight = self._neighbours(
            lon, lat, vertical, vertical_type, vertical_unit
        )
        if point_mask is not None:
            weight = weight * point_mask.unsqueeze(-1).to(weight.dtype)
        accum_dtype = (torch.float32 if features.dtype in
                       (torch.float16, torch.bfloat16) else features.dtype)
        features = features.to(accum_dtype)
        weight = weight.to(accum_dtype)
        b, n, d = features.shape
        k = index.shape[-1]
        flat_size = self.depth * self.height * self.width
        density = features.new_zeros((b, flat_size))
        density.scatter_add_(1, index.reshape(b, n * k), weight.reshape(b, n * k))
        weighted = (features.unsqueeze(2) * weight.unsqueeze(-1)).reshape(b, n * k, d)
        latent_sum = features.new_zeros((b, flat_size, d))
        latent_sum.scatter_add_(
            1, index.reshape(b, n * k, 1).expand(-1, -1, d), weighted
        )
        latent_sum = latent_sum.transpose(1, 2).reshape(
            b, d, self.depth, self.height, self.width
        )
        density = density.reshape(b, 1, self.depth, self.height, self.width)
        return latent_sum, density

    def forward(self, features, lon, lat, vertical,
                vertical_type="geopotential_height", vertical_unit=None,
                point_mask=None):
        latent_sum, density = self.aggregate(
            features, lon, lat, vertical, vertical_type, vertical_unit,
            point_mask,
        )
        return latent_sum / density.clamp_min(self.eps), density


class SetConv3DOnToOff(_LocalGaussian3D):
    """Interpolate a ``[B,D,Z,H,W]`` grid at arbitrary 3-D query points."""

    def forward(self, grid, lon, lat, vertical,
                vertical_type="geopotential_height", vertical_unit=None):
        expected = (self.depth, self.height, self.width)
        if grid.ndim != 5 or grid.shape[-3:] != expected:
            raise ValueError(f"Expected grid [B,D,{self.depth},{self.height},{self.width}]")
        if lon.shape != lat.shape or vertical.shape != lon.shape:
            raise ValueError("Expected lon/lat/vertical with identical [B,N] shapes")
        index, weight = self._neighbours(
            lon, lat, vertical, vertical_type, vertical_unit
        )
        accum_dtype = (torch.float32 if grid.dtype in
                       (torch.float16, torch.bfloat16) else grid.dtype)
        grid = grid.to(accum_dtype)
        weight = weight.to(accum_dtype)
        b, d = grid.shape[:2]
        flat_size = self.depth * self.height * self.width
        flat = grid.reshape(b, d, flat_size).transpose(1, 2).reshape(b * flat_size, d)
        batch_offset = torch.arange(b, device=grid.device).view(b, 1, 1) * flat_size
        gathered = flat[index + batch_offset]
        numerator = (gathered * weight.unsqueeze(-1)).sum(2)
        return numerator / weight.sum(2, keepdim=True).clamp_min(self.eps)
