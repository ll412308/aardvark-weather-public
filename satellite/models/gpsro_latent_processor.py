"""GPSRO wrapper around the two-dimensional satellite latent processor."""

from torch import nn

from .latent_processor import LatentGridProcessor


class GPSROLatentProcessor(nn.Module):
    """Compress a 3-D GPSRO grid to a common 2-D spatial latent.

    The SetConv height axis is folded into the channel axis before the same
    Swin-style processor used by the radiance autoencoder. ``forward`` returns
    the 2-D latent intended for later instrument fusion and forecasting.
    ``restore_3d`` is used only on the GPSRO decoder path.
    """

    def __init__(self, in_dim, latent_dim, grid_depth, grid_height, grid_width,
                 processor_dim=128, patch_size=0, patch_min_height=64,
                 patch_min_width=128, depth=2, num_heads=4, window_size=5,
                 enabled=True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.grid_depth = int(grid_depth)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)

        self.processor = LatentGridProcessor(
            in_dim=self.in_dim * self.grid_depth,
            latent_dim=self.latent_dim,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            processor_dim=processor_dim,
            patch_size=patch_size,
            patch_min_height=patch_min_height,
            patch_min_width=patch_min_width,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            enabled=enabled,
        )
        # The common 2-D latent has no explicit height axis. The GPSRO decoder
        # learns to expand it back to one feature vector per vertical grid node.
        self.restore_projection = nn.Conv2d(
            self.latent_dim,
            self.latent_dim * self.grid_depth,
            kernel_size=1,
        )

    def forward(self, x):
        expected = (self.grid_depth, self.grid_height, self.grid_width)
        if x.ndim != 5 or x.shape[1] != self.in_dim or x.shape[-3:] != expected:
            raise ValueError(
                f"Expected GPSRO grid [B,{self.in_dim},{self.grid_depth},"
                f"{self.grid_height},{self.grid_width}], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        # [B,D,Z,H,W] -> [B,D*Z,H,W]. Each channel now identifies both a
        # feature type and a vertical level; latitude/longitude remain spatial.
        x = x.reshape(
            batch, self.in_dim * self.grid_depth,
            self.grid_height, self.grid_width,
        )
        # print('reshaped x shape:', x.shape)
        return self.processor(x)

    def restore_3d(self, latent):
        """Expand the common [B,D,H,W] latent for SetConv3D OnToOff."""
        expected = (self.grid_height, self.grid_width)
        if (latent.ndim != 4 or latent.shape[1] != self.latent_dim or
                latent.shape[-2:] != expected):
            raise ValueError(
                f"Expected common latent [B,{self.latent_dim},"
                f"{self.grid_height},{self.grid_width}], got {tuple(latent.shape)}"
            )
        batch = latent.shape[0]
        latent = self.restore_projection(latent)
        return latent.reshape(
            batch, self.latent_dim, self.grid_depth,
            self.grid_height, self.grid_width,
        )

    def extra_repr(self):
        return (
            f"3d_input_channels={self.in_dim}, grid_depth={self.grid_depth}, "
            f"common_latent_dim={self.latent_dim}"
        )
