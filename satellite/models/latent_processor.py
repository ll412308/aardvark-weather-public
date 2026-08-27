"""Lightweight Swin-style processor for the regular satellite latent grid."""

import torch
import torch.nn.functional as F
from torch import nn


def _choose_patch_size(height, width, patch_size, min_height, min_width):
    if patch_size > 0:
        return int(patch_size)
    if height >= min_height and width >= min_width:
        return 2
    return 1


def _window_partition(x, window_size):
    """[B,H,W,C] -> [num_windows*B, window_size*window_size, C]."""
    b, h, w, c = x.shape
    x = x.view(
        b, h // window_size, window_size,
        w // window_size, window_size, c,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window_size * window_size, c
    )


def _window_partition_mask(mask, window_size):
    """[B,H,W] bool -> [num_windows*B, window_size*window_size] bool."""
    b, h, w = mask.shape
    mask = mask.view(
        b, h // window_size, window_size,
        w // window_size, window_size,
    )
    return mask.permute(0, 1, 3, 2, 4).reshape(
        -1, window_size * window_size
    )


def _window_reverse(windows, window_size, batch, height, width):
    """Reverse _window_partition back to [B,H,W,C]."""
    x = windows.view(
        batch, height // window_size, width // window_size,
        window_size, window_size, -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


def _latitude_shift_mask(height, width, window_size, shift_size, device):
    """Mask latitude wrap-around while leaving longitude periodic."""
    region = torch.zeros((1, height, width, 1), device=device)
    height_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    for region_id, height_slice in enumerate(height_slices):
        region[:, height_slice, :, :] = region_id
    region = _window_partition(region, window_size).squeeze(-1)
    return region.unsqueeze(1) != region.unsqueeze(2)


class WindowAttentionBlock(nn.Module):
    """Small Swin-like block with regular and shifted window attention."""

    def __init__(self, dim, num_heads=4, window_size=5, shifted=False):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shifted = bool(shifted)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        b, h, w, c = x.shape
        ws = min(self.window_size, h, w)
        if ws < 1:
            return x
        shift = ws // 2 if self.shifted and ws > 1 and h > ws and w > ws else 0
        shortcut = x
        x = self.norm1(x)
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        valid = x.new_ones((b, h, w), dtype=torch.bool)
        if pad_h or pad_w:
            valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)
        hp, wp = h + pad_h, w + pad_w
        if shift:
            # Shift both latent spatial axes, as in a standard Swin block.
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            valid = torch.roll(valid, shifts=(-shift, -shift), dims=(1, 2))
        windows = _window_partition(x, ws)
        key_padding_mask = None
        if pad_h or pad_w:
            # True entries are ignored by MultiheadAttention.
            key_padding_mask = ~_window_partition_mask(valid, ws)
        attention_mask = None
        if shift:
            # Latitude is not periodic, so rolled north/south edges must not
            # attend to each other. Longitude is periodic and is left unmasked.
            attention_mask = _latitude_shift_mask(
                hp, wp, ws, shift, x.device
            )
            # MultiheadAttention expects one mask per window and attention head.
            attention_mask = attention_mask.repeat(b, 1, 1)
            attention_mask = attention_mask.repeat_interleave(
                self.num_heads, dim=0
            )
        windows, _ = self.attn(
            windows, windows, windows,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = _window_reverse(windows, ws, b, hp, wp)
        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        x = x[:, :h, :w, :]
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class LatentGridProcessor(nn.Module):
    """Project, process, and restore a SetConv latent grid.

    Input and output are both [B, latent_dim, grid_height, grid_width]. Internally
    the grid may be resized to an even latitude count, patch embedded, processed
    with local window attention, and then interpolated back.
    """

    def __init__(self, in_dim, latent_dim, grid_height, grid_width,
                 processor_dim=128, patch_size=0, patch_min_height=64,
                 patch_min_width=128, depth=2, num_heads=4, window_size=5,
                 enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.work_height = self.grid_height - 1 if self.grid_height % 9 != 0 else self.grid_height
        self.work_width = self.grid_width
        self.patch_size = _choose_patch_size(
            self.work_height, self.work_width, int(patch_size),
            int(patch_min_height), int(patch_min_width),
        )
        if self.work_height % self.patch_size != 0 or self.work_width % self.patch_size != 0:
            self.patch_size = 1
        self.input_projection = nn.Conv2d(in_dim, latent_dim, kernel_size=1)
        if not self.enabled:
            self.processor = nn.Identity()
            return

        self.patch_embedding = nn.Conv2d(
            latent_dim, processor_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )
        self.blocks = nn.ModuleList([
            WindowAttentionBlock(
                processor_dim, num_heads=num_heads, window_size=window_size,
                shifted=bool(i % 2),
            )
            for i in range(int(depth))
        ])
        self.output_projection = nn.Conv2d(processor_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.input_projection(x)
        if not self.enabled:
            return x

        original_size = x.shape[-2:]
        if original_size != (self.work_height, self.work_width):
            x = F.interpolate(
                x, size=(self.work_height, self.work_width),
                mode="bilinear", align_corners=False,
            )
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 1, 2)

        if self.patch_size > 1:
            x = F.interpolate(
                x, size=(self.work_height, self.work_width),
                mode="bilinear", align_corners=False,
            )
        x = self.output_projection(x)
        if x.shape[-2:] != original_size:
            x = F.interpolate(
                x, size=original_size, mode="bilinear", align_corners=False
            )
        return x

    def extra_repr(self):
        return (
            f"enabled={self.enabled}, work_size=({self.work_height}, "
            f"{self.work_width}), patch_size={self.patch_size}"
        )
