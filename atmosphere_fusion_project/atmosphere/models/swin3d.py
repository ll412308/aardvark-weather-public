from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("Expected a 3-element window size")
    return tuple(int(x) for x in value)


def window_partition(x, window_size):
    """[B,D,H,W,C] -> [B*nW, wd*wh*ww, C]."""
    wd, wh, ww = window_size
    b, d, h, w, c = x.shape
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, wd * wh * ww, c
    )


def window_reverse(windows, window_size, b, d, h, w):
    wd, wh, ww = window_size
    c = windows.shape[-1]
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, c)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = int(dim)
        self.window_size = _triple(window_size)
        self.num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        wd, wh, ww = self.window_size
        table_size = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"
        ))
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += wd - 1
        relative[:, :, 1] += wh - 1
        relative[:, :, 2] += ww - 1
        relative[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative[:, :, 1] *= (2 * ww - 1)
        self.register_buffer(
            "relative_position_index", relative.sum(-1), persistent=False
        )
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(n, n, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


def _axis_slices(size, window, shift):
    if shift == 0:
        return (slice(0, size),)
    return (
        slice(0, -window),
        slice(-window, -shift),
        slice(-shift, None),
    )


class SwinBlock3D(nn.Module):
    """3-D shifted-window transformer block.

    Longitude is treated as periodic: shifted windows are allowed to communicate
    across the west/east boundary. Vertical and latitude wraparound are masked.
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 6), shift_size=(0, 0, 0),
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, periodic_longitude=True):
        super().__init__()
        self.dim = int(dim)
        self.window_size = _triple(window_size)
        self.shift_size = _triple(shift_size)
        for s, w in zip(self.shift_size, self.window_size):
            if not 0 <= s < w:
                raise ValueError("Each shift must satisfy 0 <= shift < window")
        self.periodic_longitude = bool(periodic_longitude)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, self.window_size, num_heads, qkv_bias, attn_drop, drop
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def _attention_mask(self, d, h, w, device, dtype):
        sd, sh, sw = self.shift_size
        if sd == sh == sw == 0:
            return None
        wd, wh, ww = self.window_size
        mask = torch.zeros((1, d, h, w, 1), device=device, dtype=dtype)
        d_slices = _axis_slices(d, wd, sd)
        h_slices = _axis_slices(h, wh, sh)
        # Longitude is a circle, so no artificial west/east boundary mask.
        w_slices = ((slice(0, w),) if self.periodic_longitude
                    else _axis_slices(w, ww, sw))
        count = 0
        for ds in d_slices:
            for hs in h_slices:
                for ws in w_slices:
                    mask[:, ds, hs, ws, :] = count
                    count += 1
        windows = window_partition(mask, self.window_size).squeeze(-1)
        attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

    def forward(self, x):
        # x: [B,C,D,H,W]
        b, c, d, h, w = x.shape
        shortcut = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm1(shortcut)
        wd, wh, ww = self.window_size
        pd = (wd - d % wd) % wd
        ph = (wh - h % wh) % wh
        pw = (ww - w % ww) % ww
        if pd or ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        dp, hp, wp = x.shape[1:4]
        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
        windows = window_partition(x, self.window_size)
        mask = self._attention_mask(dp, hp, wp, x.device, x.dtype)
        windows = self.attn(windows, mask=mask)
        x = window_reverse(windows, self.window_size, b, dp, hp, wp)
        if sd or sh or sw:
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(1, 2, 3))
        x = x[:, :d, :h, :w, :]
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 4, 1, 2, 3).contiguous()


class SwinForecastProcessor3D(nn.Module):
    """Residual one-step atmospheric evolution model on a 3-D latent grid."""

    def __init__(self, dim, depth=6, num_heads=6, window_size=(2, 7, 6),
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.1,
                 periodic_longitude=True):
        super().__init__()
        window_size = _triple(window_size)
        half_shift = tuple(max(w // 2, 0) for w in window_size)
        dpr = torch.linspace(0, float(drop_path), int(depth)).tolist()
        blocks = []
        for i in range(int(depth)):
            shift = (0, 0, 0) if i % 2 == 0 else half_shift
            blocks.append(SwinBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                periodic_longitude=periodic_longitude,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(dim)
        self.delta_head = nn.Conv3d(dim, dim, kernel_size=1)
        # Start close to persistence; this stabilizes early autoregressive training.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, state):
        x = state
        for block in self.blocks:
            x = block(x)
        y = x.permute(0, 2, 3, 4, 1)
        y = self.out_norm(y).permute(0, 4, 1, 2, 3).contiguous()
        return state + self.delta_head(y)
