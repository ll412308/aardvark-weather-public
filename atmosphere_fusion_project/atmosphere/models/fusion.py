from __future__ import annotations

import torch
from torch import nn


class ResidualConv3dBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class InstrumentFusion(nn.Module):
    """Confidence-aware attention across instrument latent representations."""

    def __init__(self, instrument_names, atmosphere_dim: int, refine_blocks: int = 2):
        super().__init__()
        self.instrument_names = tuple(instrument_names)
        self.score_head = nn.Sequential(
            nn.Conv3d(atmosphere_dim + 1, atmosphere_dim // 2, 1),
            nn.SiLU(),
            nn.Conv3d(atmosphere_dim // 2, 1, 1),
        )
        self.score_bias = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(())) for name in self.instrument_names
        })
        self.refine = nn.Sequential(*[
            ResidualConv3dBlock(atmosphere_dim) for _ in range(int(refine_blocks))
        ])

    def forward(self, tokens, confidences, available):
        token_stack, score_stack = [], []
        any_available = None
        for name in self.instrument_names:
            x = tokens[name]
            c = confidences[name]
            a = available[name].bool()
            any_available = a if any_available is None else (any_available | a)
            score = self.score_head(torch.cat([x, c], dim=1))
            # Confidence acts as a soft prior on which instrument is trusted locally.
            score = score + torch.log(c.clamp_min(1.0e-6)) + self.score_bias[name]
            score = score.masked_fill(~a.view(-1, 1, 1, 1, 1), -1.0e4)
            token_stack.append(x)
            score_stack.append(score)
        if not bool(any_available.all()):
            raise ValueError("Every batch item must contain at least one available instrument")
        token_stack = torch.stack(token_stack, dim=1)   # [B,K,C,L,H,W]
        score_stack = torch.stack(score_stack, dim=1)   # [B,K,1,L,H,W]
        weights = torch.softmax(score_stack, dim=1)
        fused = (weights * token_stack).sum(dim=1)
        fused = self.refine(fused)
        weight_dict = {
            name: weights[:, i] for i, name in enumerate(self.instrument_names)
        }
        return fused, weight_dict
