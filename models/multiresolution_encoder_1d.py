from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_PADDING_MODES = {"circular", "zeros", "reflect", "replicate"}


def _sanitize_padding_mode(mode: str) -> str:
    mode = str(mode).lower()
    return mode if mode in _VALID_PADDING_MODES else "circular"


def _split_channels(total_dim: int, n_scales: int) -> List[int]:
    total_dim = int(max(1, total_dim))
    n_scales = int(max(1, n_scales))
    base = total_dim // n_scales
    rem = total_dim % n_scales
    return [base + (1 if i < rem else 0) for i in range(n_scales)]


class SymmetricDilatedConv1d(nn.Module):
    """Symmetric temporal convolution + local spatial convolution.

    Input : [B, C_in, T, X]
    Output: [B, C_out, T, X]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        spatial_padding_mode: str = "circular",
    ):
        super().__init__()
        pad_mode = _sanitize_padding_mode(spatial_padding_mode)

        self.conv_x = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(dilation),
            dilation=int(dilation),
            padding_mode=pad_mode,
        )
        self.conv_t = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.norm = nn.LayerNorm(int(out_channels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, X = x.shape

        x_space = x.permute(0, 2, 1, 3).reshape(B * T, C, X)
        out_x = self.conv_x(x_space)
        C_out = out_x.shape[1]
        out_x = out_x.reshape(B, T, C_out, X).permute(0, 2, 1, 3)

        x_time = x.permute(0, 3, 1, 2).reshape(B * X, C, T)
        out_t = self.conv_t(x_time)
        out_t = out_t.reshape(B, X, C_out, T).permute(0, 2, 3, 1)

        out = out_x + out_t
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return self.act(out)


class MRScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, spatial_padding_mode: str = "circular"):
        super().__init__()
        self.conv1 = SymmetricDilatedConv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = SymmetricDilatedConv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.skip = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1) if int(in_channels) != int(out_channels) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        return self.conv2(self.conv1(x)) + residual


class MultiResolutionEncoder1D(nn.Module):
    """1D Multi-Resolution Spatiotemporal Feature Extractor.

    Input : [B, T, X, C_in]
    Output: [B, T, X, d_h]
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_h: int = 128,
        n_scales: int = 3,
        spatial_padding_mode: str = "circular",
    ):
        super().__init__()
        self.d_h = int(d_h)
        self.n_scales = max(1, int(n_scales))
        self.in_channels = int(in_channels)
        self.spatial_padding_mode = _sanitize_padding_mode(spatial_padding_mode)

        self.scale_dims = _split_channels(self.d_h, self.n_scales)
        self.scale_blocks = nn.ModuleList(
            [
                MRScaleBlock(
                    self.in_channels,
                    out_dim,
                    dilation=2 ** l,
                    spatial_padding_mode=self.spatial_padding_mode,
                )
                for l, out_dim in enumerate(self.scale_dims)
            ]
        )
        self.scale_adapters = nn.ModuleList(
            [nn.Conv2d(out_dim, out_dim, kernel_size=1) for out_dim in self.scale_dims]
        )

        fusion_dim = int(sum(self.scale_dims))
        self.proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.d_h),
            nn.GELU(),
            nn.Linear(self.d_h, self.d_h),
        )

    def _feature_concat(self, x: torch.Tensor) -> torch.Tensor:
        B, T, X, C = x.shape
        x_conv = x.permute(0, 3, 1, 2)

        scale_features = []
        for l, (block, adapter) in enumerate(zip(self.scale_blocks, self.scale_adapters)):
            if l > 0:
                sf = 2 ** l
                x_down = F.avg_pool2d(x_conv, kernel_size=(1, sf), stride=(1, sf))
            else:
                x_down = x_conv

            feat = block(x_down)
            if l > 0:
                feat = F.interpolate(feat, size=(T, X), mode="bilinear", align_corners=False)
            scale_features.append(adapter(feat))

        return torch.cat(scale_features, dim=1).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self._feature_concat(x))


if __name__ == "__main__":
    model = MultiResolutionEncoder1D(in_channels=1, d_h=128, n_scales=3)
    x = torch.randn(4, 101, 64, 1)
    out = model(x)
    print(f"MultiResolutionEncoder1D: input {tuple(x.shape)} -> output {tuple(out.shape)}")

