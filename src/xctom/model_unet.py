from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two convolution blocks:
    Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU

    GroupNorm is used instead of BatchNorm because microscopy/tomography
    projects often use small batch sizes. BatchNorm can become unstable
    when batch size is small.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()

        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Safety padding/cropping for odd image sizes.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    General 2D U-Net for paired XCT -> OM image translation.

    Input:
        [B, input_channels, H, W]

    Output:
        [B, output_channels, H, W], values in [0, 1] because of sigmoid.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if depth < 2:
            raise ValueError("UNet depth must be >= 2.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth

        channels = [base_channels * (2 ** i) for i in range(depth)]

        self.inc = DoubleConv(input_channels, channels[0], dropout=0.0)

        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(
                DownBlock(
                    channels[i - 1],
                    channels[i],
                    dropout=dropout,
                )
            )

        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(depth - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1]
            out_ch = reversed_channels[i + 1]
            self.ups.append(
                UpBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout=dropout,
                )
            )

        self.outc = nn.Conv2d(channels[0], output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips_for_decoder = list(reversed(skips[:-1]))

        for up, skip in zip(self.ups, skips_for_decoder):
            x = up(x, skip)

        x = self.outc(x)
        return torch.sigmoid(x)