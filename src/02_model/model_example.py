import torch
from torch import nn
import torch.nn.functional as F


def _safe_group_norm(num_groups: int, num_channels: int) -> nn.GroupNorm:
    # Make GroupNorm robust even when channels < groups or not divisible.
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvBlock(nn.Module):
    """
    Conv -> Norm -> ReLU -> Dropout (optional) repeated twice.
    Adding dropout here (not only at bottleneck) helps generalization.
    """
    def __init__(self, in_ch, out_ch, norm="bn", dropout=0.0):
        super().__init__()
        if norm == "bn":
            Norm = lambda c: nn.BatchNorm2d(c)
        else:
            Norm = lambda c: _safe_group_norm(8, c)

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet3(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base=32,
        norm="bn",
        dropout=0.2,
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base, norm, dropout=dropout * 0.5)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2, norm, dropout=dropout * 0.5)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base * 2, base * 4, norm, dropout=dropout * 0.5)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base * 4, base * 8, norm, dropout=dropout),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = ConvBlock(base * 8, base * 4, norm, dropout=dropout * 0.5)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base * 2, norm, dropout=dropout * 0.5)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base, norm, dropout=dropout * 0.5)

        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)