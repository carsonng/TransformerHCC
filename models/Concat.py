import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False
        )
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class Concat(nn.Module):
    def __init__(self, num_inputs=5, initial_channels=32, final_channels=64, num_classes=2):
        super(Concat, self).__init__()

        self.strided_blocks = nn.ModuleList([
            ConvBlock(1, initial_channels, stride=2) for _ in range(num_inputs)
        ])
        self.conv_blocks = nn.ModuleList([
            ConvBlock(initial_channels, final_channels, stride=1) for _ in range(num_inputs)
        ])

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(final_channels * num_inputs, num_classes)

    def forward(self, *inputs):
        features = []

        for i, x in enumerate(inputs):
            x = self.strided_blocks[i](x)
            x = self.conv_blocks[i](x)
            x = self.gap(x).flatten(1)
            features.append(x)

        x = torch.cat(features, dim=1)
        x = self.classifier(x)

        return x




