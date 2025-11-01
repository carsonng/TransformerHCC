import torch
import torch.nn as nn


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


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(p=0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class ConvLSTMNet(nn.Module):
    def __init__(self, num_inputs=5, initial_channels=32, final_channels=64, lstm_hidden=32, num_classes=2):
        super(ConvLSTMNet, self).__init__()

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(1, initial_channels),
                ConvBlock(initial_channels, final_channels)
            ) for _ in range(num_inputs)
        ])

        self.lstm1 = LSTMBlock(final_channels, lstm_hidden)
        self.lstm2 = LSTMBlock(final_channels, lstm_hidden)

        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, *inputs):
        features = []

        for i, x in enumerate(inputs):
            x = self.conv_blocks[i](x)
            x = self.gap(x).flatten(1)
            features.append(x)

        combined = torch.stack(features, dim=1)

        out = self.lstm1(combined)
        out = self.lstm2(out)

        x = out[:, -1, :]
        x = self.classifier(x)

        return x




