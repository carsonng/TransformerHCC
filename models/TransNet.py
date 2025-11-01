import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_


class CLSPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0]
        out = self.tanh(self.linear(cls_token))
        return out


class Classifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.classifier(self.layernorm(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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


class MultiInputBackbone(nn.Module):
    def __init__(self, num_inputs=5, initial_channels=32, final_channels=64):
        super().__init__()
        # 创建多输入的卷积块序列
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(1, initial_channels, stride=2),
                ConvBlock(initial_channels, final_channels, stride=1)
            ) for _ in range(num_inputs)
        ])
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, *inputs):
        features = []
        for i, x in enumerate(inputs):
            x = self.conv_blocks[i](x)
            x = self.gap(x).flatten(1)
            features.append(x)
        return torch.stack(features, dim=1)


class TransNet(nn.Module):
    def __init__(
            self,
            d_model=64,
            num_classes=2,
            max_len=6,
            num_layers=2,
            activation='gelu'
    ):
        super().__init__()
        self.d_model = d_model
        self.backbone = MultiInputBackbone(final_channels=d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, d_model // 16),
            dim_feedforward=d_model * 4,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )

        self.pooler = CLSPooling(d_model)
        self.classifier = Classifier(d_model, num_classes)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.backbone(x1, x2, x3, x4, x5)

        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.positional_encoding(x)


        x = self.transformer_encoder(x)

        x = self.pooler(x)
        x = self.classifier(x)

        return x






