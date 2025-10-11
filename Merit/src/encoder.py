# src/encoder.py

import torch
import torch.nn as nn

class TCNEncoder(nn.Module):
    """
    논문에서 사용한 시계열 인코더 (TCN 기반).
    입력: (배치 크기, 채널 수, 시계열 길이)
    출력: (배치 크기, 표현 벡터 차원)
    """
    def __init__(self, input_channels=1, output_dim=320, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal Convolution을 위한 패딩 계산
            padding = (kernel_size - 1) * dilation_size
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        self.network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # 입력 shape: (N, C, L)
        # TCN은 (N, C, L) 입력을 가정합니다. 데이터로더에서 맞춰줘야 합니다.
        out = self.network(x)
        out = self.gap(out).squeeze(-1) # (N, num_channels[-1])
        return self.fc(out)
