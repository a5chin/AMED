from typing import Dict

import torch
from torch import nn


class CenterNetHead(nn.Module):
    def __init__(self, num_classes: int, down_ratio: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.down_ratio = down_ratio
        self.down_sampler = nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=(3, 3),
            padding=1,
            stride=down_ratio,
            bias=True,
        )

        self.head_heatmap = nn.Conv2d(embedding_dim, num_classes, kernel_size=(3, 3), padding=1, bias=True)
        self.head_heatmap.bias.data.fill_(-4.0)
        self.head_width_height = nn.Conv2d(embedding_dim, 2, kernel_size=(3, 3), padding=1, bias=True)
        self.head_offset_regularizer = nn.Conv2d(embedding_dim, 2, kernel_size=(3, 3), padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = torch.relu_(self.down_sampler(torch.relu_(x)))

        value = {
            "hm": self.head_heatmap(features).sigmoid_(),
            "wh": self.head_width_height(features),
            "offset": self.head_offset_regularizer(features),
        }

        return value


class CenterNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        head: nn.Module,
        decoder,
        down_ratio: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scaling_factor = down_ratio

        self.backbone = backbone
        self.head = head
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.backbone(x)
        value = self.head(output)

        return value
