from typing import Dict

import torch
from torch import nn


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
