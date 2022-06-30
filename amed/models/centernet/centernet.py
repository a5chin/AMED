from typing import Dict

import torch
from torch import nn

from amed.losses import GaussianFocalLoss, L1Loss

from .backbone import resnet18
from .modules import CenterNetHead, CTResNetNeck


class CenterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.neck = CTResNetNeck(
            in_channels=512,
            num_deconv_filters=(256, 128, 64),
            num_deconv_kernels=(4, 4, 4),
        )
        self.bbox_head = CenterNetHead(in_channels=64, feat_channels=64, num_classes=4)

        self.loss_center_haetmap = GaussianFocalLoss(loss_weight=1.0)
        self.loss_wh = L1Loss(loss_weight=0.1)
        self.loss_offset = L1Loss(loss_weight=1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.neck(x)
        feature = self.bbox_head(x)

        loss_center_heatmap = self.loss_center_haetmap(feature["heatmap"])
        loss_wh = self.loss_wh(["wh"])
        loss_offset = self.loss_offset(["offset"])

        return {
            "los_center_heatmap": loss_center_heatmap,
            "loss_wh": loss_wh,
            "loss_offset": loss_offset,
        }
