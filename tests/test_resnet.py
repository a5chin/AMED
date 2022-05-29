import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from amed.models.backbone import resnet18


def test_resnet():
    imgs = torch.randn(4, 3, 32, 32)
    model = resnet18(num_classes=4, pretrained=True)

    preds = model(imgs)
    assert preds is not None

    return model
