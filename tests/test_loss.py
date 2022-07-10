import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.as_posix()))

from amed.losses import GaussianFocalLoss, L1Loss


def test_l1loss():
    feature = torch.randn(16, 4, 128, 128)
    target = torch.randn(16, 4, 128, 128)

    criterion = L1Loss()

    losses = criterion(feature, target).sum(dim=(3, 2, 1))
    loss = losses.mean()

    assert loss.item() > 0


def test_gaussian_focal_loss():
    heatmap = torch.randn(16, 2, 128, 128)
    center_heatmap_target = torch.randn(16, 2, 128, 128)

    criterion = GaussianFocalLoss()

    losses = criterion(heatmap, center_heatmap_target).sum(dim=(3, 2, 1))
    loss = losses.mean()

    assert loss is not None
