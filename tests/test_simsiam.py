import sys
from pathlib import Path

import torch
from torch import nn

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from test_resnet import test_resnet

from amed.models import SimSiam
from amed.models.simsiam import get_transforms


def test_transforms():
    transforms = get_transforms("train")
    imgs = torch.randn(3, 64, 64)
    two_imgs = transforms(imgs)
    assert two_imgs[0].shape == two_imgs[1].shape


def test_simsiam():
    imgs = torch.randn(4, 3, 64, 64)

    backbone = test_resnet()
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    simsiam = SimSiam(backbone=backbone, dim=feat_dim)

    out0, out1 = simsiam(imgs[:2], imgs[2:])
    assert (out0[0].shape == out0[1].shape) and (out1[0].shape == out1[1].shape)
