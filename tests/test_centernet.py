import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from amed.models.centernet import CenterNet


def test_model():
    images = torch.randn(16, 3, 512, 512)
    model = CenterNet()

    out = model(images)

    assert out["heatmap"].shape == (16, 4, 128, 128)
    assert out["wh"].shape == (16, 2, 128, 128)
    assert out["offset"].shape == (16, 2, 128, 128)
