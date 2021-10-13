import torch
from torch import nn

from lib.models import YOLOv5
from lib.models.yolov5.utils import intersect_dicts
from lib.models import YOLOX


class Trainer:
    def __init__(self, model_name: str='yolov5s'):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.pretrained_model(model_name)
        print(self.model)

    # def train(self):
    #     for epoch in

    def pretrained_model(self, model_name: str):
        ckpt = torch.load(f'lib/models/pretrained/{model_name}.pt')
        if 'yolov5' in model_name:
            model = YOLOv5(cfg=f'lib/config/{model_name}.yaml')
            csd = ckpt['model'].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
            model.load_state_dict(csd, strict=False)
        elif 'YOLOX' in model_name:
            model = YOLOX()
        return model
