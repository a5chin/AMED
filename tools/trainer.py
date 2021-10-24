import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from lib.models import YOLOv5
from lib.models.yolov5.utils import intersect_dicts
from lib.models import YOLOX
from lib.dataset import AMEDDataset
from lib.config import config


class Trainer:
    def __init__(self, model_name: str='yolov5s'):
        self.cfg = config
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.pretrained_model(model_name)
        self.transform = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def train(self):
        traindataset = AMEDDataset(root=self.cfg.DATASET.ROOT, transform=self.transform['train'])
        traindataloader = DataLoader(
            dataset=traindataset,
            batch_size=self.cfg.DATASET.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        for epoch in range(self.cfg.DATASET.TOTAL_EPOCH):
            self.model.train()
            total, running_loss, running_acc = 0, 0.0, 0.0

            with tqdm(enumerate(traindataloader, 0), total=len(traindataloader)) as pbar:
                pbar.set_description('[Epoch %d/%d]' % (epoch + 1, self.cfg.DATASET.TOTAL_EPOCH))

                for _, data in pbar:
                    print(data)

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
