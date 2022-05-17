import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import config

from .dataset import AMEDDataset
from .models import YOLOX, YOLOv5
from .models.yolov5.utils import intersect_dicts


class Trainer:
    def __init__(self, model_name: str = "YOLOX"):
        self.cfg = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.pretrained_model(model_name)
        print(self.model)
        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((512, 512)),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((512, 512)),
                ]
            ),
        }
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

    def train(self):
        traindataset = AMEDDataset(root=self.cfg.DATASET.ROOT, transform=self.transform["train"])
        traindataloader = DataLoader(dataset=traindataset, batch_size=self.cfg.DATASET.BATCH_SIZE, shuffle=True, drop_last=True)

        for epoch in range(self.cfg.DATASET.TOTAL_EPOCH):
            self.model.train()
            total, running_loss, running_acc = 0, 0.0, 0.0

            with tqdm(enumerate(traindataloader, 0), total=len(traindataloader)) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.cfg.DATASET.TOTAL_EPOCH}]")

                for _, data in pbar:
                    images, bboxes = data
                    images, bboxes = images.to(self.device), bboxes.to(self.device)

                    self.optimizer.zero_grad()

                    preds = self.model(images)
                    print(preds[0].shape)

    def pretrained_model(self, model_name: str):
        if "yolov5" in model_name:
            ckpt = torch.load(f"lib/models/pretrained/{model_name}.pt")
            model = YOLOv5(cfg=f"lib/config/{model_name}.yaml")
            csd = ckpt["model"].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
            model.load_state_dict(csd, strict=False)
        elif "YOLOX" in model_name:
            model = YOLOX()
            model.load_state_dict(torch.load("lib/models/pretrained/yolox_s.pt", map_location="cpu"))
            print("loaded!")
        return model
