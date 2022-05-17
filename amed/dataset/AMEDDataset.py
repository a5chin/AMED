import pathlib
import sys
import warnings

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")
warnings.filterwarnings("ignore")

import cv2
import torch
from config import config

from .COCODataset import COCODataset


class AMEDDataset(COCODataset):
    def __init__(self, root, typ="train", transform=None, target_transform=None):
        super().__init__(root, typ, transform, target_transform)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)[0]
        file_name = coco.loadImgs(img_id)[0]["file_name"]

        bbox, category, age, sex = target["bbox"], target["category_id"], target["age"], target["sex"]

        img = cv2.imread(config.DATASET.IMAGES + file_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            bbox = self.target_transform(bbox)
        else:
            bbox = torch.tensor(bbox)

        return img, bbox
