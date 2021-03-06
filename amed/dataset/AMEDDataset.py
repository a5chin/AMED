from typing import Tuple

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import config

from .COCODataset import COCODataset


class AMEDDataset(COCODataset):
    def __init__(
        self, root, typ="train", transform=None, target_transform=None
    ) -> None:
        super().__init__(root, typ, transform, target_transform)

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)[0]
        file_name = coco.loadImgs(img_id)[0]["file_name"]

        bbox, category, age, sex = (
            target["bbox"],
            target["category_id"],
            target["age"],
            target["sex"],
        )

        img = cv2.imread(
            config.DATASET.IMAGES + file_name,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            bbox = self.target_transform(bbox)
        else:
            bbox = torch.tensor(bbox)

        category = torch.tensor(category)
        img_shape = torch.tensor(img.shape)

        return img, bbox, category, img_shape


def get_dataset(
    root: str, train_transform: transforms.Compose, valid_transform: transforms.Compose
) -> Tuple[Dataset, Dataset]:
    traindataset = AMEDDataset(root=root, typ="train", transform=train_transform)
    valdataset = AMEDDataset(root=root, typ="validation", transform=valid_transform)

    return traindataset, valdataset


def get_loader(
    train_dataset: Dataset, valid_dataset: Dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, valid_loader
