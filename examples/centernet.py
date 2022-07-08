import argparse
import pathlib
import random
import sys

import torch
from torch import optim

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from amed.dataset import get_dataset, get_loader
from amed.models import CenterNet
from amed.models.centernet import Trainer, get_transforms


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int, help="plese set batch-size")
    parser.add_argument("--device", default="cuda:0", type=str, help="plese set device")
    parser.add_argument("--epochs", default=300, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=0.05, type=float, help="initial (base) learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument(
        "-r", "--root", default="/work/hara.e/AMED/lib/dataset/images", type=str, help="please set data root"
    )
    parser.add_argument("--seed", default=1, type=int, help="seed for initializing training")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)")

    return parser.parse_args()


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = make_parser()
    set_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    centernet = CenterNet(num_classes=4)

    train_transform, valid_transform = get_transforms("train"), get_transforms("valid")
    train_dataset, valid_dataset = get_dataset(root=args.root, train_transform=train_transform, valid_transform=valid_transform)
    train_loader, valid_loader = get_loader(
        train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=args.batch_size
    )

    optimizer = optim.SGD(centernet.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs / 4, eta_min=1e-6)

    trainer = Trainer(
        cfg=args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.fit(model=centernet)


if __name__ == "__main__":
    main()
