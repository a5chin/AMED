import argparse
import pathlib
import sys
import warnings
from typing import Dict

import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")
warnings.filterwarnings("ignore")

from amed.dataset import get_dataset, get_loader
from amed.models import SimSiam
from amed.models.simsiam import get_transforms
from amed.models.backbone import CSPDarknet


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="plese set batch-size")
    parser.add_argument("-d", "--device", default="cuda:0", type=str, help="plese set device")
    parser.add_argument("-e", "--epochs", default=300, type=int, help="number of total epochs to run")
    parser.add_argument("-l", "--learning-rate", default=0.05, type=float, help="initial (base) learning rate", dest="lr")
    parser.add_argument("-m", "--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("-r", "--root", default="/work/hara.e/AMED/lib/dataset/cutout/images", type=str, help="please set data root")
    parser.add_argument("-s", "--seed", default=1, type=int, help="seed for initializing training")
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    return parser.parse_args()

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def train(args: Dict, device: str, model: nn.Module, criterion, optimizer, train_loader: DataLoader, valid_loader: DataLoader) -> None:
    for epoch in range(args.epochs):
        model.train()

        for images, _ in tqdm(train_loader):
            x0, x1 = images[0].to(device), images[1].to(device)
            p1, p2, z1, z2 = model(x0=x0, x1=x1)


def main():
    args = make_parser()
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    backbone = CSPDarknet()
    simsiam = SimSiam(backbone=backbone)
    simsiam.to(device)
    optim_params = [
        {'params': simsiam.encoder.parameters(), 'fix_lr': False},
        {'params': simsiam.predictor.parameters(), 'fix_lr': True}
    ]

    train_transform, valid_transform = get_transforms()
    train_dataset, valid_dataset = get_dataset(
        root=args.root,
        train_transform=train_transform,
        valid_transform=valid_transform
    )
    train_loader, valid_loader = get_loader(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=args.batch_size
    )

    init_lr = args.lr * args.batch_size / 256
    criterion = nn.CosineSimilarity(dim=1).to(device=args.device)
    optimizer = torch.optim.SGD(
        optim_params,
        init_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train(
        args=args,
        device=device,
        model=simsiam,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader
    )


if __name__ == "__main__":
    main()
