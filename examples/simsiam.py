import argparse
import pathlib
import random
import sys
import warnings

import torch
from torch import nn, optim

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")
warnings.filterwarnings("ignore")

from amed.dataset import get_dataset, get_loader
from amed.models import SimSiam
from amed.models.backbone import resnet18
from amed.models.simsiam import get_transforms
from amed import Trainer


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="plese set batch-size")
    parser.add_argument("-e", "--epochs", default=300, type=int, help="number of total epochs to run")
    parser.add_argument("-l", "--learning-rate", default=0.05, type=float, help="initial (base) learning rate", dest="lr")
    parser.add_argument("-m", "--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument(
        "-r", "--root", default="/work/hara.e/AMED/lib/dataset/cutout/images", type=str, help="please set data root"
    )
    parser.add_argument("-s", "--seed", default=1, type=int, help="seed for initializing training")
    parser.add_argument("-w", "--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)")

    parser.add_argument("--fix-pred-lr", action="store_true", help="fix learning rate for the predictor")

    return parser.parse_args()


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = make_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    backbone = resnet18(num_classes=4, pretrained=True)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    simsiam = SimSiam(backbone=backbone, dim=feat_dim)
    simsiam.to(args.device)

    if args.fix_pred_lr:
        optim_params = [
            {"params": simsiam.module.encoder.parameters(), "fix_lr": False},
            {"params": simsiam.module.predictor.parameters(), "fix_lr": True},
        ]
    else:
        optim_params = simsiam.parameters()

    train_transform, valid_transform = get_transforms("train"), get_transforms("valid")
    train_dataset, valid_dataset = get_dataset(root=args.root, train_transform=train_transform, valid_transform=valid_transform)
    train_loader, valid_loader = get_loader(
        train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=args.batch_size
    )

    init_lr = args.lr * args.batch_size / 256
    criterion = nn.CosineSimilarity(dim=1).to(device=args.device)
    optimizer = optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs / 4, eta_min=5e-5)

    trainer = Trainer(
        cfg=args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer.fit(model=simsiam)


if __name__ == "__main__":
    main()
