from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from amed.utils import AverageMeter


class Trainer:
    def __init__(self, cfg, train_loader, valid_loader, criterion, optimizer, scheduler):
        self.cfg = cfg
        self.train_writer = SummaryWriter(Path(self.cfg.logdir) / "train")
        self.valid_writer = SummaryWriter(Path(self.cfg.logdir) / "valid")

        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion, self.optimizer, self.scheduler = criterion, optimizer, scheduler

    def fit(self, model):
        losses = AverageMeter("train_loss")

        for epoch in range(self.cfg.epochs):
            model.train()

            with tqdm(self.train_loader) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.cfg.epochs}]")

                for images, _ in pbar:
                    x0, x1 = images[0].to(self.cfg.device), images[1].to(self.cfg.device)
                    out0, out1 = model(x0=x0, x1=x1)

                    loss = (self.criterion(*out0).mean() + self.criterion(*out1).mean()) / 2
                    losses.update(loss.item())

                    pbar.set_postfix(loss=losses.value)
                self.train_writer.add_scalar(losses.avg, epoch)

            self.evaluate(model, epoch)

    def evaluate(self, model, epoch) -> None:
        model.eval()
        losses = AverageMeter("valid_loss")

        for images, _ in self.valid_loader:
            x0, x1 = images[0].to(self.cfg.device), images[1].to(self.cfg.device)
            out0, out1 = model(x0=x0, x1=x1)

            loss = (self.criterion(*out0).mean() + self.criterion(*out1).mean()) / 2
            losses.update(loss.item())

        self.valid_writer.add_scalar(losses.avg, epoch)
