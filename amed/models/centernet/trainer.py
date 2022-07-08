from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from amed.utils import AverageMeter, get_logger


class Trainer:
    def __init__(
        self,
        cfg,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.writer = SummaryWriter(self.cfg.logdir)

        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.optimizer, self.scheduler = optimizer, scheduler

        self.best_loss = float("inf")
        self.losses = {"loss_center_heatmap": None, "loss_wh": None, "loss_offset": None}

    def fit(self, model: nn.Module):

        for epoch in range(self.cfg.epochs):
            model.train()

            for key in self.losses.keys():
                self.losses[key] = AverageMeter(f"train/{key}")
            self.losses["train/loss"] = AverageMeter("train/loss")

            with tqdm(self.train_loader) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.cfg.epochs}]")

                for images, gt_bboxes, gt_labels, imgs_shape in pbar:
                    images, imgs_shape = images.to(self.cfg.device), imgs_shape.to(self.cfg.device)
                    gt_bboxes, gt_labels = gt_bboxes.to(self.cfg.device), gt_labels.to(self.cfg.device)

                    feature = model(images)

                    losses = model.loss(feature, gt_bboxes, gt_labels, imgs_shape)
                    for key in self.losses.keys():
                        self.losses[key].update(losses[key].item())
                    self.losses["train/loss"].update(sum(losses.values().item()))

                    pbar.set_postfix(loss=losses.value)

                for key in self.losses.keys():
                    self.writer.add_scalar(key, self.losses[key].avg, epoch + 1)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch + 1)

            self.scheduler.step()
