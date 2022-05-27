import torch
from torch import nn

from ..backbone.network_blocks import BaseConv
from .modules import Encoder, Predictor


class SimSiam(nn.Module):
    """Implementation of SimSiam[0] network
    Recommended loss: :py:class:`lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss`
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head. This should
            be `num_ftrs` / 4.
        out_dim:
            Dimension of the output (after the projection head).
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim: int = 1376256,
        pred_dim: int = 512,
    ) -> None:

        super().__init__()

        self.dim = dim
        self.pred_dim = pred_dim
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()

        self.backbone = backbone
        self.encoder = Encoder(dim=dim)
        self.predictor = Predictor(dim=dim, pred_dim=pred_dim)
        for in_channel in [256, 512, 1024]:
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channel),
                    out_channels=int(256),
                    ksize=1,
                    stride=1,
                    act="silu",
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(
                            in_channels=int(256),
                            out_channels=int(256),
                            ksize=3,
                            stride=1,
                            act="silu",
                        ),
                        BaseConv(
                            in_channels=int(256),
                            out_channels=int(256),
                            ksize=3,
                            stride=1,
                            act="silu",
                        ),
                    ]
                )
            )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """Forward pass through SimSiam.
        Extracts features with the backbone and applies the projection
        head and prediction head to the output space. If both x0 and x1 are not
        None, both will be passed through the backbone, projection, and
        prediction head. If x1 is None, only x0 will be forwarded.
        Args:
            x0: first views of images
            x1: second views of images
        Returns:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        Examples:
            >>> p0, p1, z0, z1 = model(x1=images[0], x2=images[1])
        """
        # f0 = self.backbone(x0).flatten(start_dim=1)
        f0 = self.backbone(x0)
        for i, (cls_conv, key) in enumerate(zip(self.cls_convs, f0.keys())):
            cls_x = self.stems[i](f0[key])
            f0[key] = cls_conv(cls_x).flatten(start_dim=1)
        f0 = torch.cat([v for v in f0.values()], dim=1)
        print(f0.shape)

        # f1 = self.backbone(x1).flatten(start_dim=1)
        f1 = self.backbone(x1)
        for i, (cls_conv, key) in enumerate(zip(self.cls_convs, f1.keys())):
            cls_x = self.stems[i](f1[key])
            f1[key] = cls_conv(cls_x).flatten(start_dim=1)
        f1 = torch.cat([v for v in f1.values()], dim=1)

        z0 = self.encoder(f0)
        z1 = self.encoder(f1)

        p0 = self.predictor(z0)
        p1 = self.predictor(z1)

        return p0, p1, z0.detach(), z1.detach()