import torch.nn as nn
import torch.nn.functional as F


class MatryoshkaLoss(nn.Module):
    def __init__(
        self,
        original_loss: nn.Module,
        matryoshka_dims: tuple[int] | int,
        matryoshka_weights: list[float] | None = None,
    ):
        super(MatryoshkaLoss, self).__init__()

        self.original_loss = original_loss
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights

        if isinstance(self.matryoshka_dims, int):
            if matryoshka_weights:
                raise ValueError(
                    "matryoshka_weights should not be set if matryoshka_dims is a single integer"
                )
            dims = [self.matryoshka_dims]
            min_dim = 16
            matryoshka_dim = self.matryoshka_dims
            while matryoshka_dim >= min_dim:
                matryoshka_dim //= 2
                dims.append(matryoshka_dim)

            self.matryoshka_dims = dims
        else:
            self.matryoshka_dims = matryoshka_dims
        if matryoshka_weights is None:
            self.matryoshka_weights = [1.0] * len(self.matryoshka_dims)

    def forward(self, features, no_sum=False):
        losses = []
        for dim in self.matryoshka_dims:
            l = self.original_loss(features[0][:, :dim], features[1][:, :dim])
            losses.append(l)

        if no_sum:
            return losses

        else:
            loss = 0.0

            for l, w in zip(losses, self.matryoshka_weights):
                loss += l * w

            return loss
