import torch
import torch.nn as nn


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self, nce_t):
        super(NCESoftmaxLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.nce_t = nce_t

    def forward(self, x_ret, y_ret):
        x, _ = x_ret
        y, _ = y_ret
        bsz = x.shape[0]
        scores = (
            (x / torch.norm(x, dim=1, keepdim=True))
            @ (y / torch.norm(y, dim=1, keepdim=True)).t()
            / self.nce_t
        )
        label = torch.arange(bsz, device=x.device)
        loss = self.loss(scores, label) + self.loss(scores.t(), label)
        return loss
