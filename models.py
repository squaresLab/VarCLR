import os
from collections import deque
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class CosineSimMarginLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.loss = nn.MarginRankingLoss(margin=self.delta, reduction="none")

    def forward(self, x, y, x_neg, y_neg):
        pos = F.cosine_similarity(x, y)
        neg1 = F.cosine_similarity(x, x_neg)
        neg2 = F.cosine_similarity(y, y_neg)
        ones = torch.ones(pos.shape[0], device=x.device)
        loss = self.loss(pos, neg1, ones) + self.loss(pos, neg2, ones)
        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self, nce_t):
        super(NCESoftmaxLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.nce_t = nce_t

    def forward(self, x, y, x_neg, y_neg):
        bsz = x.shape[0]
        scores = (
            (x / torch.norm(x, dim=1, keepdim=True))
            @ (y / torch.norm(y, dim=1, keepdim=True)).t()
            / self.nce_t
        )
        label = torch.arange(bsz, device=x.device)
        loss = self.loss(scores, label) + self.loss(scores.t(), label)
        return loss


class ParaModel(pl.LightningModule):
    def __init__(self, args):
        super(ParaModel, self).__init__()

        self.args = args

        # Vocab
        self.scramble_rate = args.scramble_rate
        self.zero_unk = args.zero_unk

        # Model
        self.dropout = args.dropout
        if args.loss == "margin":
            self.loss = CosineSimMarginLoss(args.delta)
        elif args.loss == "nce":
            self.loss = NCESoftmaxLoss(args.nce_t)
        args.vocab_size = len(torch.load(args.vocab_path))
        self.encoder = Encoder.build(args)

        # Megabatch & hard negative mining
        self.increment = False
        self.curr_megabatch_size = 1
        self.max_megabatch_size = args.megabatch_size
        self.megabatch_anneal = args.megabatch_anneal
        self.megabatch = deque()

    def score(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x, _ = self.encoder(x_idxs, x_lengths)
        y, _ = self.encoder(y_idxs, y_lengths)
        return F.cosine_similarity(x, y)

    def forward(self, batch):
        # (x, y) are positive pairs
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x, _ = self.encoder(x_idxs, x_lengths)
        y, _ = self.encoder(y_idxs, y_lengths)

        # find hard negative paris
        x_neg_idxs, x_neg_lengths = self.mine_hard(y_idxs, y_lengths)
        y_neg_idxs, y_neg_lengths = self.mine_hard(x_idxs, x_lengths)

        x_neg, _ = self.encoder(x_neg_idxs, x_neg_lengths)
        y_neg, _ = self.encoder(y_neg_idxs, y_neg_lengths)

        return self.loss(x, y, x_neg, y_neg)

    def mine_hard(self, idxs, lengths):
        bs = idxs.shape[0]
        shuf_idx = torch.randperm(bs)
        return idxs[shuf_idx], lengths[shuf_idx]

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch).mean()
        self.log("loss/train", loss)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.megabatch = []

    def _unlabeled_eval_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return dict(loss=loss.detach().cpu())

    def _labeled_eval_step(self, batch, batch_idx):
        *batch, labels = batch
        scores = self.score(batch)
        return dict(scores=scores.detach().cpu(), labels=labels.detach().cpu())

    def _shared_eval_step(self, batch, batch_idx):
        if len(batch) == 3:
            return self._labeled_eval_step(batch, batch_idx)
        elif len(batch) == 2:
            return self._unlabeled_eval_step(batch, batch_idx)

    def _unlabeled_epoch_end(self, outputs, prefix):
        loss = torch.cat([o["loss"] for o in outputs]).mean()
        self.log(f"loss/{prefix}", loss)

    def _labeled_epoch_end(self, outputs, prefix):
        scores = torch.cat([o["scores"] for o in outputs]).tolist()
        labels = torch.cat([o["labels"] for o in outputs]).tolist()
        self.log(f"pearsonr/{prefix}", pearsonr(scores, labels)[0])
        self.log(f"spearmanr/{prefix}", spearmanr(scores, labels).correlation)

    def _shared_epoch_end(self, outputs, prefix):
        if "labels" in outputs[0]:
            self._labeled_epoch_end(outputs, prefix)
        else:
            self._unlabeled_epoch_end(outputs, prefix)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(
            outputs,
            f"val_{os.path.basename(self.datamodule.val_dataloader().dataset.data_file)}",
        )

    def test_epoch_end(self, outputs):
        if isinstance(outputs[0], list):
            for idx, subset_outputs in enumerate(outputs):
                self._shared_epoch_end(
                    subset_outputs,
                    f"test_{os.path.basename(self.datamodule.test_dataloader()[idx].dataset.data_file)}",
                )
        else:
            self._shared_epoch_end(
                outputs,
                f"test_{os.path.basename(self.datamodule.test_dataloader().dataset.data_file)}",
            )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.lr)


class Encoder(nn.Module):
    @staticmethod
    def build(args):
        return {"avg": Averaging, "lstm": LSTM,}[
            args.model
        ](args)

    def forward(self, idxs, lengths):
        raise NotImplementedError


class Averaging(Encoder):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = args.dropout

    def forward(self, idxs, lengths):
        word_embs = self.embedding(idxs)
        word_embs = F.dropout(word_embs, p=self.dropout, training=self.training)

        bs, max_len, _ = word_embs.shape
        mask = (
            torch.arange(max_len).cuda().expand(bs, max_len) < lengths.unsqueeze(1)
        ).float()
        word_embs = (word_embs * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = word_embs / lengths.unsqueeze(dim=1)

        return pooled, (word_embs,)


class LSTM(Encoder):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout

        self.register_buffer("e_hidden_init", torch.zeros(2, 1, args.hidden_dim))
        self.register_buffer("e_cell_init", torch.zeros(2, 1, args.hidden_dim))

        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.lstm = nn.LSTM(
            args.dim,
            args.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs, lengths):
        bsz, max_len = inputs.size()
        e_hidden_init = self.e_hidden_init.expand(2, bsz, self.hidden_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.hidden_dim).contiguous()
        lens, indices = torch.sort(lengths, 0, True)

        in_embs = self.embedding(inputs)
        in_embs = F.dropout(in_embs, p=self.dropout, training=self.training)

        all_hids, (enc_last_hid, _) = self.lstm(
            pack(in_embs[indices], lens.tolist(), batch_first=True),
            (e_hidden_init, e_cell_init),
        )

        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        bs, max_len, _ = all_hids.shape
        mask = (
            torch.arange(max_len).cuda().expand(bs, max_len) < lengths.unsqueeze(1)
        ).float()
        pooled = (all_hids * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(dim=1)

        return pooled, (all_hids,)
