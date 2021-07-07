import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr, spearmanr
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from transformers import AutoModel


class Scorer(nn.Module):
    def forward(self, x, y):
        raise NotImplementedError


class CosineScorer(Scorer):
    def forward(self, x_ret, y_ret):
        x_pooled, _ = x_ret
        y_pooled, _ = y_ret
        return F.cosine_similarity(x_pooled, y_pooled)


class BiAttnScorer(Scorer):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, x_ret, y_ret):
        _, (v1, mask1) = x_ret
        _, (v2, mask2) = y_ret
        # match matrix similar to attention score: match_ij = cosine(v1_i, v2_j)
        # Shape: B x L1 x L2
        match = v1 @ v2.transpose(1, 2)
        # B x L1 x 1 @ B x 1 x L2 => B x L1 x L2
        match_mask = mask1.unsqueeze(dim=2) @ mask2.unsqueeze(dim=1)
        match[~match_mask.bool()] = -10000
        s1 = -torch.max(match, dim=2)[0] / self.temperature
        s1[~mask1.bool()] = -10000
        attn1 = F.softmax(s1, dim=1)
        v1 = (v1 * attn1.unsqueeze(dim=2)).sum(dim=1)
        s2 = -torch.max(match, dim=1)[0] / self.temperature
        s2[~mask2.bool()] = -10000
        attn2 = F.softmax(s2, dim=1)
        v2 = (v2 * attn2.unsqueeze(dim=2)).sum(dim=1)
        return F.cosine_similarity(v1, v2)


class DecAttScorer(Scorer):
    def __init__(self, dim, hid) -> None:
        super().__init__()
        self.attn_forward = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid, dim),
            nn.LayerNorm(dim),
        )
        self.compare_forward = nn.Sequential(
            nn.Linear(dim * 2, hid),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid, dim),
            nn.LayerNorm(dim),
        )
        self.aggregate_forward = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hid, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_ret, y_ret):
        _, (v1, mask1) = x_ret
        _, (v2, mask2) = y_ret
        # match matrix similar to attention score: match_ij = cosine(v1_i, v2_j)
        # Shape: B x L1 x L2
        av1 = self.attn_forward(v1)
        av2 = self.attn_forward(v2)
        match = av1 @ av2.transpose(1, 2)
        match = v1 @ v2.transpose(1, 2)
        # B x L1 x 1 @ B x 1 x L2 => B x L1 x L2
        match_mask = mask1.unsqueeze(dim=2) @ mask2.unsqueeze(dim=1)
        match[~match_mask.bool()] = -10000

        # DecAttn
        beta = F.softmax(match, dim=2) @ v2
        alpha = F.softmax(match, dim=1).transpose(1, 2) @ v1
        v1i = self.compare_forward(torch.cat([v1, beta], dim=2))
        v2j = self.compare_forward(torch.cat([v2, alpha], dim=2))
        v1i = v1i.sum(dim=1)
        v2j = v2j.sum(dim=1)
        return F.cosine_similarity(v1i, v2j)
        # # ret = self.aggregate_forward(torch.cat([v1i, v2j], dim=1))
        # # return ret.squeeze(dim=1)


class SimMarginLoss(nn.Module):
    def __init__(self, delta, scorer):
        super().__init__()
        self.delta = delta
        self.scorer = scorer
        self.loss = nn.MarginRankingLoss(margin=self.delta, reduction="none")

    def forward(self, x_ret, y_ret, x_neg_ret, y_neg_ret):
        pos = self.scorer(x_ret, y_ret)
        neg1 = self.scorer(x_ret, x_neg_ret)
        neg2 = self.scorer(y_ret, y_neg_ret)
        ones = torch.ones(pos.shape[0], device=x_ret[0].device)
        loss = self.loss(pos, neg1, ones) + self.loss(pos, neg2, ones)
        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self, nce_t):
        super(NCESoftmaxLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.nce_t = nce_t

    def forward(self, x_ret, y_ret, x_neg_ret, y_neg_ret):
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


class ParaModel(pl.LightningModule):
    def __init__(self, args):
        super(ParaModel, self).__init__()

        self.args = args

        # Vocab
        self.scramble_rate = args.scramble_rate
        self.zero_unk = args.zero_unk

        # Model
        self.dropout = args.dropout
        if args.scorer == "cosine":
            self.scorer = CosineScorer()
        elif args.scorer == "biattn":
            self.scorer = BiAttnScorer(args.temperature)
        elif args.scorer == "decatt":
            self.scorer = DecAttScorer(args.dim, args.hidden_dim)
        if args.loss == "margin":
            self.loss = SimMarginLoss(args.delta, self.scorer)
        elif args.loss == "nce":
            self.loss = NCESoftmaxLoss(args.nce_t)
        args.vocab_size = len(torch.load(args.vocab_path))
        args.parentmodel = self
        self.encoder = Encoder.build(args)

    def score(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x_ret = self.encoder(x_idxs, x_lengths)
        y_ret = self.encoder(y_idxs, y_lengths)
        return self.scorer(x_ret, y_ret)

    def forward(self, batch):
        # (x, y) are positive pairs
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x_ret = self.encoder(x_idxs, x_lengths)
        y_ret = self.encoder(y_idxs, y_lengths)

        # find hard negative paris
        x_neg_idxs, x_neg_lengths = self.mine_hard(y_idxs, y_lengths)
        y_neg_idxs, y_neg_lengths = self.mine_hard(x_idxs, x_lengths)

        x_neg_ret = self.encoder(x_neg_idxs, x_neg_lengths)
        y_neg_ret = self.encoder(y_neg_idxs, y_neg_lengths)

        return self.loss(x_ret, y_ret, x_neg_ret, y_neg_ret)

    def mine_hard(self, idxs, lengths):
        bs = idxs.shape[0]
        shuf_idx = torch.randperm(bs)
        return idxs[shuf_idx], lengths[shuf_idx]

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch).mean()
        self.log("loss/train", loss)
        return loss

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
        return {'bert': optim.AdamW}.get(self.args.model, optim.Adam)(self.parameters(), lr=self.args.lr)


class Encoder(nn.Module):
    @staticmethod
    def build(args):
        return {"avg": Averaging, "lstm": LSTM, "attn": Attn, "bert": BERT}[args.model](args)

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
        pooled = (word_embs * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(dim=1)

        return pooled, (word_embs, mask)


class Attn(Encoder):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = args.dropout
        self.attn = nn.Parameter(torch.zeros(args.dim))
        self.args = args
        nn.init.normal_(self.attn)

    def forward(self, idxs, lengths):
        word_embs = self.embedding(idxs)
        word_embs = F.dropout(word_embs, p=self.dropout, training=self.training)

        bs, max_len, _ = word_embs.shape
        mask = torch.arange(max_len).cuda().expand(bs, max_len) < lengths.unsqueeze(1)
        scores = (word_embs * self.attn).sum(dim=-1) / 100
        scores[~mask] = -10000
        a = F.softmax(scores, dim=-1)
        pooled = (a.unsqueeze(dim=1) @ word_embs).squeeze(dim=1)

        return pooled, (word_embs, mask)


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

        return pooled, (all_hids, mask)

class BERT(Encoder):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.transformer = AutoModel.from_pretrained(args.bert_model)

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_hids = output.last_hidden_state
        # pooled = all_hids[:, 0]
        pooled = (all_hids * attention_mask.unsqueeze(dim=2)).sum(dim=1)

        return pooled, (all_hids, attention_mask)
