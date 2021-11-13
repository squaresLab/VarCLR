import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import optim
from varclr.models.encoders import Encoder
from varclr.models.loss import NCESoftmaxLoss


class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.dropout = args.dropout
        self.loss = NCESoftmaxLoss(args.nce_t)
        args.vocab_size = len(torch.load(args.vocab_path))
        args.parentmodel = self
        self.encoder = Encoder.build(args)

    def _forward(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x_ret = self.encoder(x_idxs, x_lengths)
        y_ret = self.encoder(y_idxs, y_lengths)

        return self.loss(x_ret, y_ret)

    def _score(self, batch):
        (x_idxs, x_lengths), (y_idxs, y_lengths) = batch
        x_pooled, _ = self.encoder(x_idxs, x_lengths)
        y_pooled, _ = self.encoder(y_idxs, y_lengths)
        return F.cosine_similarity(x_pooled, y_pooled)

    def training_step(self, batch, batch_idx):
        loss = self._forward(batch).mean()
        self.log("loss/train", loss)
        return loss

    def _unlabeled_eval_step(self, batch, batch_idx):
        loss = self._forward(batch)
        return dict(loss=loss.detach().cpu())

    def _labeled_eval_step(self, batch, batch_idx):
        *batch, labels = batch
        scores = self._score(batch)
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
        return {"bert": optim.AdamW}.get(self.args.model, optim.Adam)(
            self.parameters(), lr=self.args.lr
        )
