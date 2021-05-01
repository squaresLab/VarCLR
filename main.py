import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel

from dataset import ParaDataModule
from models import ParaModel


def add_options(parser):
    # fmt: off
    # Dataset
    parser.add_argument("--train-data-file", default="moses_nli_for_simcse.csv", help="training data")
    parser.add_argument("--valid-data-file", default=None, type=str, help="validation data")
    parser.add_argument("--test-data-files", default="STS-B/original/sts-test.tsv,STS/STS17-test/STS.input.track5.en-en.txt,idbench/small_pair_wise.csv,idbench/medium_pair_wise.csv,idbench/large_pair_wise.csv", help="test data")
    parser.add_argument("--zero-unk", default=1, type=int, help="whether to ignore unknown tokens")
    parser.add_argument("--ngrams", default=3, type=int, help="whether to use character n-grams")
    parser.add_argument("--tokenization", default="sp", type=str, choices=["sp", "ngrams"], help="which tokenization to use")
    parser.add_argument("--sp-model", default="acl19-simple/en-es.os.1m.tok.sp.20k.model", help="SP model to load for evaluation")
    parser.add_argument("--vocab-path", default="acl19-simple/en-es.os.1m.tok.sp.20k.vocab", type=str, help="Path to vocabulary")
    parser.add_argument("--num-workers", default=4, type=int, help="Path to vocabulary")

    # Model
    parser.add_argument("--model", default="avg", choices=["avg", "lstm", "attn"], help="type of base model to train.")
    parser.add_argument("--dim", default=300, type=int, help="dimension of input embeddings")
    parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
    parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling in for LSTM")
    parser.add_argument("--loss", default="nce", type=str, choices=["margin", "nce"], help="loss")
    parser.add_argument("--delta", default=0.4, type=float, help="margin size for margin ranking loss")
    parser.add_argument("--nce-t", default=0.05, type=float, help="temperature for noise contrastive estimation loss")
    parser.add_argument("--scorer", default="cosine", choices=["cosine", "biattn", "decatt"], help="scorer to evaluate similarity")
    parser.add_argument("--temperature", default=100, type=float, help="temperature for biattn scorer")

    # Training
    parser.add_argument("--name", default="Ours-FT", help="method name")
    parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
    parser.add_argument("--grad-clip", default=1., type=float, help='clip threshold of gradients')
    parser.add_argument("--epochs", default=300, type=int, help="number of epochs to train")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0, type=float, help="dropout rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="size of batches")
    parser.add_argument("--load-file", help="filename to load a pretrained model.")
    parser.add_argument("--test", action="store_true", help="only do evaluation")
    parser.add_argument("--train-percent", default=1.0, type=float, help="percentage of data used for training")
    parser.add_argument("--seed", default=42, type=int)
    # fmt: on


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dm = ParaDataModule(
        args.train_data_file, args.valid_data_file, args.test_data_files, args
    )
    if not os.path.exists(args.vocab_path):
        dm.setup()

    model = ParaModel(args)
    if args.load_file is not None:
        model = model.load_from_checkpoint(args.load_file, args=args, strict=False)
    model.datamodule = dm

    if not args.test and "bert" in args.sp_model:
        # Load pre-trained word embeddings from bert
        bert = AutoModel.from_pretrained(args.sp_model)
        for word, idx in torch.load(args.vocab_path).items():
            try:
                model.encoder.embedding.weight.data[
                    idx
                ] = bert.embeddings.word_embeddings.weight.data[int(word)]
            except ValueError:
                pass
        del bert

    if args.valid_data_file is not None:
        callbacks = [
            EarlyStopping(
                monitor=f"spearmanr/val_{os.path.basename(dm.valid_data_file)}",
                mode="max",
                patience=args.patience,
            ),
            ModelCheckpoint(
                monitor=f"spearmanr/val_{os.path.basename(dm.valid_data_file)}",
                mode="max",
            ),
        ]
    else:
        callbacks = [
            EarlyStopping(
                monitor=f"loss/val_{os.path.basename(dm.train_data_file)}",
                patience=args.patience,
            ),
            ModelCheckpoint(monitor=f"loss/val_{os.path.basename(dm.train_data_file)}"),
        ]

    wandb_logger = WandbLogger(name=args.name, project="idbench", log_model=True)
    wandb_logger.log_hyperparams(args)
    args = argparse.Namespace(**wandb_logger.experiment.config)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        gpus=args.gpu,
        auto_select_gpus=True,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        progress_bar_refresh_rate=10,
    )

    if not args.test:
        trainer.fit(model, datamodule=dm)
        # will automatically load and test the best checkpoint instead of the last model
        trainer.test(datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
