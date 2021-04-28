import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core import datamodule
from pytorch_lightning.loggers import WandbLogger
from torch.nn.modules import activation

from dataset import ParaDataModule
from models import ParaModel


def add_options(parser):
    # fmt: off
    # Dataset
    parser.add_argument("--train-data-file", required=True, help="training data")
    parser.add_argument("--test-data-files", required=True, help="test data")
    parser.add_argument("--zero-unk", default=1, type=int, help="whether to ignore unknown tokens")
    parser.add_argument("--ngrams", default=3, type=int, help="whether to use character n-grams")
    parser.add_argument("--tokenization", default="sp", type=str, choices=["sp", "ngram"], help="which tokenization to use")
    parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling")
    parser.add_argument("--sp-model", help="SP model to load for evaluation")
    parser.add_argument("--vocab-path", required=True, type=str, help="Path to vocabulary")
    parser.add_argument("--num-workers", default=4, type=int, help="Path to vocabulary")

    # Model
    parser.add_argument("--load-emb", default="", type=str, help="load word embedding")
    parser.add_argument("--dim", default=300, type=int, help="dimension of input embeddings")
    parser.add_argument("--model", default="avg", choices=["avg", "lstm"], help="type of base model to train.")
    parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
    parser.add_argument("--delta", default=0.4, type=float, help="margin")
    parser.add_argument("--share-encoder", default=1, type=int, help="whether to share the encoder (LSTM only)")

    # Training
    parser.add_argument("--name", default="Ours-FT", help="method name")
    parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
    parser.add_argument("--grad-clip", default=5., type=float, help='clip threshold of gradients')
    parser.add_argument("--epochs", default=30, type=int, help="number of epochs to train")
    parser.add_argument("--patience", default=5, type=int, help="early stopping patience")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0, type=float, help="dropout rate")
    parser.add_argument("--megabatch-size", default=60, type=int, help="number of batches in megabatch")
    parser.add_argument("--megabatch-anneal", default=150., type=int, help="rate of megabatch annealing in terms of "
                                                                        "number of batches to process before incrementing")
    parser.add_argument("--batch-size", default=128, type=int, help="size of batches")
    parser.add_argument("--load-file", help="filename to load a pretrained model.")
    parser.add_argument("--save-every-epoch", default=1, type=int, help="whether to save a checkpoint every epoch")
    parser.add_argument("--test", action="store_true", help="only do evaluation")

    # Evaluation
    parser.add_argument('--small', default="results/small_pair_wise.csv", help="Pairwise scores in small dataset")
    parser.add_argument('--medium', default="results/medium_pair_wise.csv", help="Pairwise scores in medium dataset")
    parser.add_argument('--large', default="results/large_pair_wise.csv", help="Pairwise scores in large dataset")
    parser.add_argument('--combined', help="Add combined embedding", default=False, action='store_true')
    # fmt: on


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    dm = ParaDataModule(args.train_data_file, args.test_data_files, args)
    if not os.path.exists(args.vocab_path):
        dm.setup()

    model = ParaModel(args)

    wandb_logger = WandbLogger(name=args.name, project="idbench", log_model=True)
    wandb_logger.log_hyperparams(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        gpus=args.gpu,
        auto_select_gpus=True,
        gradient_clip_val=1,
        # callbacks=[EarlyStopping(monitor="loss/val", mode="min", patience=args.patience)],
        callbacks=[EarlyStopping(monitor="spearmanr/val", mode="max", patience=args.patience)],
        progress_bar_refresh_rate=10,
        resume_from_checkpoint=args.load_file,
    )

    if not args.test:
        trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
