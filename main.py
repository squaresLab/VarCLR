import utils
import random
import numpy as np
import sys
import argparse
import torch
from models import Averaging, LSTM, load_model
from utils import Example, get_data

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("--data-file", required=True, help="training data")
parser.add_argument("--load-emb", default="", type=str, help="load word embedding")
parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
parser.add_argument("--dim", default=300, type=int, help="dimension of input embeddings")
parser.add_argument("--model", default="avg", choices=["avg", "lstm"], help="type of base model to train.")
parser.add_argument("--grad-clip", default=5., type=float, help='clip threshold of gradients')
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--dropout", default=0., type=float, help="dropout rate")
parser.add_argument("--batchsize", default=128, type=int, help="size of batches")
parser.add_argument("--megabatch-size", default=60, type=int, help="number of batches in megabatch")
parser.add_argument("--megabatch-anneal", default=150., type=int, help="rate of megabatch annealing in terms of "
                                                                       "number of batches to process before incrementing")
parser.add_argument("--pool", default="mean", choices=["mean", "max"], help="type of pooling")
parser.add_argument("--zero-unk", default=1, type=int, help="whether to ignore unknown tokens")
parser.add_argument("--load-file", help="filename to load a pretrained model.")
parser.add_argument("--save-every-epoch", default=0, type=int, help="whether to save a checkpoint every epoch")
parser.add_argument("--outfile", default="model", help="output file name")
parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
parser.add_argument("--delta", default=0.4, type=float, help="margin")
parser.add_argument("--ngrams", default=0, type=int, help="whether to use character n-grams")
parser.add_argument("--share-encoder", default=1, type=int, help="whether to share the encoder (LSTM only)")
parser.add_argument("--share-vocab", default=1, type=int, help="whether to share the embeddings")
parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling")
parser.add_argument("--sp-model", help="SP model to load for evaluation")
parser.add_argument("--temperature", default=10, type=float)
parser.add_argument('--small', default="results/small_pair_wise.csv", help="Pairwise scores in small dataset")
parser.add_argument('--medium', default="results/medium_pair_wise.csv", help="Pairwise scores in medium dataset")
parser.add_argument('--large', default="results/large_pair_wise.csv", help="Pairwise scores in large dataset")
parser.add_argument('--combined', help="Add combined embedding", default=False, action='store_true')
parser.add_argument("--name", default="Ours-FT", help="method name")

args = parser.parse_args()

data = get_data(args)

if args.load_file is not None:
    model, epoch = load_model(data, args)
    print("Loaded model at epoch {0} and resuming training.".format(epoch))
    model.train_epochs(start_epoch=epoch)
else:
    if args.ngrams:
        vocab, vocab_fr = utils.get_ngrams(data, args.share_vocab, n=args.ngrams)
    else:
        vocab, vocab_fr = utils.get_words(data, args.share_vocab)

    if args.model == "avg":
        model = Averaging(data, args, vocab, vocab_fr)
    elif args.model == "lstm":
        model = LSTM(data, args, vocab, vocab_fr)

    print(" ".join(sys.argv))
    print("Num examples:", len(data))
    print("Num words:", len(vocab))
    if vocab_fr is not None:
        print("Num fr words:", len(vocab_fr))

    model.train_epochs()
