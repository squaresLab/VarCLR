import argparse
import re

import sentencepiece as spm
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity

from models import load_model
from utils import get_data, unk_string, Example

def canonicalize(var):
    var = var.replace("@", "")
    var = re.sub("([a-z]|^)([A-Z]{1})", r"\1_\2", var).lower().replace("_", " ").strip()
    return var

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-file", help="filename to load a pretrained model.")
    parser.add_argument("--sp-model", help="SP model to load for evaluation")
    parser.add_argument("--csv", help="idbench test file to save results")
    parser.add_argument("--name", help="method name")
    args = parser.parse_args()
    model, epoch = load_model(None, args)
    # model.to("cpu")
    # model.gpu = False
    print("Loaded model at epoch {0} and resuming training.".format(epoch))
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model)

    for csv in args.csv.split(","):
        pairs = pd.read_csv(csv, dtype=object)
        sim = []
        for var1, var2 in zip(pairs["id1"].tolist(), pairs["id2"].tolist()):
            var1_pieces = " ".join(sp.encode_as_pieces(canonicalize(var1)))
            var2_pieces = " ".join(sp.encode_as_pieces(canonicalize(var2)))
            print(var1_pieces, var2_pieces)
            wp1 = Example(var1_pieces)
            wp2 = Example(var2_pieces)
            wp1.populate_embeddings(model.vocab, model.zero_unk, False)
            wp2.populate_embeddings(model.vocab, model.zero_unk, False)
            wx1, wl1 = model.torchify_batch([wp1])
            wx2, wl2 = model.torchify_batch([wp2])
            scores = model.scoring_function(wx1, wl1, wx2, wl2, fr0=False, fr1=False)
            print(scores.item())
            sim.append(f"{scores.item():.4f}")
        pairs[args.name] = sim
        pairs.to_csv(csv, index=False)
    
