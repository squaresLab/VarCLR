import argparse
import sys
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from models import ParaModel
from utils import CodePreprocessor, lookup, unk_string
from main import add_options


def forward(model, input_ids, attention_mask):
    output = model(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    all_hids = output.hidden_states
    pooled = all_hids[-4][:, 0]
    return pooled


class MockArgs:
    def __init__(self):
        self.tokenization = ""


def batcher(batch_size):
    uniq = set()
    with open("var.txt") as f:
        vars = []
        for var in f:
            var_id = processor(var.strip())
            var = processor2(var.strip())
            if var_id not in uniq:
                uniq.add(var_id)
                vars.append((var_id, var))
            if len(vars) == batch_size:
                yield zip(*vars)
                vars = []
    yield zip(*vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParaModel(args)
    model = model.load_from_checkpoint(args.load_file, args=args, strict=False)
    model = model.to(device)
    model.eval()
    vocab = torch.load(args.vocab_path)

    processor = CodePreprocessor(args)
    processor2 = CodePreprocessor(MockArgs())
    ret_dict = dict(vars=[], embs=[])

    def torchify(batch):
        idxs = pad_sequence(
            [torch.tensor(ex, dtype=torch.long) for ex in batch],
            batch_first=True,
        )
        lengths = torch.tensor([len(e) for e in batch], dtype=torch.long)
        return idxs, lengths

    for var_ids, vars in tqdm(batcher(64)):
        batch = torchify(
            [
                [
                    lookup(vocab, w, args.zero_unk)
                    for w in var.split()
                    if lookup(vocab, w, args.zero_unk) is not None
                ]
                or [vocab[unk_string]]
                for var in var_ids
            ]
        )
        x_idxs, x_lengths = batch
        ret = model.encoder(x_idxs.to(device), x_lengths.to(device))
        embs, _ = ret
        embs = embs.detach().cpu()
        ret_dict["vars"].extend(
            [
                "".join(
                    [
                        word.capitalize() if idx > 0 else word
                        for idx, word in enumerate(var.split())
                    ]
                )
                for var in vars
            ]
        )
        ret_dict["embs"].extend(embs)
    ret_dict["embs"] = torch.stack(ret_dict["embs"])
    print(len(ret_dict["vars"]))
    print(ret_dict["embs"].shape)
    torch.save(ret_dict, "saved_lstm")
