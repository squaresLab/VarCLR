import sys

import torch
from tqdm import tqdm

from utils import CodePreprocessor


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
    with open(sys.argv[1]) as f:
        vars = []
        for uncanon_var in f:
            uncanon_var = uncanon_var.strip()
            var = processor(uncanon_var)
            if var not in uniq:
                uniq.add(var)
                vars.append((var, uncanon_var))
            if len(vars) == batch_size:
                yield list(zip(*vars))
                vars = []
    yield list(zip(*vars))

def read_embs(fname):
    all_embs = {}
    with open(fname) as f:
        for line in f:
            if not '"ID:' in line: continue
            name, *emb = line.strip().split()
            name = name[1:1 + name[1:].index('"')]
            all_embs[name] = torch.tensor(list(map(float, emb)))
    return all_embs

if __name__ == "__main__":
    processor = CodePreprocessor(MockArgs())
    ret_dict = dict(vars=[], embs=[])
    all_embs = read_embs(sys.argv[2])
    for vars, uncanon_vars in tqdm(batcher(64)):
        embs = (all_embs[f"ID:{v}"] for v in uncanon_vars)
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
    torch.save(ret_dict, "saved_ft")
