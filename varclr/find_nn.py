import torch

from infer import MockArgs
from utils import CodePreprocessor

if __name__ == "__main__":
    ret = torch.load("saved")
    vars, embs = ret["vars"], ret["embs"]
    var2idx = dict([(var, idx) for idx, var in enumerate(vars)])
    # while (line := input()) != "":
    processor = CodePreprocessor(MockArgs())
    for line in [
        "substr",
        "item",
        "count",
        "rows",
        "setInterval",
        "minText",
        "files",
        "miny",
    ]:
        line = "".join(
            [
                word.capitalize() if idx > 0 else word
                for idx, word in enumerate(processor(line.strip()).split())
            ]
        )
        if line not in var2idx:
            print("variable not found")
            continue
        result = torch.topk(embs @ embs[var2idx[line]], k=21)
        print([vars[idx] for idx in result.indices][1:])
