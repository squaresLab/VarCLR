import sys
from collections import defaultdict

import torch

from varclr.utils.infer import MockArgs
from varclr.data.preprocessor import CodePreprocessor

if __name__ == "__main__":
    ret = torch.load(sys.argv[2])
    vars, embs = ret["vars"], ret["embs"]
    embs /= embs.norm(dim=1, keepdim=True)
    embs = embs.cuda()
    var2idx = dict([(var, idx) for idx, var in enumerate(vars)])
    processor = CodePreprocessor(MockArgs())
    Ks = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    topk_succ = defaultdict(int)
    tot = 0
    with open(sys.argv[1], "r") as f:
        for line in f:
            try:
                var1, var2 = line.strip().split()
            except ValueError:
                print("skpped: ", line)

            def canon(var):
                return "".join(
                    [
                        word.capitalize() if idx > 0 else word
                        for idx, word in enumerate(processor(var).split())
                    ]
                )

            var1, var2 = canon(var1), canon(var2)
            if var1 not in var2idx or var2 not in var2idx:
                print(f"variable {var1} or {var2} not found")
                continue
            tot += 1
            for k in Ks:
                result = torch.topk(embs @ embs[var2idx[var1]], k=k + 1)
                topk_succ[k] += var2 in [vars[idx] for idx in result.indices][1:]

    print(f"Total {tot} variable pairs")
    for k in Ks:
        print(f"Recall@{k} = {100 * topk_succ[k] / tot:.1f}")
