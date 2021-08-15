import argparse
import json
from logging import StringTemplateStyle
import random
from collections import Counter
from re import template

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer, pipeline

from utils import canonicalize
from compute_correlations import test_correlation


def test_template(fill_mask, template, args):
    fill_mask_results = {}
    template_filled = []
    for csv in [args.small, args.medium, args.large]:
        pairs = pd.read_csv(csv, dtype=object)
        all_rets = []
        for var1, var2 in zip(pairs["id1"].tolist(), pairs["id2"].tolist()):
            var1 = "_".join(canonicalize(var1).split())
            var2 = "_".join(canonicalize(var2).split())
            template_filled.append(template.format(var1, var2))
    template_filled = list(set(template_filled))
    for tfilled, ret in zip(template_filled, fill_mask(template_filled)):
        fill_mask_results[tfilled] = ret

    for csv in [args.small, args.medium, args.large]:
        all_rets = []
        pairs = pd.read_csv(csv, dtype=object)
        for var1, var2 in zip(pairs["id1"].tolist(), pairs["id2"].tolist()):
            var1 = "_".join(canonicalize(var1).split())
            var2 = "_".join(canonicalize(var2).split())
            all_rets.append(fill_mask_results[template.format(var1, var2)])

        token_scores = Counter()
        ys = []
        for rets, similarity in zip(all_rets, pairs["similarity"].tolist()):
            if similarity == "NAN":
                continue
            for ret in rets:
                token_scores[ret["token_str"]] += ret["score"]
            ys.append(similarity)
        y = np.array(ys)

        # V = 3 if csv == args.small else 10
        # token2id = {x[0]: idx for idx, x in enumerate(token_scores.most_common(V))}
        V = 3
        token2id = {" this": 0, "this": 1, " that": 2}
        print(token2id)
        Xs = []
        for var1, var2, rets, similarity in zip(
            pairs["id1"].tolist(),
            pairs["id2"].tolist(),
            all_rets,
            pairs["similarity"].tolist(),
        ):
            if similarity == "NAN":
                continue
            x = np.zeros(V)
            for ret in rets:
                if ret["token_str"] in token2id:
                    x[token2id[ret["token_str"]]] = ret["score"]
            Xs.append(x)
        X = np.stack(Xs)

        # leave-one-out
        preds = []
        N = y.shape[0]
        for i in range(N):
            select = np.ones(N, dtype=bool)
            select[i] = 0
            reg = LinearRegression().fit(X[select], y[select])
            preds.append(reg.predict(X[i : i + 1]).item())
        ret = []
        idx = 0
        for similarity in pairs["similarity"].tolist():
            if similarity == "NAN":
                ret.append("NAN")
            else:
                ret.append(preds[idx])
                idx += 1

        pairs[args.name] = ret
        pairs.to_csv(csv, index=False)

    ret = test_correlation(args)
    return ret[args.name]


def read_templates(fname):
    with open(fname, encoding="utf-8-sig") as f:
        templates = []
        for line in f:
            tokens = json.loads(line)["UpdatedCodeChunkTokens"]
            if (
                len(tokens) < 40
                and len(
                    [
                        token
                        for token in tokens
                        if token.startswith("@@") and token.endswith("@@")
                    ]
                )
                == 2
            ):
                new_tokens = []
                first = False
                for token in tokens:
                    if token.startswith("@@") and token.endswith("@@"):
                        new_tokens.append("{0}" if not first else "{1}")
                        first = True
                    else:
                        new_tokens.append(token.replace("{", "{{").replace("}", "}}"))
                templates.append(new_tokens)
    return templates


def sample_template(templates):
    while True:
        template = random.choice(templates).copy()
        s = template.index("{0}")
        t = template.index("{1}")
        if s + 1 <= t - 1:
            template[random.randint(s + 1, t - 1)] = "<mask>"
            yield " ".join(template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--small", default="results/small_pair_wise.csv", help="Pairwise scores in small dataset")
    parser.add_argument("--medium", default="results/medium_pair_wise.csv", help="Pairwise scores in medium dataset")
    parser.add_argument("--large", default="results/large_pair_wise.csv", help="Pairwise scores in large dataset")
    parser.add_argument("--combined", help="Add combined embedding", default=False, action="store_true")
    parser.add_argument("--sample-template-json", default="github_commits_all_unfilt_awesome.dataset.jsonl")
    parser.add_argument("--name", help="method name")
    # fmt: on
    args = parser.parse_args()

    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    model.cuda()
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    # TEMPLATE = "for ({0} <mask> {1}) return 0; "
    fill_mask = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=20, device=0
    )

    results = {}
    max_score = 0

    template = "this . WriteBinaryString ( this . {0} ) ; <mask> . WriteBinaryString ( this . {1} ) ;"
    # template = "for ({0} <mask> {1}) return 0; "
    print(test_template(fill_mask, template, args))

    # tqdm_iter = tqdm(sample_template(read_templates(args.sample_template_json)))
    # for template in tqdm_iter:
    #     try:
    #         # print(template)
    #         results[template] = test_template(fill_mask, template, args)
    #         s, m, l = results[template]
    #         if m > max_score:
    #             best_T = template
    #             max_score = m
    #             tqdm_iter.set_description(f"best_score: {s:.3f}, {m: .3f}, {l: .3f}; best_T: {best_T}")
    #     except KeyboardInterrupt:
    #         t = input("continue? (Y/n)")
    #         with open("prompt_results.json", "w") as f:
    #             f.write(json.dumps(results))
    #         if t == "n":
    #             break
