#!/usr/bin/python3

# Author: Michael Pradel
#
# Computes the correlation between
#  - similaries computed by semantic representations and
#  - the human-based ground truth in IdBench.
#

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
# import enchant
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


kinds = ["relatedness", "similarity", "contextual_similarity"]
sizes = ["small", "medium", "large"]
approaches = ["FT-cbow", "FT-SG", "w2v-SG",
              "w2v-cbow", "Path-based", "LV", "NW"]
new_approaches = []  # filled automatically based on given .csv files

# dict = enchant.Dict("en_US")


parser = argparse.ArgumentParser()
parser.add_argument(
    '--small', help="Pairwise scores in small dataset", required=True)
parser.add_argument(
    '--medium', help="Pairwise scores in medium dataset", required=True)
parser.add_argument(
    '--large', help="Pairwise scores in large dataset", required=True)
parser.add_argument(
    '--combined', help="Add combined embedding", default=False, action='store_true')


def subtokens(id):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', id)
    ts = [m.group(0) for m in matches]
    ts = [t.split("_") for t in ts]
    ts = [item for sublist in ts for item in sublist]
    ts = [t for t in ts if t != ""]
    return ts


def extract_id_features(w1, w2):
    features = []

    # length of ids
    features.append(len(w1))
    features.append(len(w2))

    # nb of subtokens
    subs1 = subtokens(w1)
    subs2 = subtokens(w2)
    features.append(len(subs1))
    features.append(len(subs2))

    # nb of (non-)dictionary words
    w1_dict = w2_dict = 0
    for t in subs1:
        if dict.check(t):
            w1_dict += 1
    for t in subs2:
        if dict.check(t):
            w2_dict += 1
    features.append(w1_dict)
    features.append(len(subs1) - w1_dict)
    features.append(w2_dict)
    features.append(len(subs2) - w2_dict)

    return features


def create_model():
    model = Pipeline([("scaler", StandardScaler()),
                      ("SVM", SVR())])

    return model


def add_combined_prediction(pairs, ground_truth_label: str):
    xs = []
    ys = []
    for _, (_, p) in enumerate(pairs.iterrows()):
        gt = getattr(p, ground_truth_label)
        ys.append(gt)
        preds = []
        for approach in approaches + new_approaches:
            preds.append(p[approach])
        preds.extend(extract_id_features(p.id1, p.id2))
        xs.append(preds)

    pred_ys = []
    for p_idx, (_, p) in enumerate(pairs.iterrows()):
        sel_xs = xs[: p_idx] + xs[p_idx+1:]
        sel_ys = ys[: p_idx] + ys[p_idx+1:]
        model = create_model()
        model.fit(sel_xs, sel_ys)
        pred_y = model.predict([xs[p_idx]])[0]
        pred_ys.append(pred_y)

    pairs[f"combined_{ground_truth_label}"] = pred_ys


def compute_correlations(pairs, kind, label_to_approach):
    correlations = []
    for _, approach in label_to_approach.items():
        c = spearmanr(pairs[kind], pairs[approach]).correlation
        correlations.append(c)
    return correlations


def plot_correlations(ys_large, ys_medium, ys_small, out_file, labels):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.bar(x - width, ys_large, width, label='Large benchm.')
    ax.bar(x, ys_medium, width, label='Medium benchm.')
    ax.bar(x + width, ys_small, width, label='Small benchm.')

    ax.set_ylim([0.0, 0.85])
    ax.set_yticks(np.arange(0, 0.9, step=0.2))
    ax.set_xlabel('Similarity functions')
    ax.set_ylabel('Correlation with benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)

    fig.tight_layout()

    plt.savefig(out_file, format="pdf")


def plot_correlations_all(size_to_kind_to_pairs, combined, plot=False):
    plt.rcParams.update({'font.size': 17})

    for kind in kinds:
        label_to_approach = {
            "LV": "LV",
            "NW": "NW",
            "FT-cbow": "FT-cbow",
            "FT-SG": "FT-SG",
            "w2v-cbow": "w2v-cbow",
            "w2v-SG": "w2v-SG",
            "Path-\nbased": "Path-based"
        }
        for a in new_approaches:
            label_to_approach[a] = a
        if combined:
            label_to_approach["Combined"] = f"combined_{kind}"

        small_ys = compute_correlations(
            size_to_kind_to_pairs["small"][kind], kind, label_to_approach)
        medium_ys = compute_correlations(
            size_to_kind_to_pairs["medium"][kind], kind, label_to_approach)
        large_ys = compute_correlations(
            size_to_kind_to_pairs["large"][kind], kind, label_to_approach)

        labels = label_to_approach.keys()
        large_ys, medium_ys, small_ys, labels = zip(*[
            (l, m, s, label)
            for (l, m, s, label) in reversed(sorted(zip(large_ys, medium_ys, small_ys, labels)))
        ])
        if kind == "similarity":
            # print(list(zip(labels, map(lambda x: f"{x:.3f}", small_ys))))
            # print(list(zip(labels, map(lambda x: f"{x:.3f}", medium_ys))))
            # print(list(zip(labels, map(lambda x: f"{x:.3f}", large_ys))))
            if plot:
                plot_correlations(large_ys, medium_ys, small_ys,
                                f"correlations_{kind}.pdf",
                                labels=labels)
            return {label: (s, m, l) for label, s, m, l in zip(labels, small_ys, medium_ys, large_ys)}


def read_and_clean_pairs(args):
    size_to_additional_embeddings = {}
    size_to_kind_to_pairs = {}
    for size in sizes:
        size_to_kind_to_pairs[size] = {}
        pairs = pd.read_csv(getattr(args, size), dtype=object)
        # print(pairs)

        # check for additional embeddings beyond what IdBench contains by default
        new_column_headers = list(pairs.columns[12:])
        size_to_additional_embeddings[size] = new_column_headers

        def row_filter(r):
            for col_name in pairs.columns:
                if r[col_name] == "NAN":
                    return False
            return True

        for kind in kinds:
            filtered_pairs = pairs[pairs.apply(row_filter, axis=1)]
            size_to_kind_to_pairs[size][kind] = filtered_pairs

    # ensure that if new embeddings added, they are added for all three sizes
    for size1 in sizes:
        for size2 in sizes:
            if size1 != size2:
                if size_to_additional_embeddings[size1] != size_to_additional_embeddings[size2]:
                    raise Exception(
                        f"New embedding columns must be added for all three sizes. Found {size_to_additional_embeddings[size1]} for {size1} but {size_to_additional_embeddings[size2]} for {size2}.")
    global new_approaches
    new_approaches = size_to_additional_embeddings[sizes[0]]

    return size_to_kind_to_pairs

def test_correlation(args, plot=False):
    size_to_kind_to_pairs = read_and_clean_pairs(args)

    for size in sizes:
        for kind in kinds:
            pairs = size_to_kind_to_pairs[size][kind]
            if args.combined:
                add_combined_prediction(pairs, kind)

    return plot_correlations_all(size_to_kind_to_pairs, combined=args.combined, plot=plot)

if __name__ == "__main__":
    args = parser.parse_args()
    test_correlation(args, plot=True)
