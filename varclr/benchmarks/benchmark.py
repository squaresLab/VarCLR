import os
import pathlib
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import pearsonr, spearmanr


class Benchmark:
    @staticmethod
    def build(benchmark: str, **kwargs):
        return {"idbench": IdBench}[benchmark](**kwargs)

    def get_inputs(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def evaluate(self, predictions) -> Dict[str, float]:
        raise NotImplementedError


class IdBench(Benchmark):

    BASELINES = ["FT-cbow", "FT-SG", "w2v-SG", "w2v-cbow", "Path-based"]

    def __init__(self, variant: str, metric: str) -> None:
        super().__init__()
        assert variant in {"small", "medium", "large"}
        assert metric in {"similarity", "relatedness"}
        self.variant = variant
        self.metric = metric

        pairs = pd.read_csv(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                f"idbench/{self.variant}_pair_wise.csv",
            ),
        )

        df = pairs[
            pairs.apply(
                lambda r: r[self.metric] != "NAN"
                and all(r[b] != "NAN" for b in IdBench.BASELINES),
                axis=1,
            )
        ]
        self.varlist1 = df["id1"].tolist()
        self.varlist2 = df["id2"].tolist()
        self.scores = df[self.metric].astype(float).tolist()

    def get_inputs(self) -> Tuple[List[str], List[str]]:
        return self.varlist1, self.varlist2

    def get_labels(self) -> List[float]:
        return self.scores

    def evaluate(self, predictions) -> Dict[str, float]:
        return {
            "spearmanr": spearmanr(predictions, self.scores).correlation,
            "pearsonr": pearsonr(predictions, self.scores)[0],
        }
