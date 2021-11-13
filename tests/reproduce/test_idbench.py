import numpy as np

from varclr.benchmarks import Benchmark
from varclr.models import Encoder


def test_idbench():
    model = Encoder.from_pretrained("varclr-codebert")

    paper_results = {
        ("small", "similarity"): 0.53,
        ("medium", "similarity"): 0.53,
        ("large", "similarity"): 0.51,
        ("small", "relatedness"): 0.79,
        ("medium", "relatedness"): 0.79,
        ("large", "relatedness"): 0.80,
    }
    for (variant, metric), expected in paper_results.items():
        b = Benchmark.build("idbench", variant=variant, metric=metric)
        actual = b.evaluate(model.score(*b.get_inputs()))["spearmanr"]
        assert np.allclose(actual, expected, atol=1e-2)
