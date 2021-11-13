import random
import re
from typing import Tuple

from sacremoses import MosesTokenizer
from varclr.models import Tokenizer


class Preprocessor:
    @staticmethod
    def build(data_file, args) -> Tuple["Preprocessor", "Preprocessor"]:
        if "STS" in data_file:
            print(f"Using STS processor for {data_file}")
            return STSTextPreprocessor("en", args), STSTextPreprocessor("en", args)
        elif "idbench" in data_file:
            print(f"Using code processor for {data_file}")
            return CodePreprocessor(args), CodePreprocessor(args)
        elif "20k" in data_file:
            return Preprocessor(), Preprocessor()
        elif "nli" in data_file or "cs-cs" in data_file:
            print(f"Using NLI processor for {data_file}")
            return NLITextPreprocessor(args), NLITextPreprocessor(args)
        else:
            raise NotImplementedError

    def __call__(self, sentence):
        return sentence


class NLITextPreprocessor(Preprocessor):
    def __init__(self, args) -> None:
        self.tokenization = args.tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(args)

    def __call__(self, sentence):
        sent = sentence.lower()
        if self.tokenization == "sp":
            sent = " ".join(self.tokenizer.encode(sent))
        return sent


class STSTextPreprocessor(Preprocessor):
    def __init__(self, lang, args) -> None:
        self.moses = MosesTokenizer(lang=lang)
        self.tokenization = args.tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(args)

    def __call__(self, sentence):
        sent = " ".join(self.moses.tokenize(sentence))
        sent = sent.lower()
        if self.tokenization == "sp":
            sent = " ".join(self.tokenizer.encode(sent))
        return sent


class CodePreprocessor(Preprocessor):
    def __init__(self, args) -> None:
        self.tokenization = args.tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(args)

    def __call__(self, var):
        var = var.replace("@", "")
        var = (
            re.sub("([a-z]|^)([A-Z]{1})", r"\1_\2", var)
            .lower()
            .replace("_", " ")
            .strip()
        )
        if self.tokenization == "sp":
            var = " ".join(self.tokenizer.encode(var))
        return var
