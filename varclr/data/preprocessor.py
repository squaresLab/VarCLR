import re
from typing import List, Tuple, Union

from sacremoses import MosesTokenizer

from varclr.models.tokenizers import Tokenizer


class Preprocessor:
    @staticmethod
    def build(data_file, args) -> Tuple["Preprocessor", "Preprocessor"]:
        if "STS" in data_file:
            print(f"Using STS processor for {data_file}")
            return STSTextPreprocessor.from_args(args), STSTextPreprocessor.from_args(
                args
            )
        elif "idbench" in data_file:
            print(f"Using code processor for {data_file}")
            return CodePreprocessor.from_args(args), CodePreprocessor.from_args(args)
        elif "20k" in data_file:
            return Preprocessor(), Preprocessor()
        elif "nli" in data_file or "cs-cs" in data_file:
            print(f"Using NLI processor for {data_file}")
            return NLITextPreprocessor.from_args(args), NLITextPreprocessor.from_args(
                args
            )
        else:
            raise NotImplementedError

    def __call__(self, sentence):
        return sentence


class NLITextPreprocessor(Preprocessor):
    def __init__(self, tokenization, sp_model) -> None:
        self.tokenization = tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(sp_model)

    @staticmethod
    def from_args(args) -> "NLITextPreprocessor":
        return NLITextPreprocessor(args.tokenization, args.sp_model)

    def __call__(self, sentence):
        sent = sentence.lower()
        if self.tokenization == "sp":
            sent = " ".join(self.tokenizer.encode(sent))
        return sent


class STSTextPreprocessor(Preprocessor):
    def __init__(self, lang, tokenization, sp_model) -> None:
        self.moses = MosesTokenizer(lang=lang)
        self.tokenization = tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(sp_model)

    @staticmethod
    def from_args(args) -> "STSTextPreprocessor":
        return STSTextPreprocessor(args.tokenization, args.sp_model)

    def __call__(self, sentence):
        sent = " ".join(self.moses.tokenize(sentence))
        sent = sent.lower()
        if self.tokenization == "sp":
            sent = " ".join(self.tokenizer.encode(sent))
        return sent


class CodePreprocessor(Preprocessor):
    def __init__(self, tokenization=None, sp_model=None):
        self.tokenization = tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(sp_model)

    @staticmethod
    def from_args(args) -> "CodePreprocessor":
        return CodePreprocessor(args.tokenization, args.sp_model)

    def __call__(self, var: Union[str, List[str]]):
        if isinstance(var, str):
            return self._process(var)
        elif isinstance(var, list) and all(isinstance(v, str) for v in var):
            return [self._process(v) for v in var]
        else:
            raise NotImplementedError

    def _process(self, var):
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
