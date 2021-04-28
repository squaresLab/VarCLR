import random
import re
from typing import Tuple

import sentencepiece as spm
from sacremoses import MosesTokenizer

unk_string = "UUUNKKK"


def lookup(words, w, zero_unk):
    w = w.lower()
    if w in words:
        return words[w]
    else:
        if zero_unk:
            return None
        else:
            return words[unk_string]


class Example(object):
    def __init__(self, sentence):
        self.sentence = sentence.strip().lower()
        self.embeddings = []

    def populate_embeddings(
        self, words, zero_unk, tokenization, ngrams, scramble_rate=0
    ):
        self.embeddings = []
        if tokenization == "ngrams":
            sentence = " " + self.sentence.strip() + " "
            for j in range(len(sentence)):
                idx = j
                gr = ""
                while idx < j + ngrams and idx < len(sentence):
                    gr += sentence[idx]
                    idx += 1
                if not len(gr) == ngrams:
                    continue
                wd = lookup(words, gr, zero_unk)
                if wd is not None:
                    self.embeddings.append(wd)
        elif tokenization == "sp":
            arr = self.sentence.split()
            if scramble_rate:
                if random.random() <= scramble_rate:
                    random.shuffle(arr)
            for i in arr:
                wd = lookup(words, i, zero_unk)
                if wd is not None:
                    self.embeddings.append(wd)
        else:
            raise NotImplementedError
        if len(self.embeddings) == 0:
            self.embeddings = [words[unk_string]]


class TextPreprocessor:
    @staticmethod
    def build(data_file, args) -> Tuple["TextPreprocessor", "TextPreprocessor"]:
        if "STS" in data_file:
            if "en-en" in data_file:
                return STSTextPreprocessor("en", args), STSTextPreprocessor("en", args)
            else:
                raise NotImplementedError
        else:
            return TextPreprocessor(args), TextPreprocessor(args)

    def __init__(self, args) -> None:
        pass

    def __call__(self, sentence):
        return sentence


class STSTextPreprocessor(TextPreprocessor):
    def __init__(self, lang, args) -> None:
        super().__init__(args)
        self.moses = MosesTokenizer(lang=lang)
        self.tokenization = args.tokenization
        if self.tokenization == "sp":
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(args.sp_model)

    def __call__(self, sentence):
        sent = " ".join(self.moses.tokenize(sentence))
        sent = sent.lower()
        if self.tokenization == "sp":
            sent = " ".join(self.sp.EncodeAsPieces(sent))
        return sent


def canonicalize(var):
    var = var.replace("@", "")
    var = re.sub("([a-z]|^)([A-Z]{1})", r"\1_\2", var).lower().replace("_", " ").strip()
    return var
