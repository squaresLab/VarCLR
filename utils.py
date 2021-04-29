import random
import re
from collections import Counter
from typing import Tuple

import sentencepiece as spm
from sacremoses import MosesTokenizer
from transformers import AutoTokenizer

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


class Tokenizer:

    @staticmethod
    def build(args):
        if "sp.20k.model" in args.sp_model:
            return SPTokenizer(args.sp_model)
        elif "bert" in args.sp_model:
            return PretrainedTokenizer(args.sp_model)
        else:
            raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError

class SPTokenizer(Tokenizer):

    def __init__(self, model_path) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        return self.sp.EncodeAsPieces(text)

class PretrainedTokenizer(Tokenizer):

    def __init__(self, tokenizer_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, text):
        return list(map(str, self.tokenizer.encode(text, add_special_tokens=False, truncation=True)))

class TextPreprocessor:
    @staticmethod
    def build(data_file, args) -> Tuple["TextPreprocessor", "TextPreprocessor"]:
        if "STS" in data_file:
            print(f"Using STS processor for {data_file}")
            return STSTextPreprocessor("en", args), STSTextPreprocessor("en", args)
        elif "idbench" in data_file:
            print(f"Using code processor for {data_file}")
            return CodePreprocessor(args), CodePreprocessor(args)
        elif "nli" in data_file:
            print(f"Using NLI processor for {data_file}")
            return NLITextPreprocessor(args), NLITextPreprocessor(args)
        else:
            return TextPreprocessor(), TextPreprocessor()

    def __call__(self, sentence):
        return sentence

class NLITextPreprocessor(TextPreprocessor):
    def __init__(self, args) -> None:
        self.tokenization = args.tokenization
        if self.tokenization == "sp":
            self.tokenizer = Tokenizer.build(args)

    def __call__(self, sentence):
        sent = sentence.lower()
        if self.tokenization == "sp":
            sent = " ".join(self.tokenizer.encode(sent))
        return sent


class STSTextPreprocessor(TextPreprocessor):
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


class CodePreprocessor(TextPreprocessor):
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


class Vocab:
    @staticmethod
    def build(examples, args):
        if args.tokenization == "ngrams":
            return Vocab.get_ngrams(examples, n=args.ngrams)
        elif args.tokenization == "sp":
            return Vocab.get_words(examples)
        else:
            raise NotImplementedError

    @staticmethod
    def get_ngrams(examples, max_len=200000, n=3):
        def update_counter(counter, sentence):
            word = " " + sentence.strip() + " "
            lis = []
            for j in range(len(word)):
                idx = j
                ngram = ""
                while idx < j + n and idx < len(word):
                    ngram += word[idx]
                    idx += 1
                if not len(ngram) == n:
                    continue
                lis.append(ngram)
            counter.update(lis)

        counter = Counter()

        for i in examples:
            update_counter(counter, i[0].sentence)
            update_counter(counter, i[1].sentence)

        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0:max_len]

        vocab = {}
        for i in counter:
            vocab[i[0]] = len(vocab)

        vocab[unk_string] = len(vocab)
        return vocab

    @staticmethod
    def get_words(examples, max_len=200000):
        def update_counter(counter, sentence):
            counter.update(sentence.split())

        counter = Counter()

        for i in examples:
            update_counter(counter, i[0].sentence)
            update_counter(counter, i[1].sentence)

        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0:max_len]

        vocab = {}
        for i in counter:
            vocab[i[0]] = len(vocab)

        vocab[unk_string] = len(vocab)
        return vocab
