import io
import os
import random
from collections import Counter
from typing import List, Optional, Text, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from utils import TextPreprocessor, Example, unk_string


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

class ParaDataset(Dataset):
    def __init__(self, data_file: str, args, training=True) -> None:
        super().__init__()
        self.data_file = data_file
        self.tokenization = args.tokenization
        self.ngrams = args.ngrams
        self.scramble_rate = args.scramble_rate
        self.zero_unk = args.zero_unk
        self.training = training
        self.processor1, self.processor2 = TextPreprocessor.build(data_file, args)
        self.examples_pairs = self.read_examples()
        if not os.path.exists(args.vocab_path):
            print(f"Vocab not found. Creating from {data_file}")
            self.vocab = Vocab.build(self.examples_pairs, args)
            torch.save(self.vocab, args.vocab_path)
        else:
            self.vocab = torch.load(args.vocab_path)

    def __getitem__(self, i):
        self.examples_pairs[i][0].populate_embeddings(
            self.vocab,
            self.zero_unk,
            self.tokenization,
            self.ngrams,
            scramble_rate=self.scramble_rate if self.training else 0,
        )
        self.examples_pairs[i][1].populate_embeddings(
            self.vocab,
            self.zero_unk,
            self.tokenization,
            self.ngrams,
            scramble_rate=self.scramble_rate if self.training else 0,
        )
        return self.examples_pairs[i]

    def __len__(self):
        return len(self.examples_pairs)

    def read_examples(self):
        examples = []
        finished = set([])  # check for duplicates
        with io.open(self.data_file, "r", encoding="utf-8") as f:
            for i in f:
                if i in finished:
                    continue
                else:
                    finished.add(i)

                i = i.split("\t")
                if len(i[0].strip()) == 0 or len(i[1].strip()) == 0:
                    continue

                i[0] = self.processor1(i[0])
                i[1] = self.processor2(i[1])

                if self.training:
                    e = (Example(i[0]), Example(i[1]))
                else:
                    e = (Example(i[0]), Example(i[1]), float(i[2]))
                examples.append(e)
        return examples

    @staticmethod
    def collate_fn(example_pairs):
        def torchify(batch: List[Example]):
            idxs = pad_sequence(
                [torch.tensor(ex.embeddings, dtype=torch.long) for ex in batch],
                batch_first=True,
            )
            lengths = torch.tensor([len(e.embeddings) for e in batch], dtype=torch.long)
            return idxs, lengths

        ret = torchify([pair[0] for pair in example_pairs]), torchify(
            [pair[1] for pair in example_pairs]
        )
        if len(example_pairs[0]) == 3:
            return *ret, torch.tensor([e[2] for e in example_pairs])
        else:
            return ret

class ParaDataModule(pl.LightningDataModule):
    def __init__(self, train_data_file: str, test_data_file, args):
        super().__init__()
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.args = args

    def prepare_data(self):
        assert os.path.exists(self.train_data_file)
        assert os.path.exists(self.test_data_file)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = ParaDataset(self.train_data_file, self.args, training=True)
            # self.train, self.valid = random_split(
            #     self.train, [len(self.train) - 5000, 5000]
            # )
            # self.valid.dataset.training = False
            # HACK: Wieting et al. uses SST test set for validation
            self.valid = ParaDataset(self.test_data_file, self.args, training=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = ParaDataset(self.test_data_file, self.args, training=False)
            # TODO: idbench

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=ParaDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=ParaDataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=ParaDataset.collate_fn,
        )
