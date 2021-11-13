import io
import os
import random
from typing import List, Optional, Text, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from varclr.data.preprocessor import Preprocessor
from varclr.data.vocab import Vocab
from varclr.models.tokenizers import PretrainedTokenizer


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
                wd = Vocab.lookup(words, gr, zero_unk)
                if wd is not None:
                    self.embeddings.append(wd)
        elif tokenization == "sp":
            arr = self.sentence.split()
            if scramble_rate:
                if random.random() <= scramble_rate:
                    random.shuffle(arr)
            for i in arr:
                wd = Vocab.lookup(words, i, zero_unk)
                if wd is not None:
                    self.embeddings.append(wd)
        else:
            raise NotImplementedError
        if len(self.embeddings) == 0:
            self.embeddings = [words[Vocab.unk_string]]


class RenamesDataset(Dataset):
    def __init__(self, data_file: str, args, training=True) -> None:
        super().__init__()
        self.data_file = data_file
        self.tokenization = args.tokenization
        self.ngrams = args.ngrams
        self.scramble_rate = args.scramble_rate
        self.zero_unk = args.zero_unk
        self.training = training
        self.processor1, self.processor2 = Preprocessor.build(data_file, args)
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
        spliter = "," if "csv" in self.data_file else "\t"
        with io.open(self.data_file, "r", encoding="utf-8") as f:
            for idx, i in enumerate(f):
                if "csv" in self.data_file and idx == 0:
                    # skip the first line in IdBench csv
                    continue
                if i in finished:
                    continue
                else:
                    finished.add(i)

                i = i.split(spliter)
                if len(i[0].strip()) == 0 or len(i[1].strip()) == 0:
                    continue

                i[0] = self.processor1(i[0])
                i[1] = self.processor2(i[1])

                if self.training:
                    e = (Example(i[0]), Example(i[1]))
                else:
                    if np.isnan(float(i[2])):
                        continue
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
            return ret[0], ret[1], torch.tensor([e[2] for e in example_pairs])
        else:
            return ret

    @staticmethod
    def collate_fn_transformers(example_pairs):
        def torchify(batch: List[Example]):
            tokenizer = PretrainedTokenizer.get_instance()
            ret = tokenizer(
                [ex.sentence for ex in batch], return_tensors="pt", padding=True
            )
            return ret["input_ids"], ret["attention_mask"]

        ret = torchify([pair[0] for pair in example_pairs]), torchify(
            [pair[1] for pair in example_pairs]
        )
        if len(example_pairs[0]) == 3:
            return ret[0], ret[1], torch.tensor([e[2] for e in example_pairs])
        else:
            return ret


class RenamesDataModule(pl.LightningDataModule):
    def __init__(
        self, train_data_file: str, valid_data_file: str, test_data_files: str, args
    ):
        super().__init__()
        self.train_data_file = train_data_file
        self.valid_data_file = valid_data_file
        self.test_data_files = test_data_files.split(",")
        self.train_percent = args.train_percent
        self.args = args

    def prepare_data(self):
        assert os.path.exists(self.train_data_file)
        assert all(os.path.exists(test) for test in self.test_data_files)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = RenamesDataset(self.train_data_file, self.args, training=True)
            if self.valid_data_file is None:
                self.train, self.valid = random_split(
                    self.train, [len(self.train) - 1000, 1000]
                )
                self.valid.training = False
                self.valid.data_file = self.train_data_file
            else:
                self.valid = RenamesDataset(
                    self.valid_data_file, self.args, training=False
                )
            num_for_training = int(len(self.train) * self.train_percent)
            self.train = random_split(
                self.train, [num_for_training, len(self.train) - num_for_training]
            )[0]

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.tests = [
                RenamesDataset(test_data_file, self.args, training=False)
                for test_data_file in self.test_data_files
            ]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=RenamesDataset.collate_fn
            if self.args.model != "bert"
            else RenamesDataset.collate_fn_transformers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=RenamesDataset.collate_fn
            if self.args.model != "bert"
            else RenamesDataset.collate_fn_transformers,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                test,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=RenamesDataset.collate_fn
                if self.args.model != "bert"
                else RenamesDataset.collate_fn_transformers,
            )
            for test in self.tests
        ]
