import os
from typing import List, Union

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from varclr.data.preprocessor import CodePreprocessor
from varclr.data.vocab import Vocab
from varclr.models import urls_pretrained_model


class Encoder(nn.Module):
    @staticmethod
    def build(args) -> "Encoder":
        return {"avg": Averaging, "lstm": LSTM, "bert": BERT}[args.model].from_args(
            args
        )

    @staticmethod
    def from_pretrained(model_name: str, save_path: str = "saved/") -> "Encoder":
        return {
            "varclr-avg": Averaging,
            "varclr-lstm": LSTM,
            "varclr-codebert": BERT,
            "codebert": CodeBERT,
        }[model_name].load(save_path)

    @staticmethod
    def from_args(args) -> "Encoder":
        raise NotImplementedError

    @staticmethod
    def load(save_path: str) -> "Encoder":
        raise NotImplementedError

    def forward(self, idxs, lengths):
        raise NotImplementedError

    def encode(self, inputs: Union[str, List[str]]) -> "torch.Tensor":
        raise NotImplementedError

    def score(
        self, inputx: Union[str, List[str]], inputy: Union[str, List[str]]
    ) -> "torch.Tensor":
        if type(inputx) != type(inputy):
            raise Exception("Input X and Y must be either string or list of strings.")
        if isinstance(inputx, list) and len(inputx) != len(inputy):
            raise Exception("Input X and Y must have the same length")
        embx = self.encode(inputx)
        emby = self.encode(inputy)
        return F.cosine_similarity(embx, emby).tolist()

    def cross_score(
        self, inputx: Union[str, List[str]], inputy: Union[str, List[str]]
    ) -> "torch.Tensor":
        if isinstance(inputx, str):
            inputx = [inputx]
        if isinstance(inputy, str):
            inputy = [inputy]
        assert all(isinstance(inp, str) for inp in inputx)
        assert all(isinstance(inp, str) for inp in inputy)
        embx = self.encode(inputx)
        embx /= embx.norm(dim=1, keepdim=True)
        emby = self.encode(inputy)
        emby /= emby.norm(dim=1, keepdim=True)
        return (embx @ emby.t()).tolist()

    @staticmethod
    def decor_forward(model_forward):
        """Decorate an encoder's forward pass to deal with raw inputs."""
        processor = CodePreprocessor(
            tokenization="sp", sp_model=urls_pretrained_model.PRETRAINED_TOKENIZER
        )

        def torchify(batch):
            idxs = pad_sequence(
                [torch.tensor(ex, dtype=torch.long) for ex in batch],
                batch_first=True,
            )
            lengths = torch.tensor([len(e) for e in batch], dtype=torch.long)
            return idxs, lengths

        def tokenize_and_forward(self, inputs: Union[str, List[str]]) -> "torch.Tensor":
            if isinstance(inputs, str):
                inputs = [inputs]
            var_ids = processor(inputs)
            batch = torchify(
                [
                    [
                        Vocab.lookup(self.vocab, w, True)
                        for w in var.split()
                        if Vocab.lookup(self.vocab, w, True) is not None
                    ]
                    or [self.vocab[Vocab.unk_string]]
                    for var in var_ids
                ]
            )
            idxs, lengths = batch
            return model_forward(self, idxs, lengths)[0].detach()

        return tokenize_and_forward

    @staticmethod
    def decor_bert_forward(model_forward):
        """Decorate an encoder's forward pass to deal with raw inputs."""
        processor = CodePreprocessor()
        tokenizer = AutoTokenizer.from_pretrained(
            urls_pretrained_model.PRETRAINED_TOKENIZER
        )

        def tokenize_and_forward(self, inputs: Union[str, List[str]]) -> "torch.Tensor":
            inputs = processor(inputs)
            return_dict = tokenizer(inputs, return_tensors="pt", padding=True)
            return model_forward(
                self, return_dict["input_ids"], return_dict["attention_mask"]
            )[0].detach()

        return tokenize_and_forward


class Averaging(Encoder):
    def __init__(self, vocab_size, dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dropout = dropout

    @staticmethod
    def from_args(args):
        return Averaging(args.vocab_size, args.dim, args.dropout)

    @staticmethod
    def load(save_path: str) -> "Encoder":
        gdown.cached_download(
            urls_pretrained_model.PRETRAINED_AVG_URL,
            os.path.join(save_path, "lstm.zip"),
            md5=urls_pretrained_model.PRETRAINED_AVG_MD5,
            postprocess=gdown.extractall,
        )
        state_dict = torch.load(
            os.path.join(
                save_path, urls_pretrained_model.PRETRAINED_AVG_FOLDER, "model"
            ),
            map_location=torch.device("cpu"),
        )
        vocab_size, dim = state_dict["encoder.embedding.weight"].shape
        m = nn.Module()
        m.encoder = Averaging(vocab_size, dim, 0)
        m.load_state_dict(state_dict)
        m = m.encoder
        # HACK: for inference
        vocab = torch.load(
            os.path.join(
                save_path, urls_pretrained_model.PRETRAINED_AVG_FOLDER, "vocab"
            )
        )
        m.vocab = vocab
        return m

    def forward(self, idxs, lengths):
        word_embs = self.embedding(idxs)
        word_embs = F.dropout(word_embs, p=self.dropout, training=self.training)

        bs, max_len, _ = word_embs.shape
        mask = (
            torch.arange(max_len).to(word_embs.device).expand(bs, max_len)
            < lengths.unsqueeze(1)
        ).float()
        pooled = (word_embs * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(dim=1)

        return pooled, (word_embs, mask)

    encode = Encoder.decor_forward(forward)


class LSTM(Encoder):
    def __init__(self, hidden_dim, dropout, vocab_size, dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.register_buffer("e_hidden_init", torch.zeros(2, 1, hidden_dim))
        self.register_buffer("e_cell_init", torch.zeros(2, 1, hidden_dim))

        self.embedding = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(
            dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    @staticmethod
    def from_args(args):
        return LSTM(args.hidden_dim, args.dropout, args.vocab_size, args.dim)

    @staticmethod
    def load(save_path: str) -> "Encoder":
        gdown.cached_download(
            urls_pretrained_model.PRETRAINED_LSTM_URL,
            os.path.join(save_path, "lstm.zip"),
            md5=urls_pretrained_model.PRETRAINED_LSTM_MD5,
            postprocess=gdown.extractall,
        )
        state_dict = torch.load(
            os.path.join(
                save_path, urls_pretrained_model.PRETRAINED_LSTM_FOLDER, "model"
            ),
            map_location=torch.device("cpu"),
        )
        hidden_dim = state_dict["encoder.e_hidden_init"].shape[2]
        vocab_size, dim = state_dict["encoder.embedding.weight"].shape
        m = nn.Module()
        m.encoder = LSTM(hidden_dim, 0, vocab_size, dim)
        m.load_state_dict(state_dict)
        m = m.encoder
        # HACK: for inference
        vocab = torch.load(
            os.path.join(
                save_path, urls_pretrained_model.PRETRAINED_AVG_FOLDER, "vocab"
            )
        )
        m.vocab = vocab
        return m

    def forward(self, inputs, lengths):
        bsz, max_len = inputs.size()
        e_hidden_init = self.e_hidden_init.expand(2, bsz, self.hidden_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.hidden_dim).contiguous()
        lens, indices = torch.sort(lengths, 0, True)

        in_embs = self.embedding(inputs)
        in_embs = F.dropout(in_embs, p=self.dropout, training=self.training)

        all_hids, (enc_last_hid, _) = self.lstm(
            pack(in_embs[indices], lens.tolist(), batch_first=True),
            (e_hidden_init, e_cell_init),
        )

        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        bs, max_len, _ = all_hids.shape
        mask = (
            torch.arange(max_len).to(in_embs.device).expand(bs, max_len)
            < lengths.unsqueeze(1)
        ).float()
        pooled = (all_hids * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(dim=1)

        return pooled, (all_hids, mask)

    encode = Encoder.decor_forward(forward)


class BERT(Encoder):
    """VarCLR-CodeBERT Model."""

    def __init__(self, bert_model: str, last_n_layer_output: int = 4):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(bert_model)
        self.last_n_layer_output = last_n_layer_output

    @staticmethod
    def from_args(args):
        return BERT(args.bert_model, args.last_n_layer_output)

    @staticmethod
    def load(save_path: str) -> "BERT":
        gdown.cached_download(
            urls_pretrained_model.PRETRAINED_CODEBERT_URL,
            os.path.join(save_path, "bert.zip"),
            md5=urls_pretrained_model.PRETRAINED_CODEBERT_MD5,
            postprocess=gdown.extractall,
        )
        return BERT(
            bert_model=os.path.join(
                save_path, urls_pretrained_model.PRETRAINED_CODEBERT_FOLDER
            )
        )

    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hids = output.hidden_states
        pooled = all_hids[-self.last_n_layer_output][:, 0]

        return pooled, (all_hids, attention_mask)

    encode = Encoder.decor_bert_forward(forward)


class CodeBERT(BERT):
    """Original CodeBERT model https://github.com/microsoft/CodeBERT."""

    @staticmethod
    def load(save_path: str) -> "BERT":
        return BERT(bert_model="microsoft/codebert-base")
