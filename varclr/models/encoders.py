import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from transformers import AutoModel


class Encoder(nn.Module):
    @staticmethod
    def build(args):
        return {"avg": Averaging, "lstm": LSTM, "bert": BERT}[args.model](args)

    def forward(self, idxs, lengths):
        raise NotImplementedError


class Averaging(Encoder):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = args.dropout

    def forward(self, idxs, lengths):
        word_embs = self.embedding(idxs)
        word_embs = F.dropout(word_embs, p=self.dropout, training=self.training)

        bs, max_len, _ = word_embs.shape
        mask = (
            torch.arange(max_len).cuda().expand(bs, max_len) < lengths.unsqueeze(1)
        ).float()
        pooled = (word_embs * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(dim=1)

        return pooled, (word_embs, mask)


class LSTM(Encoder):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout

        self.register_buffer("e_hidden_init", torch.zeros(2, 1, args.hidden_dim))
        self.register_buffer("e_cell_init", torch.zeros(2, 1, args.hidden_dim))

        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.lstm = nn.LSTM(
            args.dim,
            args.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

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
            torch.arange(max_len).cuda().expand(bs, max_len) < lengths.unsqueeze(1)
        ).float()
        pooled = (all_hids * mask.unsqueeze(dim=2)).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(dim=1)

        return pooled, (all_hids, mask)


from transformers import AutoModel
from varclr.models import Encoder


class BERT(Encoder):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.transformer = AutoModel.from_pretrained(args.bert_model)
        self.last_n_layer_output = args.last_n_layer_output

    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hids = output.hidden_states
        pooled = all_hids[-self.last_n_layer_output][:, 0]

        return pooled, (all_hids, attention_mask)
