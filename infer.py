import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import CodePreprocessor


def forward(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    all_hids = output.hidden_states
    pooled = all_hids[-4][:, 0]
    return pooled


class MockArgs:
    def __init__(self):
        self.tokenization = ""


def batcher(batch_size):
    uniq = set()
    with open(sys.argv[1]) as f:
        vars = []
        for var in f:
            var = processor(var.strip())
            if var not in uniq:
                uniq.add(var)
                vars.append(var)
            if len(vars) == batch_size:
                yield vars
                vars = []
    yield vars


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("bert_saved/")
    model.to(device)
    processor = CodePreprocessor(MockArgs())
    ret_dict = dict(
        vars=[],
        embs=[]
    )
    for idx, vars in enumerate(tqdm(batcher(64))):
        ret = tokenizer(vars, return_tensors="pt", padding=True)
        embs = forward(model, ret['input_ids'].to(device), ret['attention_mask'].to(device)).detach().cpu()
        ret_dict["vars"].extend(["".join([word.capitalize() if idx > 0 else word for idx, word in enumerate(var.split())]) for var in vars])
        ret_dict["embs"].extend(embs)
    ret_dict["embs"] = torch.stack(ret_dict["embs"])
    print(len(ret_dict["vars"]))
    print(ret_dict["embs"].shape)
    torch.save(ret_dict, "saved")
