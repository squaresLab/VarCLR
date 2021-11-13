import sentencepiece as spm
from transformers import AutoTokenizer


class Tokenizer:
    @staticmethod
    def build(sp_model):
        if "sp.20k.model" in sp_model:
            return SPTokenizer(sp_model)
        elif "bert" in sp_model:
            return PretrainedTokenizer(sp_model)
        elif "split" in sp_model:
            return SplitTokenizer()
        else:
            raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError


class SplitTokenizer(Tokenizer):
    def encode(self, text):
        return text.strip().split()


class SPTokenizer(Tokenizer):
    def __init__(self, model_path) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        return self.sp.EncodeAsPieces(text)


class PretrainedTokenizer(Tokenizer):

    _instance = None

    @staticmethod
    def get_instance():
        return PretrainedTokenizer._instance

    @staticmethod
    def set_instance(tokenizer_name):
        PretrainedTokenizer._instance = AutoTokenizer.from_pretrained(tokenizer_name)

    def __init__(self, tokenizer_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, text):
        return list(
            map(
                str,
                self.tokenizer.encode(text, add_special_tokens=False, truncation=True),
            )
        )
