import sentencepiece as spm
from transformers import AutoTokenizer


class Tokenizer:
    @staticmethod
    def build(args):
        if "sp.20k.model" in args.sp_model:
            return SPTokenizer(args.sp_model)
        elif "bert" in args.sp_model:
            return PretrainedTokenizer(args.sp_model)
        elif "split" in args.sp_model:
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
