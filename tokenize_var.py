import argparse
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="txt to load untokenized data.")
    parser.add_argument("--output", help="tokenized output")
    parser.add_argument("--sp-model", help="SP model to load for evaluation")
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            var1, var2 = line.split("\t")
            fout.write(f"{' '.join(sp.encode_as_pieces(var1))}\t{' '.join(sp.encode_as_pieces(var2))}\n")
