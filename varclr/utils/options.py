def add_options(parser):
    # fmt: off
    # Dataset
    parser.add_argument("--train-data-file", default="moses_nli_for_simcse.csv", help="training data")
    parser.add_argument("--valid-data-file", default=None, type=str, help="validation data")
    parser.add_argument("--test-data-files", default="STS-B/original/sts-test.tsv,STS/STS17-test/STS.input.track5.en-en.txt,idbench/small_pair_wise.csv,idbench/medium_pair_wise.csv,idbench/large_pair_wise.csv", help="test data")
    parser.add_argument("--zero-unk", default=1, type=int, help="whether to ignore unknown tokens")
    parser.add_argument("--ngrams", default=3, type=int, help="whether to use character n-grams")
    parser.add_argument("--tokenization", default="sp", type=str, choices=["sp", "ngrams"], help="which tokenization to use")
    parser.add_argument("--sp-model", default="acl19-simple/en-es.os.1m.tok.sp.20k.model", help="SP model to load for evaluation")
    parser.add_argument("--vocab-path", default="acl19-simple/en-es.os.1m.tok.sp.20k.vocab", type=str, help="Path to vocabulary")
    parser.add_argument("--num-workers", default=4, type=int, help="Path to vocabulary")

    # Model
    parser.add_argument("--model", default="avg", choices=["avg", "lstm", "attn", "bert"], help="type of base model to train.")
    parser.add_argument("--bert-model", default="microsoft/codebert-base-mlm", help="type of bert model to load.")
    parser.add_argument("--dim", default=300, type=int, help="dimension of input embeddings")
    parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
    parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling in for LSTM")
    parser.add_argument("--delta", default=0.4, type=float, help="margin size for margin ranking loss")
    parser.add_argument("--nce-t", default=0.05, type=float, help="temperature for noise contrastive estimation loss")
    parser.add_argument("--temperature", default=100, type=float, help="temperature for biattn scorer")
    parser.add_argument("--last-n-layer-output", default=1, type=int, help="last layer representation used as output")

    # Training
    parser.add_argument("--name", default="Ours-FT", help="method name")
    parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
    parser.add_argument("--grad-clip", default=1., type=float, help='clip threshold of gradients')
    parser.add_argument("--epochs", default=300, type=int, help="number of epochs to train")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0, type=float, help="dropout rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="size of batches")
    parser.add_argument("--load-file", help="filename to load a pretrained model.")
    parser.add_argument("--test", action="store_true", help="only do evaluation")
    parser.add_argument("--train-percent", default=1.0, type=float, help="percentage of data used for training")
    parser.add_argument("--seed", default=42, type=int)
    # fmt: on
