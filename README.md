# simple-and-effective-paraphrastic-similarity

Code to train models from "Simple and Effective Paraphrastic Similarity from Parallel Translations".

The code is written in Python 3.7 and requires numpy, scipy, sentencepiece, sacremoses, and PyTorch >= 1.0 libraries.

To get started, download the data files used for training from http://www.cs.cmu.edu/~jwieting and download the STS evaluation data:

    wget http://www.cs.cmu.edu/~jwieting/acl19-simple.zip
    unzip acl19-simple.zip
    rm acl19-simple.zip
    bash download_sts17.sh

If you use our code, models, or data for your work please cite:

    @inproceedings{wieting19simple,
        title={Simple and Effective Paraphrastic Similarity from Parallel Translations},
        author={Wieting, John and Gimpel, Kevin and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        booktitle={Proceedings of the Association for Computational Linguistics},
        url={https://arxiv.org/abs/1909.13872},
        year={2019}
    }

To train sp-average models in language xx on GPU, choices are es, ar, or tr:

    python main.py --data-file acl19-simple/en-xx.os.1m.tok.20k.txt --model avg --dim 300 --epochs 10 --share-vocab 1 --dropout 0.3 --sp-model acl19-simple/en-xx.os.1m.tok.sp.20k.model

To train trigram-average models in language xx on GPU:

    python main.py --data-file acl19-simple/en-xx.os.1m.tok.txt --model avg --dim 300 --epochs 10 --ngrams 3 --share-vocab 1 --dropout 0.3

To train the bidirectional LSTM models in language xx on GPU:

    python main.py --data-file acl19-simple/en-xx.os.1m.tok.20k.txt --model lstm --dim 300 --epochs 10 --dropout 0.3 --scramble-rate 0.3 --share-vocab 1 --share-encoder 1 --sp-model acl19-simple/en-xx.os.1m.tok.sp.20k.model
