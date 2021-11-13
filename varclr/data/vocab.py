from collections import Counter


class Vocab:
    unk_string = "UUUNKKK"

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

        vocab[Vocab.unk_string] = len(vocab)
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

        vocab[Vocab.unk_string] = len(vocab)
        return vocab

    @staticmethod
    def lookup(words, w, zero_unk):
        w = w.lower()
        if w in words:
            return words[w]
        else:
            if zero_unk:
                return None
            else:
                return words[Vocab.unk_string]
