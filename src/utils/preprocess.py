import numpy as np
from torch.utils import data

from utils.img_trans import trans_to_img


class Preprocess(data.Dataset):
    def __init__(self, corpus, word2idx, idx2word, idx2freq, window=5, neg_sample=5):
        self.corpus = corpus
        self.n_vocab = len(word2idx)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.idx2freq = idx2freq
        self.c2img = trans_to_img
        self.window = window
        self.neg_sample = neg_sample
        self.neg_weghit = np.power(
            [idx2freq[i] for i in range(self.n_vocab)], 3 / 4)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        corpus = self.corpus[i]
        print(corpus)
        corpus = np.array([self.word2idx[w] for w in corpus])
        weight = 1 - \
            np.sqrt(1e-5 / np.array([self.idx2freq[w] for w in corpus]))
        corpus = np.array([self.word2idx[w] for w in corpus])
        idx = np.random.choice(len(corpus), p=weight)
        pos_idx = []
        while True:
            i = np.random.randint(idx - self.window, idx + self.window)
            if 0 < i + idx and i + idx < len(corpus) and i != idx:
                pos_idx.append(corpus[i])
                break
        neg_idx = self.get_negative()
        pos_img = [[self.c2img(c) for c in self.idx2word(i)] for i in pos_idx]
        neg_img = [[self.c2img(c) for c in self.idx2word(i)] for i in neg_idx]

        return idx, pos_idx, pos_img, neg_idx, neg_img

    def get_negative(self, pos):
        neg = {}
        while len(neg) < self.neg_sample:
            n = np.random.choice(self.n_vocab, p=self.neg_weghit)
            if n not in neg:
                neg.append(n)
        return neg
