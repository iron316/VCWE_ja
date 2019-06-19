import numpy as np
from torch.utils import data

from utils.img_trans import trans_to_img


class Preprocess(data.Dataset):
    def __init__(self, corpus, word2idx, idx2word, freq, n_vocab, window=5, neg_sample=5):
        self.corpus = corpus
        self.n_vocab = n_vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.freq = freq
        self.c2img = trans_to_img
        self.window = window
        self.neg_sample = neg_sample
        neg_weghit = np.power(freq, 3 / 4)
        self.neg_weghit = neg_weghit / neg_weghit.sum()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        corpus = self.corpus[i]
        corpus = np.array([self.word2idx[w] for w in corpus])
        weight = 1 - np.sqrt(1e-5 / self.freq[corpus])
        idx = np.random.choice(len(corpus), p=weight / weight.sum())
        while True:
            i = np.random.randint(max(0, idx - self.window), min(idx + self.window, len(corpus)))
            if i != idx:
                pos_idx = corpus[i]
                break
        neg_idx = self.get_negative(pos_idx)
        pos_img = np.concatenate([self.c2img(c) for c in self.idx2word[pos_idx]], axis=0)
        neg_words = [self.idx2word[i] for i in neg_idx]
        neg_max_len = max([len(w) for w in neg_words])
        neg_img = np.zeros((self.neg_sample, neg_max_len, 40, 40))
        for i, w in enumerate(neg_words):
            neg_img[i, :len(w)] += np.concatenate([self.c2img(c) for c in w], axis=0)
        return corpus[idx], pos_idx, pos_img, neg_idx, neg_img

    def get_negative(self, pos):
        neg = np.random.choice(self.n_vocab, size=self.neg_sample, p=self.neg_weghit)
        if pos in neg:
            return self.get_negative(pos)
        return neg
