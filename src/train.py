import gc
import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastprogress import master_bar, progress_bar

from models.vcwe import VCWE
from utils.collate import max_pad
from utils.preprocess import Preprocess
from utils.seed import set_random_seed

plt.switch_backend('agg')


def main():

    device = torch.device('cuda:0')
    lr = 0.001
    max_epoch = 1
    batch_size = 256
    encode_dim = 200
    char_dim = 200
    atten_hidden = 256
    dataset_path = Path("data/dataset.pkl")
    embed_path = Path("result/embed.pkl")

    set_random_seed(2434)

    print('##### load dataset #####')

    with dataset_path.open("rb") as rf:
        dataset = pickle.load(rf)
    data_flatten = []
    for d in dataset:
        data_flatten.extend(d)
    word2freq = Counter(data_flatten)
    unique_words = list(word2freq.keys())
    unique_words.sort()
    word2idx = {w: i for i, w in enumerate(unique_words)}
    idx2word = {i: w for i, w in enumerate(unique_words)}
    freq = np.array([word2freq[w] for w in unique_words])
    n_vacab = len(unique_words)

    del unique_words, word2freq, data_flatten
    gc.collect()

    print('##### preprocess dataset #####')

    dataset = Preprocess(dataset, word2idx, idx2word, freq, n_vacab)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, collate_fn=max_pad, num_workers=4)

    model = VCWE(encode_dim, n_vacab, char_dim, atten_hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mb = master_bar(range(max_epoch))

    all_loss = []
    for epoch in mb:
        s = time.time()
        epoch_loss = 0.0
        for pos_u, pos_v, pos_img, neg_v, neg_img in progress_bar(data_loader, parent=mb):
            import pdb
            pdb.set_trace()
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            pos_img = pos_img.to(device)
            neg_v = neg_v.to(device)
            neg_img = neg_img.to(device)
            loss = model(pos_u, pos_v, pos_img, neg_v, neg_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.items() / len(data_loader)
        elapsed_time = time.time() - s
        message = f'epoch {epoch+1}/{max_epoch} loss : {epoch_loss} time {elapsed_time} sec'
        mb.write(message)
        all_loss.append(epoch_loss)
    torch.save(model.state_dict(), "result/model")
    print("########## finish ##########")

    embed = model.u_embed.weight.cpu().numpy
    with embed_path.open("wb") as wf:
        pickle.dump(embed, wf)


if __name__ == '__main__':
    main()
