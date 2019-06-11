import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
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

    set_random_seed(2434)

    with dataset_path.open("rb") as rf:
        dataset = pickle.load(rf)
    data_flatten = []
    for d in dataset:
        data_flatten.extend(d)
    word2freq = Counter(data_flatten)
    word2idx = {w: i for i, w in enumerate(word2freq.keys())}
    idx2word = {i: w for i, w in enumerate(word2freq.keys())}
    idx2freq = {i: word2freq[w] for i, w in enumerate(word2freq.keys())}

    n_vacab = len(word2idx)

    dataset = Preprocess(dataset, word2idx, idx2word, idx2freq)

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
    with open("result/embed.pkl", "wb") as wf:
        pickle.dump(embed, wf)


if __name__ == '__main__':
    main()
