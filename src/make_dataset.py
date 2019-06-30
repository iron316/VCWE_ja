import pickle
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from MeCab import Tagger
from tqdm import tqdm
import numpy as np


def make_dataset():

    data_path = Path("../dataset/text/")
    data_path_list = list(data_path.glob("*/*"))

    p = Pool()

    print(f'number of files is {len(data_path_list)}')
    s = time.time()
    dataset = []
    for data in p.imap(file2dataset, data_path_list):
        dataset.extend(data)
    p.close()
    print(f'###### finish segmentation time {(time.time()-s)//60} minute #####')

    surfaces = []
    for data in tqdm(dataset):
        surfaces.extend(data)
    c = Counter(surfaces)
    w_freq = np.array(list(c.values()))
    print(f'#####  number of word is {len(c)} -> {np.sum(w_freq>100)}#####')

    remove_dataset = [[w for w in sentence if c[w] > 100] for sentence in tqdm(dataset)]
    new_dataset = [sentence for sentence in tqdm(remove_dataset) if len(sentence) > 10]

    print(f'##### finish load and process text data {(time.time() - s)//60} minute #####')

    return new_dataset


def file2dataset(f):
    with f.open("r") as rf:
        texts = rf.read()
    t = Tagger('-Owakati')
    dataset = []
    texts = re.sub(r"<doc.*?>|</doc>", "", texts)
    texts = texts.replace("\n", "")
    for sentence in texts.split("。"):
        if len(sentence) == 0:
            continue
        surface = t.parse(sentence).split()
        dataset.append(surface)
    return dataset


def main():
    result_path = Path("data/dataset.pkl")
    dataset = make_dataset()

    s = time.time()
    with result_path.open("wb") as wf:
        pickle.dump(dataset, wf)
    print(f'##### finish pickle dump {time.time()-s} sec ######')


if __name__ == '__main__':
    main()
