import gc
import pickle
import re
import time
from pathlib import Path

import MeCab
from tqdm import tqdm


def make_dataset():

    data_path = Path("../dataset/text/")
    data_path_list = list(data_path.glob("*/*"))

    print(f'number of files is {len(data_path_list)}')

    dataset = []
    s = time.time()

    for path in tqdm(data_path_list):
        with path.open("r") as rf:
            texts = rf.read()
        dataset += text2dataset(texts)
        del texts
        gc.collect()

    print(f'##### finish load and process text data {s-time.time()} sec #####')

    return dataset


def text2dataset(texts):
    t = MeCab.Tagger('-Owakati')
    t.parse('')
    dataset = []
    texts = re.sub(r"(.*?)", "", texts)
    texts = re.sub(r"<doc.*?>|</doc>", "", texts)
    texts = texts.replace("\n", "")
    import pdb
    pdb.set_trace()
    for sentence in tqdm(texts.split("。")):
        if len(sentence) == 0:
            continue
        node = t.parseToNode(sentence)
        surface = []
        while node:
            surface.append(node.surface)
            node.next
        dataset.append(surface)
    return dataset


def main():
    result_path = Path("data/dataset.pkl")
    dataset = make_dataset()

    with result_path.open("wb") as wf:
        pickle.dump(dataset, wf)


if __name__ == '__main__':
    main()
