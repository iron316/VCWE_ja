import gc
import pickle
import re
import time
from pathlib import Path
from janome.tokenizer import Tokenizer
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

    print(f'##### finish load and process text data {time.time() - s} sec #####')

    return dataset


def text2dataset(texts):
    t = Tokenizer()
    dataset = []
    texts = re.sub(r"<doc.*?>|</doc>", "", texts)
    texts = texts.replace("\n", "")
    for sentence in tqdm(texts.split("。")):
        if len(sentence) == 0:
            continue
        surface = [token.surface for token in t.tokenize(sentence) if len(token.surface) <= 10]
        dataset.append(surface)
    return dataset


def main():
    result_path = Path("data/dataset.pkl")
    dataset = make_dataset()

    with result_path.open("wb") as wf:
        pickle.dump(dataset, wf)


if __name__ == '__main__':
    main()
