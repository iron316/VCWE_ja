import gc
import pickle
import re
import time
from pathlib import Path

import MeCab


def make_dataset():

    data_path = Path("../dataset/ja_wiki.txt")

    print("####### load and clean wikipedia dataset ########")
    start_time = time.time()

    t = MeCab.Tagger('-Owakati')
    t.parse('')
    dataset = []

    with data_path.open("r") as rf:
        for line in rf:
            line = re.sub(r"(.*?)", "", line)
            line = re.sub(r"<doc.*?>|</doc>", "", line)
            line = line.replace("\n", "")
            if len(line) == 0:
                continue
            for sentence in line.split("。"):
                if len(sentence) == 0:
                    continue
                node = t.parseToNode(sentence)
                surface = []
                while node:
                    surface.append(node.surface)
                    node.next
                dataset.append(surface)

    print(f"######## finish load time {time.time() - start_time} sec ########")
    return dataset


def main():
    result_path = Path("data/dataset.pkl")
    dataset = make_dataset()

    with result_path.open("wb") as wf:
        pickle.dump(dataset, wf)


if __name__ == '__main__':
    main()
