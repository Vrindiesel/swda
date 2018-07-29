from __future__ import print_function

from collections import defaultdict
from operator import itemgetter
from swda import CorpusReader

from constants import SwDA
import pandas as pd
import random

def tag_counts():
    """Gather and print counts of the tags."""
    d = defaultdict(int)
    corpus = CorpusReader('swda')
    # Loop, counting tags:
    for utt in corpus.iter_utterances(display_progress=True):
        d[utt.damsl_act_tag()] += 1

    # Print the results sorted by count, largest to smallest:
    for key, val in sorted(d.items(), key=itemgetter(1), reverse=True):
        print(key, val)

    print("num tag labels:", len(d))


def get_basename(utt):
    name = utt.ptb_basename
    name = name[name.find("/")+1:]
    if len(name) < 1:
        name = "sw{}".format(utt.conversation_no)

    return name



def load_dataset_OLD():
    corpus = CorpusReader('swda')
    data = defaultdict(list)
    N = 221616

    not_found_set = []
    found = []
    skipp_count = 0
    for utt in corpus.iter_utterances(display_progress=False):
        d = {
            "basename": get_basename(utt),
            "words": " ".join(utt.pos_words()),
            "label": utt.damsl_act_tag(),
        }

        if len(d["words"]) < 1:
            #print("skipping ... ")
            skipp_count += 1
            #print(utt.text_words())
            continue

        not_found = True
        for splitname in SwDA:
            if d["basename"] in SwDA[splitname]:
                not_found = False
                data[splitname].append(d)
                found.append(d["basename"])

        if not_found:
            not_found_set.append(d["basename"])


    not_found_set = set(not_found_set)
    print("not found count:", len(not_found_set))
    print("skipp count:", skipp_count)
    #for name in not_found_set:
    #    print(name)

    print("label counts:")
    for k, v in data.items():
        print("\t{} size:".format(k), len(v))

    # 1115 seen dialogs, 19 unseen dialogs.
    size = len(set(found))
    #assert size == 1115 + 19, "{} != 1115 + 19; difference = {}".format(size, 1115 + 19 - size)

    return data



def load_dataset():
    corpus = CorpusReader('swda')
    data = []
    skipp_count = 0

    for utt in corpus.iter_utterances(display_progress=False):
        d = {
            "basename": get_basename(utt),
            "words": " ".join(utt.pos_words()),
            "label": utt.damsl_act_tag(),
        }

        if len(d["words"]) < 1:
            skipp_count += 1
            continue
        data.append(d)

    print("skipp count:", skipp_count)
    return data


def build_dataset(all_data, split_ratio=(0.8, 0.1, 0.1)):
    """
    Make it reproducible.

    :param all_data:
    :param split_ratio:
    :return:
    """
    assert sum(split_ratio) == 1.0
    print("num examples:", len(all_data))

    label_data = defaultdict(list)
    for row in all_data.itertuples():
        label_data[row.label].append(("{}-{}-{}".format(row.basename, row.words, row.label), row))

    split_data = defaultdict(list)
    random.seed(230)

    print("num labels:", len(label_data))
    #print("label counts:")
    # make sure that the distribution of labels in each data split
    # is representative of the dataset as a whole
    for k, v in label_data.items():
        #print("\t{} size:".format(k), len(v))
        # Make sure that the examples have a fixed order before shuffling.
        # The call to .sort() makes sure that if you build examples in a
        # different way, the output is still the same.
        v.sort()
        v = [val[1] for val in v]
        random.shuffle(v)

        split_1 = int(split_ratio[0] * len(v))
        split_data["train"] += v[:split_1]

        split_2 = int((split_ratio[0]+split_ratio[1]) * len(v))
        split_data["dev"] += v[split_1:split_2]
        split_data["test"] += v[split_2:]


    for k,v in split_data.items():
        print("{}: {}".format(k, len(v)))

    return split_data


def convert2text():
    splitnames = ["train", "dev", "test"]
    for splitname in splitnames:
        df = pd.read_csv("ready_data/swda-{}.csv".format(splitname))
        sents = [row.words.lower().strip().rstrip(" \n") for row in df.itertuples()]
        with open("ready_data/{}-sents.txt".format(splitname), "w") as fout:
            fout.write("\n".join(sents))

        labels = [row.label.strip().rstrip(" \n") for row in df.itertuples()]
        with open("ready_data/{}-labels.txt".format(splitname), "w") as fout:
            fout.write("\n".join(labels))


def split_data():
    all_data = pd.read_csv("ready_data/all_data.csv")
    dataset = build_dataset(all_data, (0.8, 0.1, 0.1))
    for splitname, vals in dataset.items():
        df = pd.DataFrame(vals)
        df.to_csv("ready_data/swda-{}.csv".format(splitname), index=False)

def main():
    #data = load_dataset()
    #all_data = pd.DataFrame(data)
    #all_data.to_csv("ready_data/all_data.csv", index=False)

    #split_data()

    convert2text()




if __name__ == '__main__':
    main()



