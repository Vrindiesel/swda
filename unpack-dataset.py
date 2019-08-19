
from swda import CorpusReader
from data_config import valid_set_idx, test_set_idx, train_set_idx
import pandas as pd

import json

def get_partition(conv_id):
    if conv_id in valid_set_idx:
        return "dev"
    elif conv_id in test_set_idx:
        return "test"
    else:
        return "train"


def load_swda_corpus_data(swda_directory):
    print('Loading SwDA Corpus...')
    corpus_reader = CorpusReader(swda_directory)
    conversations = []

    for transcript in corpus_reader.iter_transcripts(display_progress=False):
        name = 'sw' + str(transcript.conversation_no)

        conv = {
            "name": name,
            "utterances": [],
            "partition_name": get_partition(name)
        }

        for j, utterance in enumerate(transcript.utterances):
            utt = {
                "text": " ".join(utterance.text_words(filter_disfluency=True)),
                "act_tag": utterance.act_tag,
                "damsl_act_tag": utterance.damsl_act_tag(),
                "caller": utterance.caller,
            }

            #utt_text = " ".join(utterance.text_words(filter_disfluency=True))
            #print("[{}] {}".format(j, utt_text))
            #print("\t==>", utterance.act_tag, utterance.damsl_act_tag())

            conv["utterances"].append(utt)

        conversations.append(conv)

    corpus = {
        "partition_source": "https://github.com/Franck-Dernoncourt/naacl2016",
        "train_ids": list(train_set_idx),
        "test_ids": list(test_set_idx),
        "dev_set_ids": list(valid_set_idx),
        "conversations": conversations
    }

    return corpus





def main():

    corpus = load_swda_corpus_data("swda")

    with open("swda-corpus.json", "w") as fout:
        json.dump(corpus, fout)






if __name__ == "__main__":
    main()


