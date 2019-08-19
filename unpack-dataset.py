
from swda import CorpusReader
from data_config import valid_set_idx, test_set_idx
import pandas as pd

def load_swda_corpus_data(swda_directory):
    print('Loading SwDA Corpus...')
    corpus_reader = CorpusReader(swda_directory)

    talks = []
    talk_names = []
    tags_seen = {}
    tag_occurances = {}
    num_tags_seen = 0
    for transcript in corpus_reader.iter_transcripts(False):
        name = 'sw' + str(transcript.conversation_no)
        talk_names.append(name)
        conversation_content = []
        conversation_tags = []
        for utterance in transcript.utterances:
            conversation_content.append( utterance.text_words(True) )
            tag = utterance.damsl_act_tag()
            conversation_tags.append( tag )
            if tag not in tags_seen:
                tags_seen[tag] = num_tags_seen
                num_tags_seen += 1
                tag_occurances[tag] = 1
            else:
                tag_occurances[tag] += 1
        talks.append( (conversation_content, conversation_tags) )

    print('\nFound ' + str(len(tags_seen))+ ' different utterance tags.\n')

    for talk in talks:
        conversation_tags = talk[1]
        for i in range(len(conversation_tags)):
            #conversation_tags[i] = tags_seen[ conversation_tags[i] ]
            conversation_tags[i] = conversation_tags[i]

    print('Loaded SwDA Corpus.')
    return talks, talk_names, tags_seen, tag_occurances


def form_datasets(talks, talk_names, max_sentence_length, word_dimensions):
    print('Forming dataset appropriately...')

    x_train_list = []
    y_train_list = []
    x_valid_list = []
    y_valid_list = []
    x_test_list = []
    y_test_list = []
    t_i = 0
    for i in range(len(talks)):
        t = talks[i]
        if talk_names[i] in test_set_idx:
            x_test_list.append(t[0])
            y_test_list.append(t[1])
        if talk_names[i] in valid_set_idx:
            x_valid_list.append(t[0])
            y_valid_list.append(t[1])
        else:
            x_train_list.append(t[0])
            y_train_list.append(t[1])
        t_i += 1

    print('Formed dataset appropriately.')


    return ((x_train_list, y_train_list), (x_valid_list, y_valid_list), (x_test_list, y_test_list))



def main():

    tag_df = pd.read_csv("swda/Tags.tsv", sep="\t")
    tag_map = {}
    for row in tag_df.itertuples():
        tag_map[row.act_tag] = row.name


    talks_read, talk_names, tags_seen, tag_occurances = load_swda_corpus_data("swda")

    for c in talks_read:
        for u in c[0]:
            for i in range(len(u)):
                w = u[i]

                if w[-1] in {",", ".", "?", "!"}:
                    pass

                if w.rstrip(',') != w or w.rstrip('.') != w or w.rstrip('?') != w or w.rstrip('!') != w:
                    u[i] = w.rstrip(',').rstrip('.').rstrip('?').rstrip('!')



    #talks = [([[word_to_index[w.lower()] for w in u] for u in c[0]], c[1]) for c in talks_read]
    #talks_read.clear()
    talks = talks_read
    train, dev, test = form_datasets(talks, talk_names, None, None)
    data = {"train": train, "dev": dev, "test": test}

    all_labels = []
    for name, d in data.items():
        df = {"text": [], "tag": [], "label": []}

        for x,y in zip(*d):
            x = [" ".join(words) for words in x]
            df["text"].extend(x)
            df["tag"].extend(y)
            #print("y:", y)
            #input(">>>")

            df["label"].extend([tag_map.get(t, "unk") for t in y])
        all_labels.extend(df["label"])

        df = pd.DataFrame(df)
        df.to_csv("output/{}.tsv".format(name), sep="\t", index=False)

    print("num_labels:", len(set(all_labels)))

if __name__ == "__main__":
    main()


