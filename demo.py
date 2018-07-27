from __future__ import print_function

from collections import defaultdict
from operator import itemgetter
from swda import CorpusReader

import constants as const



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

tag_counts()






