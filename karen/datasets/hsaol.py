import os
import re

import torch
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import *

from ..base_dataset import BaseDataset
from ..register_dataset import RegisterDataset

# From https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/src/Automated%20Hate%20Speech%20Detection%20and%20the%20Problem%20of%20Offensive%20Language%20Python%203.6.ipynb
stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


@RegisterDataset("Hsaol")
class Hsaol(BaseDataset):
    """
    Automated Hate Speech Detection and the Problem of Offensive Language

    Original repo: https://github.com/t-davidson/hate-speech-and-offensive-language

    Paper: https://arxiv.org/abs/1703.04009

    """

    def __init__(
        self,
        url="https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        name="Hsaol.dataset",
    ):
        super().__init__(url, name)
        self.preprocessed = False

    def __getitem__(self, idx):
        if not self.preprocessed:
            self.preprocess()

        return [self.data[x][idx] for x in self.__class__.get_properties()[0]] + [
            self.data[x][idx] for x in self.__class__.get_properties()[1]
        ]

    def __len__(self):
        if not self.preprocessed:
            self.preprocess()

        return self.len

    def preprocess(self, debug=True):
        assert os.path.exists(self.location)  # sanity check

        if debug:
            print(f"Preprocessing {self.__class__.__name__}")

        df = pd.read_csv(self.location)

        raw_tweets = df['tweet'].values
        label = df['class'].values

        assert len(raw_tweets) == len(label)
        dset_len = len(label)

        tokens = []
        vocab = set()

        for i in range(dset_len):
            tokenized_tweet = tokenize(preprocess(raw_tweets[i]))
            vocab.update(set(tokenized_tweet))
            tokens.append(tokenized_tweet)

        label_to_idx = {
            'hate speech': 0,
            'offensive language': 1,
            'neither': 2
        }
        word_to_idx = {word: i for i, word in enumerate(sorted(list(vocab)))}

        # padding of tokens and transformation
        max_size = max(map(lambda x: len(x), tokens))
        padding_mask = [[True] * len(x) + [False] * (max_size - len(x)) for x in tokens]

        text = [" ".join(x) for x in tokens]
        newtokens = [list(map(lambda y: word_to_idx[y], x)) + [0] * (max_size - len(x)) for x in tokens]

        ids = [i for i in range(dset_len)]

        # This is done like this to avoid warnings from numpy
        self.data = {
            "id": np.array(ids, dtype=int),
            "tokens": np.array(newtokens, dtype=int),
            "mask": np.array(padding_mask, dtype=bool),
            "label": np.array(label, dtype=int),
            "text": text,
        }

        self.extras = {}

        self.label_to_idx = label_to_idx
        self.words2idx = word_to_idx
        self.preprocessed = True
        self.len = dset_len

    @classmethod
    def get_properties(cls):
        return (
            ["id", "tokens", "mask", "label"],
            ["text"],
            [],
        )

    def get_input_feat_size(self):
        if not self.preprocessed:
            self.preprocess()

        return self.data["tokens"].shape[-1]

    def get_output_feat_size(self):
        if not self.preprocessed:
            self.preprocess()

        return len(self.label_to_idx)

    @staticmethod
    def make_dataset(args):
        return Hsaol(args.url_hsaol, args.savename_hsaol)

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument(
            "--url-hsaol",
            type=str,
            default="https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        )
        group.add_argument("--savename-hsaol", type=str, default="Hsaol.dataset")

    def words_to_idx(self):
        return self.words2idx

    def get_vocab_size(self):
        return len(self.words2idx)

    def get_labels(self):
        return list(self.label_to_idx.keys())

    def get_extra_properties(self):
        return self.extras
