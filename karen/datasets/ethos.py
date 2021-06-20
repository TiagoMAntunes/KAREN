import os
import re

import torch
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from ..base_dataset import BaseDataset
from ..register_dataset import RegisterDataset

# From https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/ca4e6c0c788f15a37f325a23b659aed4e7dd8bbe/ethos/experiments/utilities/preprocess.py#L18
def my_clean(text, stops=False, stemming=False):
    text = str(text)
    text = re.sub(r" US ", " american ", text)
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"aren't", "are not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"%", " percent ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"doesn't", "does not ", text)
    text = re.sub(r"he's", "he is ", text)
    text = re.sub(r"she's", "she is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower().split()
    text = [w for w in text if len(w) >= 2]
    if stemming and stops:
        text = [
            word for word in text if word not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        text = [
            word for word in text if word not in stopwords.words('english')]
    elif stops:
        text = [
            word for word in text if word not in stopwords.words('english')]
    elif stemming:
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    return text

@RegisterDataset("ethos")
class Ethos(BaseDataset):
    """
    ETHOS: an Online Hate Speech Detection Dataset

    Original repo: https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset

    Paper: https://arxiv.org/abs/2006.08328

    """

    def __init__(
        self,
        url="https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv",
        name="Ethos.dataset",
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

        df = pd.read_csv(self.location, delimiter=';')

        raw_text = df['comment'].values
        label_unfiltered = df['isHate'].values

        assert len(raw_text) == len(label_unfiltered)

        tokens = []
        label = []
        vocab = set()

        for i in range(len(label_unfiltered)):
            tokenized_text = nltk.word_tokenize(my_clean(raw_text[i], True, True))
            if (len(tokenized_text) > 0):
                vocab.update(set(tokenized_text))
                tokens.append(tokenized_text)
                label.append(1 if label_unfiltered[i]>=0.5 else 0)

        assert len(tokens) == len(label)
        dset_len = len(label)

        label_to_idx = {
            'Not hate': 0,
            'Is hate': 1,
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
        return Ethos(args.url_ethos, args.savename_ethos)

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument(
            "--url-ethos",
            type=str,
            default="https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv",
        )
        group.add_argument("--savename-ethos", type=str, default="Ethos.dataset")

    def words_to_idx(self):
        return self.words2idx

    def get_vocab_size(self):
        return len(self.words2idx)

    def get_labels(self):
        return list(self.label_to_idx.keys())

    def get_extra_properties(self):
        return self.extras
