import torch
from .base_dataset import BaseDataset
import numpy as np
from collections import Counter

from ..register_dataset import RegisterDataset


@RegisterDataset('HateXPlain')
class HateXPlain(BaseDataset):
    """
        Original repo: https://github.com/hate-alert/HateXplain/tree/master/Data

        Paper: https://arxiv.org/abs/2012.10289

        HateXPlain is a benchmark dataset handcrafter for the specific task of hate speech detection.
    """

    def __init__(self, url='https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json', name='HateXPlain.dataset'):
        super().__init__(url, name)
        self.preprocessed = False

    def __getitem__(self, idx):
        if not self.preprocessed:
            self.preprocess()

        # return self.data.iloc[idx]
        return [self.data[x][idx] for x in self.__class__.get_properties()[0]] + [self.data[x][idx] for x in self.__class__.get_properties()[1]]

    def __len__(self):
        if not self.preprocessed:
            self.preprocess()

        return self.len

    def preprocess(self, debug=True):
        import os
        import json
        assert os.path.exists(self.location)  # sanity check

        if debug:
            print(f"Preprocessing {self.__class__.__name__}")

        with open(self.location) as f:
            data = json.load(f)

        ids = []
        tokens = []
        label = []
        annotator_labels = []
        annotator_targets = []
        rationales = []

        labels = set(['undecided'])
        vocab = set()
        for idx, content in data.items():
            ids.append(idx)
            tokens.append(content['post_tokens'])
            vocab.update(set(tokens[-1]))

            annotator_labels.append([x['label']
                                     for x in content['annotators']])
            annotator_targets.append([x['target']
                                      for x in content['annotators']])

            max_value = ('', 0)
            counter = 0

            # compares the results for each type of classification and picks the most voted one. undecided if not specified
            for val, count in Counter(annotator_labels[-1]).items():
                labels.update(set([val]))
                if count > max_value[1]:
                    max_value = (val, count)
                    counter = 1
                elif count == max_value[1]:
                    counter += 1

            label.append(max_value[0] if counter == 1 else "undecided")

            rationales.append(content['rationales'])

        # transform labels into numbers for classification
        label_to_idx = {label: i for i, label in enumerate(labels)}

        word_to_idx = {word: i for i, word in enumerate(vocab)}

        # padding of tokens and transformation
        max_size = max(map(lambda x: len(x), tokens))
        padding_mask = [[True] * len(x) + [False]
                        * (max_size - len(x)) for x in tokens]
        tokens = [list(map(lambda y: word_to_idx[y], x)) + [0]
                  * (max_size - len(x)) for x in tokens]

        # transform ids into ints
        ids_to_idx = {idx: i for i, idx in enumerate(ids)}
        ids = [ids_to_idx[idx] for idx in ids]

        # This is done like this to avoid warnings from numpy
        self.data = {
            'id': np.array(ids, dtype=int),
            'tokens': np.array(tokens, dtype=int),
            'padding': np.array(padding_mask, dtype=bool),
            'label': np.array([label_to_idx[x] for x in label], dtype=int),
            'annotator_labels': np.array([[label_to_idx[y] for y in x] for x in annotator_labels], dtype=int),
            'annotator_targets': np.array(annotator_targets, dtype=object),
            'rationales': np.array(rationales, dtype=object)
        }

        self.ids_to_idx = ids_to_idx
        self.label_to_idx = label_to_idx
        self.word_to_idx = word_to_idx
        self.preprocessed = True
        self.len = len(data)

    @classmethod
    def get_properties(cls):
        return ['id', 'tokens', 'padding', 'label', 'annotator_labels'], ['annotator_targets', 'rationales']

    def get_input_feat_size(self):
        if not self.preprocessed:
            self.preprocess()

        return self.data['tokens'].shape[-1]

    def get_output_feat_size(self):
        if not self.preprocessed:
            self.preprocess()
            
        return len(self.label_to_idx)

    @staticmethod
    def make_dataset(args):
        return HateXPlain(args.url_hatexplain, args.savename_hatexplain)
    
    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument('--url-hatexplain', type=str, default='https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json')
        group.add_argument('--savename-hatexplain', type=str, default='HateXPlain.dataset')