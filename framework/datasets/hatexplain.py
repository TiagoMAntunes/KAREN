import torch
from base_dataset import BaseDataset
import pandas as pd
from collections import Counter


class HateXPlain(BaseDataset):

    def __init__(self, url='https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json', name='HateXPlain.dataset'):
        super().__init__(url, name)
        self.preprocessed = False

    def __getitem__(self, idx):
        if not self.preprocessed:
            self.preprocess()

        return self.data.iloc[idx]

    def __len__(self):
        if not self.preprocessed:
            self.preprocess()

        return len(self.data)

    def preprocess(self, debug=True):
        import os
        import json
        assert os.path.exists(self.location)  # sanity check

        if debug:
            print(f"Preprocessing {self.__class__.__name__}")

        with open(self.location) as f:
            data = json.load(f)

        dataframe_data = []
        for idx, content in data.items():
            entry = {}

            entry['id'] = idx
            entry['tokens'] = content['post_tokens']

            entry['annotator_labels'] = [x['label']
                                         for x in content['annotators']]
            entry['annotator_targets'] = [x['target']
                                          for x in content['annotators']]

            max_value = ('', 0)
            counter = 0

            # compares the results for each type of classification and picks the most voted one. undecided if not specified
            for val, count in Counter(entry['annotator_labels']).items():
                if count > max_value[1]:
                    max_value = (val, count)
                    counter = 1
                elif count == max_value[1]:
                    counter += 1
            
            entry['label'] = max_value[0] if counter == 1 else "undecided"
            # TODO is this slow?

            entry['rationales'] = content['rationales']
            dataframe_data.append(entry)

        self.data = pd.DataFrame(dataframe_data)
        self.preprocessed = True

        assert set(self.data.columns) == self.__class__.get_properties()

    @classmethod
    def get_properties(cls):
        return set(['id', 'tokens', 'label', 'annotator_labels', 'annotator_targets', 'rationales'])
