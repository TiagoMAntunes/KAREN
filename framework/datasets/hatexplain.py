import torch
from base_dataset import BaseDataset

class HateXPlain(BaseDataset):

    def __init__(self, url='https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json', name='HateXPlain.dataset'):
        super().__init__(url, name)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def preprocess(self):
        pass

    @classmethod
    def get_properties(cls):
        
        pass
