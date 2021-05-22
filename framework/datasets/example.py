import torch

from ..base_dataset import BaseDataset
from ..register_dataset import RegisterDataset


@RegisterDataset()
class DatasetName(BaseDataset):
    """
        This is a template file of a dataset implementation
    """

    def __init__(self, url='', name=''):
        super().__init__(url, name)
        

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def preprocess(self):
        # this is optional but a good practice for readability
        pass

    @classmethod
    def get_properties(cls):
        pass

    def get_input_feat_size(self):
        pass

    def get_output_feat_size(self):
        pass

    @staticmethod
    def make_dataset(args):
        pass

    @staticmethod
    def add_required_arguments(parser):
        pass