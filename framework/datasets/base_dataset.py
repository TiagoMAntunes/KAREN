import torch


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, *args):
        raise NotImplementedError