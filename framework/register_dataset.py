from .datasets import BaseDataset
DATASETS = {}


def RegisterDataset(name):
    """
        Decorator to keep track of the datasets on the framework

        @RegisterDataset('HateXPlain')
        class HateXPlain(BaseDataset):
            (...)
    """
    name = name.lower()

    def register(cls):
        if name in DATASETS:
            raise ValueError(f'Duplicate registry of dataset {name}')
        if not issubclass(cls, BaseDataset):
            raise ValueError(
                f'All models should be an extension of {BaseDataset.__name__}')
        DATASETS[name] = cls
        print(f'Registered {cls}')
        return cls

    return register
