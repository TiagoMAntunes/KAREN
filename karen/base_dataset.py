import torch
import os

DATASET_FOLDER = "available_datasets/"


class BaseDataset(torch.utils.data.Dataset):
    """
    This is the base class for the available datasets to extend
    All the scripts should only use the interface described in this class
    """

    def __init__(self, url, name, debug=True):
        """
        This will automatically download the dataset into the corresponding folder and set the `location` class parameter for usage in the subclass

        In case the default download method does not work, subclasses can specialize it and provide a custom download method
        """
        super(BaseDataset, self).__init__()

        # check if the dataset location exists and if it's already downloaded
        if not os.path.exists(DATASET_FOLDER):
            if debug:
                print(f"Creating datasets folder")
            os.mkdir(DATASET_FOLDER)

        location = DATASET_FOLDER + name

        if not os.path.exists(location):
            if debug:
                print(f"Saving dataset {self.__class__.__name__} to {location}")
            self.download(url, location)

        self.location = location

    def download(self, url, location, debug=True):
        """
        This class is given the url and location which are also specified in the creation of the subclass
        The location is not supposed to be changed as it will include the name that was assigned to the dataset
        """
        import wget

        print(f"Downloading file from {url}")
        wget.download(url, out=location)
        print()

    def __getitem__(self, idx):
        """
        This function must return the elements in the order given by @get_properties
        """
        raise NotImplementedError(f"No getitem built-in method implemented for class {self.__class__.__name__}")

    def __len__(self):
        raise NotImplementedError(f"No len built-in method implemented for class {self.__class__.__name__}")

    @classmethod
    def get_properties(cls):
        """
        Because datasets have different properties, it makes sense for class each one of them to state what properties it contains. The keywords should be shared across datasets for class a unified format

        Should return three list with the available content in the dataset. List 1 must contains properties that can be converted into tensor, List 2 must contain types not supported (ex: lists, strings), List 3 must contain always present properties that do not depend on the batch (e.g. TF-IDF)
        """
        raise NotImplementedError(f"No get_properties method implemented for class {cls.__name__}")

    def get_input_feat_size(self):
        raise NotImplementedError(f"No input feat size implementation")

    def get_output_feat_size(self):
        raise NotImplementedError(f"No output feat size implementation")

    @staticmethod
    def make_dataset(args):
        raise NotImplementedError(f"No make_dataset implementation")

    @staticmethod
    def add_required_arguments(parser):
        raise NotImplementedError(f"No add requried arguments implementation")

    def words_to_idx(self):
        """
        This function must return a dictionary of string to int
        It is a way to transform pre-trained word embeddings
        """
        raise NotImplementedError(f"No implementation for words_to_idx")

    def get_vocab_size(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def get_extra_properties(self):
        """
        The third list of @get_properties defines always present data.
        This function will be the one that returns such data.
        The return value must be a dictionary with the keys pointing to a data structure uniform across the whole available data
        """

        raise NotImplementedError
