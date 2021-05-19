import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
        This is the class which all models in the framework must extend
        All the scrips will be considering the interface defined here as the way to communicate with the models
    """

    def __init__(self):
        super(DefaultModel, self).__init__()

    def forward(self, *args):
        raise NotImplementedError('Model must implement a forward function that processes the input')

    @staticmethod
    def add_required_arguments(parser):
        """
            Models must define this class to request the required arguments from the parser
        """
        raise NotImplementedError('Model must specify which arguments it needs to collect from the parser')

    @staticmethod
    def make_model(parser):
        """
            Scripts should not specify which arguments they are giving to the model, all the data must be extracted by them
            Therefore, to create a model, a script should do something like:
            >>> model = modelclass.make_model(parser)
        """
        raise NotImplementedError('Model must implement a parser input extraction method, returning the model object with the correct arguments.')

    @staticmethod
    def data_requirements():
        """
            Models have different data requirements. This function must return the information that it must get from the dataset so that it can be given to it as an arguments to forward

            TODO
            TIAGO: I think the best way would be to treat the data as a dictionary and they could select keywords from there to get the specific data. This automatically filters which datasets they can access and allows for an easy extension of the way we feed in the data in case there are a lot of different types. The datasets must be restricted to have these keywords though, as to avoid having different labels: id vs idx, etc...
        """

        raise NotImplementedError('Model must specify what are the types of data it needs from the dataset')
