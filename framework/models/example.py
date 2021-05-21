import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel()
class ModelName(BaseModel):
    """
        This is a template file of a model implementation
    """

    def __init__(self):
        super(ModelName, self).__init__()

    def forward(self, data):
        pass

    @staticmethod
    def add_required_arguments(parser):
        pass

    @staticmethod
    def make_model(args):
        pass

    @staticmethod
    def data_requirements():
        pass
