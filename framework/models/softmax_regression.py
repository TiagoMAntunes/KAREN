import torch
import torch.nn as nn
from .base_model import BaseModel
from ..register import RegisterModel


@RegisterModel('SoftmaxRegression')
class SoftmaxRegression(BaseModel):
    """
        A simple model that applies a Feed Forward Network with a softmax at the end
    """

    def __init__(self, in_feat, out_feat, dropout=0.1):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # to mask just multiply by zero the values
        res = data['tokens'] * data['padding']
        return self.dropout(self.linear(res.float()))

    @staticmethod
    def add_required_arguments(parser):
        pass

    @staticmethod
    def make_model(parser):
        pass

    @staticmethod
    def data_requirements():
        return ['tokens', 'padding']
