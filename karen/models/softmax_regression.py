import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel('SoftmaxRegression')
class SoftmaxRegression(BaseModel):
    """
        A simple model that applies a Feed Forward Network with a softmax at the end
    """

    def __init__(
        self, 
        in_feat, 
        hidden_size, 
        out_feat, 
        dropout=0.1
    ):
        super(SoftmaxRegression, self).__init__()
        self.tohidden = nn.Linear(in_feat, hidden_size)
        self.toout = nn.Linear(hidden_size, out_feat)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # to mask just multiply by zero the values
        res = (data['tokens'] * data['mask']).float()
        out = self.activation(self.dropout(self.tohidden(res)))
        return self.dropout(self.toout(out))

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument('--softmaxregression-hidden-size', type=int,
                           default=512, help='The size of the hidden layer')

    @staticmethod
    def make_model(args):
        return SoftmaxRegression(
            args.in_feat,
            args.softmaxregression_hidden_size,
            args.out_feat,
            dropout=args.dropout
        )

    @staticmethod
    def data_requirements():
        return ['tokens', 'mask']
