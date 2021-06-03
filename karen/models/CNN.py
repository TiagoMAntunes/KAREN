import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("CNN")
class CNN(BaseModel):
    def __init__(
        self, 
        out_feat, 
        embeddings, 
        dropout, 
        filter_range=4, 
        out_channels=100
    ):
        super(CNN, self).__init__()

        self.embedding = embeddings
        filter_sizes = range(1, filter_range + 1)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=out_channels,
                          kernel_size=(fs, self.embedding.weight.shape[-1]))
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * out_channels, out_feat)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, text):
        x = self.embedding(text["tokens"]).unsqueeze(1)

        x = [self.relu(conv(x)).squeeze(-1) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in x]
        x = self.dropout(torch.cat(x, dim=1))
        x = self.fc(x)

        return x

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--cnn-filter-range", type=int, default=4,
                           help="Kernel sizes range from 1 to this value")
        group.add_argument("--cnn-out-channels", type=int,
                           default=100, help="Out channels for each convolution")

    @staticmethod
    def make_model(args):
        return CNN(
            args.out_feat,
            args.embeddings,
            args.dropout,
            args.cnn_filter_range,
            args.cnn_out_channels
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
