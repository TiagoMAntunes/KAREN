import torch
import torch.nn as nn
import math

from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("Transformer")
class Transformer(BaseModel):
    """
    Simple transformer model for prediction

    Because this is a multi-class classification model, no decoder part is used consisting only of an encoder module replicated multiple times

    Allows for usage of pre-trained word embeddings
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        hidden_size,
        n_heads,
        n_layers,
        embeddings,
        dropout,
        batch_size,
    ):
        super(Transformer, self).__init__()
        self.embeddings = embeddings
        size = self.embeddings.weight.shape[-1]

        encoder = nn.TransformerEncoderLayer(size, n_heads, dim_feedforward=hidden_size, dropout=dropout)

        self.model = nn.TransformerEncoder(encoder, n_layers)
        self.linear = nn.Linear(in_feat * size, out_feat)

    def forward(self, data):
        src = self.embeddings(data["tokens"]).float()
        src = src.reshape(src.shape[1], src.shape[0], src.shape[2])
        out = self.model(src, src_key_padding_mask=~data["mask"])
        out = out.reshape(out.shape[1], -1)
        return self.linear(out)

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument(
            "--transformer-hidden-size",
            type=int,
            default=2048,
            help="Size of the feedforward network model inside the encoder layer",
        )
        group.add_argument(
            "--transformer-n-heads", type=int, default=8, help="Number of multiheadattention in each encoder layer."
        )
        group.add_argument(
            "--transformer-n-layers", type=int, default=4, help="The number of encoder layers to be stacked"
        )

    @staticmethod
    def make_model(args):
        return Transformer(
            args.in_feat,
            args.out_feat,
            args.transformer_hidden_size,
            args.transformer_n_heads,
            args.transformer_n_layers,
            args.embeddings,
            args.dropout,
            args.batch_size,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
