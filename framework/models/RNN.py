import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("RNN")
class RNN(BaseModel):
    """
    RNN sentiment classification
    """

    def __init__(
        self,
        out_feat,
        hidden_dim,
        linear_size,
        embeddings,
        n_layers,
        non_linearity,
        dropout,
        rnn_dropout,
        bidirectional=True,
    ):
        super(RNN, self).__init__()

        self.embedding = embeddings

        self.rnn = nn.RNN(
            input_size=self.embedding.weight.shape[-1],
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity=non_linearity,
            bias=True,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        self.number_of_directions = 2 if bidirectional else 1
        self.linears = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim * self.number_of_directions, linear_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_size, out_feat),
        )

    def forward(self, data):
        shape = data["tokens"].shape
        embedded = self.embedding(data["tokens"]).reshape(
            shape[0], shape[1], -1)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(embedded)
        out = out[:, 0, :]
        out = self.linears(out)
        return out

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--rnn-hidden-size", type=int,
                           default=64, help="rnn hidden size")
        group.add_argument("--rnn-linear-size", type=int,
                           default=8, help="Linear hidden size")
        group.add_argument("--rnn-n-layers", type=int,
                           default=2, help="Number of layers in the rnn")
        group.add_argument("--rnn-dropout-hidden", type=float,
                           default=0.5, help="Dropout between the rnn layers")
        group.add_argument("--rnn-bidirectional", type=bool,
                           default=True, help="Train Bidirectional RNN or RNN")
        group.add_argument("--rnn-non-linearity", type=str, default="tanh",
                           help="Non linearity function used in RNN, must be one of 'tanh' and 'relu'")

    @staticmethod
    def make_model(args):
        return RNN(
            args.out_feat,
            args.rnn_hidden_size,
            args.rnn_linear_size,
            args.embeddings,
            args.rnn_n_layers,
            args.rnn_non_linearity,
            args.dropout,
            args.rnn_dropout_hidden,
            args.rnn_bidirectional
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
