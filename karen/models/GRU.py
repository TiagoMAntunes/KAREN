import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("GRU")
class GRU(BaseModel):
    """
    GRU sentiment classification
    """

    def __init__(
        self,
        out_feat,
        hidden_dim,
        linear_size,
        embeddings,
        n_layers,
        dropout,
        gru_dropout,
        bidirectional=True,
    ):
        super(GRU, self).__init__()

        self.embedding = embeddings

        self.gru = nn.GRU(
            input_size=self.embedding.weight.shape[-1],
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=gru_dropout,
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
        embedded = self.embedding(data["tokens"]).reshape(shape[0], shape[1], -1)
        self.gru.flatten_parameters()
        out, _ = self.gru(embedded)
        out = out[:, 0, :]
        out = self.linears(out)
        return out

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--gru-hidden-size", type=int, default=64, help="gru hidden size")
        group.add_argument("--gru-linear-size", type=int, default=8, help="Linear hidden size")
        group.add_argument("--gru-n-layers", type=int, default=2, help="Number of layers in the gru")
        group.add_argument("--gru-dropout-hidden", type=float, default=0.5, help="Dropout between the gru layers")
        group.add_argument("--gru-bidirectional", type=bool, default=True, help="Train Bidirectional gru or gru")

    @staticmethod
    def make_model(args):
        return GRU(
            args.out_feat,
            args.gru_hidden_size,
            args.gru_linear_size,
            args.embeddings,
            args.gru_n_layers,
            args.dropout,
            args.gru_dropout_hidden,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
