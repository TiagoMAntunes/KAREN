import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("BiLSTM")
class BiLSTM(BaseModel):
    """
    BiLSTM sentiment classification
    """

    def __init__(
        self,
        out_feat,
        hidden_dim,
        embeddings,
        n_layers,
        dropout,
        dropout_lstm,
        bidirectional=True,
    ):
        super(BiLSTM, self).__init__()

        self.embedding = embeddings

        self.lstm = nn.LSTM(
            input_size=self.embedding.weight.shape[-1],
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidirectional,
        )
        self.number_of_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_dim * self.number_of_directions, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        shape = data["mask"].shape
        embedded = self.embedding(data["tokens"]).view(shape[0], shape[1], -1)
        out, (hidden, cell) = self.lstm(embedded.float())
        out = out[:, 0, :]
        out = self.dropout(out)
        out = self.fc1(out)
        return out

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument(
            "--bilstm-hidden-size",
            type=int,
            default=64,
            help="BiLSTM hidden size",
        )
        group.add_argument("--bilstm-n-layers", type=int, default=1, help="Number of layers in the BiLSTM")
        group.add_argument("--bilstm-dropout-hidden", type=float, default=0.5, help="Dropout between the BiLSTM layers")
        group.add_argument("--bilstm-bidirectional", type=bool, default=True, help="Train BiLSTM or LSTM")

    @staticmethod
    def make_model(args):
        if args.embeddings is not None:
            embeddings = nn.Embedding.from_pretrained(torch.tensor(args.embeddings))
        else:
            embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)

        return BiLSTM(
            args.out_feat,
            args.bilstm_hidden_size,
            embeddings,
            args.bilstm_n_layers,
            args.dropout,
            args.bilstm_dropout_hidden,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
