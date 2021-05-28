import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("NetLSTM")
class NetLSTM(BaseModel):
    """
    This is a template file of a model implementation
    """

    def __init__(self, out_feat, hidden_dim, embeddings, n_layers, dropout, dropout_lstm):
        super(NetLSTM, self).__init__()

        self.word_embeddings = embeddings
        self.lstm = nn.LSTM(
            self.word_embeddings.weight.shape[-1],
            hidden_dim,
            dropout=dropout_lstm,
            bidirectional=True,
            num_layers=n_layers,
            batch_first=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(2 * n_layers * hidden_dim, out_feat)

    def forward(self, text):
        shape = text["tokens"].shape
        embeds = self.word_embeddings(text["tokens"]).reshape(shape[0], shape[1], -1)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)

        max_pool = F.adaptive_max_pool1d(lstm_out.permute(0, 2, 1), 1).reshape(shape[0], -1)
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute(0, 2, 1), 1).reshape(shape[0], -1)
        outp = torch.cat([max_pool, avg_pool], dim=1)

        y = self.dropout(self.relu(outp))
        y = self.hidden2label(y)
        return y

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--netlstm-hidden-size", type=int, default=64, help="BiLSTM hidden size")
        group.add_argument("--netlstm-linear-size", type=int, default=8, help="Linear hidden size")
        group.add_argument("--netlstm-n-layers", type=int, default=2, help="Number of layers in the BiLSTM")
        group.add_argument("--netlstm-dropout-hidden", type=float, default=0.5, help="Dropout between BiLSTM layers")

    @staticmethod
    def make_model(args):
        if args.embeddings is not None:
            embeddings = nn.Embedding.from_pretrained(torch.tensor(args.embeddings))
        else:
            embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)

        return NetLSTM(
            args.out_feat,
            args.netlstm_hidden_size,
            embeddings,
            args.netlstm_n_layers,
            args.dropout,
            args.netlstm_dropout_hidden,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
