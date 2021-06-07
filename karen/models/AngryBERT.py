from re import S
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from ..base_model import BaseModel
from ..register_model import RegisterModel
from transformers import BertTokenizer, BertModel


class FFN(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(FFN, self).__init__()
        self.in2hid = nn.Linear(in_feat, in_feat)
        self.hid2out = nn.Linear(in_feat, out_feat)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hid = self.activation(self.dropout(self.in2hid(input)))
        return self.hid2out(hid)


@RegisterModel("AngryBERT")
class AngryBERT(BaseModel):
    """
    AngryBERT https://arxiv.org/pdf/2103.11800.pdf

    TODO: Secondary task - framework doesn't support it yet
    """

    def __init__(
        self,
        bilstm_n_layers,
        bilstm_hidden_dim,
        ffn_dim,
        out_feat,
        embeddings,
        dropout,
        device,
    ):
        super(AngryBERT, self).__init__()

        self.embeddings = embeddings  # bilstm embeddings
        self.bilstm = nn.LSTM(
            input_size=self.embeddings.weight.shape[-1],
            hidden_size=bilstm_hidden_dim,
            num_layers=bilstm_n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # a gate is simply a weighted linear transformation
        self.gate = nn.Linear(768 + bilstm_hidden_dim * 2, ffn_dim)

        self.ffn = FFN(ffn_dim, out_feat, dropout)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, data):
        emb = self.embeddings(data["tokens"])
        self.bilstm.flatten_parameters()

        padded_data = nn.utils.rnn.pack_padded_sequence(emb, data['mask'].sum(dim=-1).cpu(), batch_first=True, enforce_sorted=False)

        bilstmout, _ = self.bilstm(padded_data)
        bilstmout, _ = nn.utils.rnn.pad_packed_sequence(bilstmout, batch_first=True)
        bilstmout = self.dropout(bilstmout[:, 0, :])

        inputs = self.tokenizer(data["text"], padding=True, return_tensors="pt").to(self.device)
        bertout = self.bert(**inputs).pooler_output
        bertout = self.dropout(bertout)

        gatein = torch.cat((bilstmout, bertout), dim=-1)
        chosen = self.activation(self.dropout(self.gate(gatein)))

        res = self.ffn(chosen)
        return res

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument(
            "--angrybert-bilstm-hidden-size",
            type=int,
            default=64,
            help="BiLSTM hidden size",
        )
        group.add_argument(
            "--angrybert-bilstm-n-layers",
            type=int,
            default=2,
            help="Number of layers in the BiLSTM",
        )
        group.add_argument(
            "--angrybert-bilstm-bidirectional",
            type=bool,
            default=True,
            help="Train BiLSTM or LSTM",
        )
        group.add_argument(
            "--angrybert-ffn-in-dim",
            type=int,
            default=512,
            help="The input size of the FFN layer, which will also be the output size of the Gate",
        )

    @staticmethod
    def make_model(args):
        return AngryBERT(
            args.angrybert_bilstm_n_layers,
            args.angrybert_bilstm_hidden_size,
            args.angrybert_ffn_in_dim,
            args.out_feat,
            args.embeddings,
            args.dropout,
            args.device,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask", "text"]
