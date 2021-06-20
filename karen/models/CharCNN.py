import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("CharCNN")
class CharCNN(BaseModel):
    """
    CharCNN https://arxiv.org/abs/1509.01626
    """

    def __init__(
        self,
        embeddings,
        num_channels=256,
        linear_size=256,
        output_size=4,
        seq_len=165,
        dropout_keep=0.5,
    ):
        super(CharCNN, self).__init__()

        self.embeddings = embeddings

        self.convolutional_layers = []

        in_channels = [self.embeddings.weight.shape[-1]] + [num_channels] * 5
        out_channels = [num_channels] * 6
        kernel_sizes = [7] * 2 + [3] * 4
        maxpool_sizes = [3] * 2 + [-1] * 3 + [3]

        for i in range(6):
            self.convolutional_layers.append(
                self.init_conv(in_channels[i], out_channels[i], kernel_sizes[i], maxpool_sizes[i])
            )

        self.convolutional_layers = nn.Sequential(*self.convolutional_layers)

        conv_output_size = num_channels * ((seq_len - 96) // 27)

        if conv_output_size < 1:
            message = "Due to the number of convolutional layers in this model, this dataset with {} input features is not permissible.".format(
                seq_len
            )
            raise ValueError(message)

        self.linear_layers = nn.Sequential(
            nn.Linear(conv_output_size, linear_size),
            nn.ReLU(),
            nn.Dropout(dropout_keep),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Dropout(dropout_keep),
            nn.Linear(linear_size, output_size),
        )

    def init_conv(self, in_ch, out_ch, ks, mpks):
        if mpks > 0:
            return nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=mpks),
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks),
                nn.ReLU(),
            )

    def forward(self, x):
        x = x["tokens"]
        embedded_sent = self.embeddings(x).permute(0, 2, 1)  # shape=(batch_size,embed_size,seq_len)
        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        linear_output = self.linear_layers(conv_out)
        return linear_output

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--charcnn-n-channels", type=int, default=256, help="CharCNN number of channels")
        group.add_argument("--charcnn-linear-size", type=int, default=256, help="CharCNN size if linear layers")

    @staticmethod
    def make_model(args):
        return CharCNN(
            args.embeddings,
            args.charcnn_n_channels,
            args.charcnn_linear_size,
            args.out_feat,
            args.in_feat,
            args.dropout,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
