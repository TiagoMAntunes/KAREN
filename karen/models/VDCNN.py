import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..register_model import RegisterModel


class KMaxPool(nn.Module):
    def __init__(self, k="half"):
        super(KMaxPool, self).__init__()
        self.k = k

    def forward(self, x):
        if self.k == "half":
            time_steps = x.shape[2]
            self.k = time_steps // 2
        kmax, kargmax = x.topk(self.k, dim=2)
        return kmax


def ConvolutionalBlock(in_channels, out_channels, kernel_size=3, first_stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=first_stride, padding=kernel_size // 2),
        nn.BatchNorm1d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(num_features=out_channels),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, downsample_type="resnet", optional_shortcut=True):
        super(ResidualBlock, self).__init__()
        self.optional_shortcut = optional_shortcut
        self.downsample = downsample

        if self.downsample:
            if downsample_type == "resnet":
                self.pool = None
                first_stride = 2
            elif downsample_type == "kmaxpool":
                self.pool = KMaxPool(k="half")
                first_stride = 1
            elif downsample_type == "vgg":
                self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                first_stride = 1
            else:
                raise NotImplementedError()
        else:
            first_stride = 1

        self.convolutional_block = ConvolutionalBlock(in_channels, out_channels, first_stride=first_stride)

        if self.optional_shortcut and self.downsample:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):

        residual = x
        if self.downsample and self.pool:
            x = self.pool(x)
        x = self.convolutional_block(x)

        if self.optional_shortcut and self.downsample:
            residual = self.shortcut(residual)

        if self.optional_shortcut:
            x = x + residual

        return x


@RegisterModel("VDCNN")
class VDCNN(BaseModel):
    """
    Very Deep Convolutional Networks for Text Classification https://arxiv.org/pdf/1606.01781.pdf
    """

    def __init__(
        self,
        n_classes,
        embeddings,
        dropout,
        depth=9,
        optional_shortcut=True,
        kmax=8,
        downsample_type="resnet",
    ):
        super(VDCNN, self).__init__()
        self.kmax = kmax

        if depth == 9:
            n_conv_layers = {
                "conv_block_64": 2,
                "conv_block_128": 2,
                "conv_block_256": 2,
                "conv_block_512": 2,
            }
        elif depth == 17:
            n_conv_layers = {
                "conv_block_64": 2,
                "conv_block_128": 2,
                "conv_block_256": 2,
                "conv_block_512": 2,
            }
        elif depth == 29:
            n_conv_layers = {
                "conv_block_64": 10,
                "conv_block_128": 10,
                "conv_block_256": 4,
                "conv_block_512": 4,
            }
        elif depth == 49:
            n_conv_layers = {
                "conv_block_64": 16,
                "conv_block_128": 16,
                "conv_block_256": 10,
                "conv_block_512": 6,
            }
        else:
            message = "Invalid model depth parameter: {depth}. Please use one of the following values: 9, 17, 29, 49"
            raise ValueError(message)

        self.embedding = embeddings
        embed_size = self.embedding.weight.shape[-1]

        conv_layers = []
        conv_layers.append(nn.Conv1d(embed_size, 64, kernel_size=3, padding=1))

        for i in range(n_conv_layers["conv_block_64"]):
            conv_layers.append(ResidualBlock(64, 64, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers["conv_block_128"]):
            if i == 0:
                conv_layers.append(
                    ResidualBlock(
                        in_channels=64,
                        out_channels=128,
                        downsample=True,
                        downsample_type=downsample_type,
                        optional_shortcut=optional_shortcut,
                    )
                )
            conv_layers.append(ResidualBlock(128, 128, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers["conv_block_256"]):
            if i == 0:
                conv_layers.append(
                    ResidualBlock(
                        in_channels=128,
                        out_channels=256,
                        downsample=True,
                        downsample_type=downsample_type,
                        optional_shortcut=optional_shortcut,
                    )
                )
            conv_layers.append(ResidualBlock(256, 256, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers["conv_block_512"]):
            if i == 0:
                conv_layers.append(
                    ResidualBlock(
                        in_channels=256,
                        out_channels=512,
                        downsample=True,
                        downsample_type=downsample_type,
                        optional_shortcut=optional_shortcut,
                    )
                )
            conv_layers.append(ResidualBlock(512, 512, optional_shortcut=optional_shortcut))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.kmax_pooling = KMaxPool(k=self.kmax)

        # original implementation used three linear layers without dropout or relu. I found that
        # one linear layer was better except for when depth=47. in this case, the three layers
        # with dropout and relu gave the best performance.
        # The result was 0.5995587424158852 and the output table at the bottom of this file.

        # self.linear_layers = nn.Sequential(
        #     nn.Linear(512 * kmax, 2048),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        #     nn.Linear(2048, 2048),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        #     nn.Linear(2048, n_classes),
        # )

        self.linear_layers = nn.Linear(512 * self.kmax, n_classes)

    def forward(self, sentences):
        sentences = sentences["tokens"]
        x = self.embedding(sentences).permute(0, 2, 1)
        x = self.conv_layers(x)
        if x.shape[-1] < self.kmax:
            message = "Due to the number of convolutional layers in this model, this dataset with {} input features is not permissible.".format(
                sentences.shape[-1]
            )
            raise ValueError(message)
        x = self.kmax_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--vdcnn-depth", type=int, default=17, help="VDCNN depth. Options: 9, 17, 29, 49")
        group.add_argument("--vdcnn-no-shortcut", action="store_true", help="Optional shortcut not used if used")
        group.add_argument("--vdcnn-kmax", type=int, default=8, help="KMaxPool value")
        group.add_argument(
            "--vdcnn-downsample-type",
            type=str,
            default="resnet",
            help="Type of downsampling to use. Options: resnet, vgg, kmaxpool",
        )

    @staticmethod
    def make_model(args):
        return VDCNN(
            args.out_feat,
            args.embeddings,
            args.dropout,
            args.vdcnn_depth,
            not args.vdcnn_no_shortcut,
            args.vdcnn_kmax,
            args.vdcnn_downsample_type,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]


# Test accuracy: 0.5995587424158852
# !python3 run.py --model vdcnn --dataset hatexplain --embeddings twitterglove --embedding-dim 200 --batch-size 64 --max-epoch 15 --scheduler-step 7 --vdcnn-depth 49
# Label name      Precision    Recall        F1    Counts
# ------------  -----------  --------  --------  --------
# hatespeech       0.633127  0.76306   0.692047       536
# offensive        0.534247  0.377176  0.442177       517
# undecided        0         0         0               89
# normal           0.602244  0.719821  0.655804       671

# | VDCNN + Glove 9 | 0.587 | 0.666 | 0.681 | 0.673 |
# | VDCNN + Glove 17 | 0.598 | 0.655 | 0.709 | 0.681 |
# | VDCNN + Glove 29 | 0.581 | 0.664 | 0.655 | 0.659 |
# | VDCNN + Glove 49 | 0.588 | 0.678 | 0.654 | 0.666 |
