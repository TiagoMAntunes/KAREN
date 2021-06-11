import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..register_model import RegisterModel
from math import log2


def convnorm(in_ch, out_ch, ks):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
        nn.BatchNorm1d(out_ch),
        nn.LeakyReLU(0.1),
    )


@RegisterModel("UNet")
class UNet(BaseModel):
    def __init__(
        self,
        embeddings,
        in_feat,
        out_feat,
        unet_depth,
        channel_jump,
        encoding_ks,
        decoding_ks,
        use_linear,
    ):
        super(UNet, self).__init__()
        self.embeddings = embeddings
        channels_in = self.embeddings.weight.shape[-1]

        self.init_linear_converters(unet_depth, in_feat, out_feat, use_linear)
        self.init_channel_sizes(channel_jump, unet_depth, channels_in)

        self.depth = unet_depth

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(self.depth):
            self.encoder.append(convnorm(self.echannelin[i], self.echannelout[i], encoding_ks))
            self.decoder.append(convnorm(self.dchannelin[i], self.dchannelout[i], decoding_ks))

        self.middle = convnorm(self.echannelout[-1], self.echannelout[-1], encoding_ks)

        self.out = nn.Sequential(
            nn.Conv1d(channels_in + (channels_in // channel_jump + 1) * channel_jump, 1, 1),
            nn.Tanh(),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.downsample = nn.MaxPool1d(2)

    def init_linear_converters(self, unet_depth, in_feat, out_feat, ul):
        self.linear_size = max(int(2 ** unet_depth), int(2 ** int(log2(in_feat))))
        self.conversion_lin_in = nn.Linear(in_feat, self.linear_size)
        self.resize_image = nn.Upsample(size=self.linear_size, mode="linear", align_corners=False)
        self.conversion_lin_out = nn.Linear(self.linear_size, out_feat)
        print(
            "Unet needs to convert the sentence length to max({}, {}) in order to work correctly.\nResize performed using {}.\n".format(
                int(2 ** unet_depth), int(2 ** int(log2(in_feat))), "linear layer" if ul else "linear upsampling"
            )
        )

    def init_channel_sizes(self, channel_jump, unet_depth, channels_in):

        strt = channels_in // channel_jump
        fnsh = unet_depth + channels_in // channel_jump

        self.echannelout = [(i + 1) * channel_jump for i in range(strt, fnsh)]
        self.echannelin = [channels_in] + self.echannelout[:-1]

        self.dchannelout = self.echannelout[::-1]
        self.dchannelin = [self.dchannelout[0] * 2] + [(2 * i - 1) * channel_jump for i in range(fnsh, strt, -1)]

        print("Encoding convolutions:")
        for i, p in enumerate(zip(self.echannelin, self.echannelout)):
            print("Layer: {}, Conv Channels: {}".format(i, p))

        print("Middle layer has channels {}".format((self.echannelout[-1], self.echannelout[-1])))

        print()
        print("Decoding convolutions:")
        for i, p in enumerate(zip(self.dchannelin, self.dchannelout)):
            print(
                "Layer: {}, Conv Channels: ({} + {} = {}, {})".format(
                    len(self.dchannelin) - i - 2, p[0] - p[1], p[1], p[0], p[1]
                )
            )
        print()

    def forward(self, x):
        x = x["tokens"]
        x = self.embeddings(x).permute(0, 2, 1)
        x = self.resize_image(x)
        # x = self.conversion_lin_in(x)

        encoder = list()
        input = x

        for i in range(self.depth):
            x = self.encoder[i](x)
            encoder.append(x)
            x = self.downsample(x)

        x = self.middle(x)
        encoder.reverse()

        for i in range(self.depth):
            x = self.upsample(x)
            x = torch.cat([x, encoder[i]], dim=1)
            x = self.decoder[i](x)
        x = torch.cat([x, input], dim=1)

        x = self.out(x).squeeze()
        x = self.conversion_lin_out(x)
        return x

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--unet-depth", type=int, default=6, help="Depth of UNet")
        group.add_argument(
            "--unet-channel-jump",
            type=int,
            default=32,
            help="Convolutional channels increase by this amount when encoding",
        )
        group.add_argument("--unet-encoding-ks", type=int, default=9, help="Encoding kernel size")
        group.add_argument("--unet-decoding-ks", type=int, default=9, help="Decoding kernel size")
        group.add_argument(
            "--unet-use-linear",
            type=bool,
            default=False,
            help="Use neural network for image resizing (True) or use simple upsampling",
        )

    @staticmethod
    def make_model(args):
        return UNet(
            args.embeddings,
            args.in_feat,
            args.out_feat,
            args.unet_depth,
            args.unet_channel_jump,
            args.unet_encoding_ks,
            args.unet_decoding_ks,
            args.unet_use_linear,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
