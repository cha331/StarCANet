import torch
import torch.nn as nn
from utils import DropPath, logger

__all__ = ["starcanet"]

IMG_W = 32  # The number of subcarriers (Nc)
IMG_H = 32  # The number of transmit antennas (Nt)
IMG_CHANNEL = 2  # Real, Imaginary
IMG_TOTAL_SIZE = IMG_W * IMG_H * IMG_CHANNEL


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dilation=1, groups=1, with_bn=True):
        super(ConvBN, self).__init__()
        bias = not with_bn
        self.add_module('conv',
                        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=bias))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))


class StarCA(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, 1, 0, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, 1, 0, with_bn=False)
        self.g = nn.Sequential(CoordAtt(mlp_ratio * dim, mlp_ratio * dim, 1),
                               ConvBN(mlp_ratio * dim, dim, 1, 1, 0, with_bn=True))
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        # self.ca = CoordAtt(dim, dim, 1)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        # x = self.act(x1) + x2
        x = self.dwconv2(self.g(x))
        # x = self.ca(x)
        x = input + self.drop_path(x)
        return x


class Encoder(nn.Module):
    def __init__(self, reduction=4, embed_dim=8, drop_path_rate=0.1):
        super(Encoder, self).__init__()
        self.mlp_ratio = 2
        self.depth = 2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth
        self.conv1 = nn.Sequential(ConvBN(IMG_CHANNEL, embed_dim, kernel_size=1, stride=1, padding=0), nn.ReLU6())
        self.star_layer1 = nn.Sequential(
            StarCA(embed_dim, self.mlp_ratio, dpr[0]),
            StarCA(embed_dim, self.mlp_ratio, dpr[1]),
        )
        self.norm_layer = nn.BatchNorm2d(embed_dim)
        self.conv2 = ConvBN(embed_dim, IMG_CHANNEL, kernel_size=1, stride=1, padding=0)
        self.en_fc = nn.Linear(IMG_TOTAL_SIZE, IMG_TOTAL_SIZE // reduction)

    def forward(self, x):
        n, c, h, w = x.detach().size()

        out = self.conv1(x)
        out = self.star_layer1(out)
        out = self.norm_layer(out)
        out = self.conv2(out)

        out = out.view(n, -1)
        out = self.en_fc(out)

        return out


class Decoder(nn.Module):
    def __init__(self, reduction=4, embed_dim=8, drop_path_rate=0.1):
        super(Decoder, self).__init__()
        self.mlp_ratio = 2
        self.depth = 4

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth
        self.de_fc = nn.Linear(IMG_TOTAL_SIZE // reduction, IMG_TOTAL_SIZE)
        self.conv1 = nn.Sequential(ConvBN(IMG_CHANNEL, embed_dim, kernel_size=1, stride=1, padding=0), nn.ReLU6())
        self.star_layer1 = nn.Sequential(
            StarCA(embed_dim, self.mlp_ratio, dpr[0]),
            StarCA(embed_dim, self.mlp_ratio, dpr[1]),
            StarCA(embed_dim, self.mlp_ratio, dpr[2]),
            StarCA(embed_dim, self.mlp_ratio, dpr[3]),
        )
        self.norm_layer = nn.BatchNorm2d(embed_dim)
        self.conv2 = ConvBN(embed_dim, IMG_CHANNEL, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        out = self.de_fc(x)
        out = out.view(-1, IMG_CHANNEL, IMG_H, IMG_W)

        out = self.conv1(out)
        out = self.star_layer1(out)
        out = self.norm_layer(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return out


def _init_weights(m):
    if isinstance(m, nn.Linear or nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class StarCANet(nn.Module):
    """
    The main StarCANet model, composed of an Encoder and a Decoder.
    """

    def __init__(self, reduction=4, embed_dim=8, drop_path_rate=0.1):
        super(StarCANet, self).__init__()

        self.encoder = Encoder(reduction, embed_dim, drop_path_rate)
        self.decoder = Decoder(reduction, embed_dim, drop_path_rate)
        self.apply(_init_weights)

    def forward(self, x):
        codeword = self.encoder(x)
        out = self.decoder(codeword)
        return out


def starcanet(reduction=4, model_size='M'):
    r""" Create a StarCANet model with specified size and compression ratio.

    Args:
        reduction (int): The reciprocal of the compression ratio (e.g., 4 for CR=1/4).
        model_size (str): The size of the model, one of 'S', 'M', 'L'.
                          'S': Small model with embed_dim=4
                          'M': Medium model with embed_dim=8
                          'L': Large model with embed_dim=16

    Returns:
        An instance of the StarCANet model.
    """
    model_size = model_size.upper()
    size_to_embed_dim = {
        'S': 4,
        'M': 8,
        'L': 16,
    }

    if model_size not in size_to_embed_dim:
        raise ValueError(f"Unknown model_size '{model_size}'. Please choose from 'S', 'M', 'L'.")

    embed_dim = size_to_embed_dim[model_size]
    drop_path_rate = 0.1

    logger.info(f"Creating StarCANet model. Size: {model_size}, Reduction: {reduction}")

    model = StarCANet(reduction=reduction, embed_dim=embed_dim, drop_path_rate=drop_path_rate)
    return model
