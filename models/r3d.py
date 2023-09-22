"""R3D"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.temporal_spatial_conv(x)
        return x



class SpatioTemporalDeConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super(SpatioTemporalDeConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        self.temporal_spatial_deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                        stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        x = self.temporal_spatial_deconv(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            if isinstance(downsample, bool):
                stride = 2
            else:
                stride = downsample
            # downsample with stride = 2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=stride)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)

class SpatioTemporalDeResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalDeConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, upsample=False):
        super(SpatioTemporalDeResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.upsample = upsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.upsample:
            if isinstance(upsample, bool):
                stride = 2
                output_padding = 1
            else:
                stride = upsample
                output_padding = [0,1,1]
            # downsample with stride = 2 the input x
            self.upsampleconv = SpatioTemporalDeConv(in_channels, out_channels, 1, output_padding=output_padding, stride=stride)
            self.upsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = SpatioTemporalDeConv(in_channels, out_channels, kernel_size, padding=padding, output_padding=output_padding, stride=stride)
        else:
            self.conv1 = SpatioTemporalDeConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalDeConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.upsample:
            x = self.upsamplebn(self.upsampleconv(x))

        return self.outrelu(x + res)




class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x

class SpatioTemporalDeResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalDeResBlock,
                 upsample=False):

        super(SpatioTemporalDeResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, upsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have upsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, ngf=64, block_type=SpatioTemporalResBlock, motion_dims=14, app_dims=13):
        super(R3DNet, self).__init__()

        block_type_de = SpatioTemporalDeResBlock
        self.motion_dims= motion_dims
        self.app_dims = app_dims

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(1, ngf, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.bn1 = nn.BatchNorm3d(ngf)
        self.relu1 = nn.ReLU()
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(ngf, ngf, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(ngf, ngf * 2, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(ngf * 2, ngf * 4, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(ngf * 4, ngf * 8, 3, layer_sizes[3], block_type=block_type, downsample=True)
        # self.conv6 = SpatioTemporalResLayer(ngf * 8, ngf * 8, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # self.deconv6 = SpatioTemporalDeResLayer(ngf * 8, ngf * 8, 3, layer_sizes[3], block_type=block_type_de, upsample=True)
        self.deconv5 = SpatioTemporalDeResLayer(ngf * 8, ngf * 4, 3, layer_sizes[3], block_type=block_type_de, upsample=True)
        self.deconv4 = SpatioTemporalDeResLayer(ngf * 4, ngf * 2, 3, layer_sizes[2], block_type=block_type_de, upsample=True)
        self.deconv3 = SpatioTemporalDeResLayer(ngf * 2, ngf, 3, layer_sizes[1], block_type=block_type_de, upsample=True)

        self.deconv2 = SpatioTemporalDeResLayer(ngf, ngf, 3, layer_sizes[0], block_type=block_type_de)
        # reproduce to raw size
        self.deconv1 = SpatioTemporalDeConv(ngf, 1, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3], output_padding=[0, 1, 1])
        self.bn2 = nn.BatchNorm3d(1)
        self.relu2 = nn.ReLU()

        # global average pooling of the output
        # self.pool = nn.AdaptiveAvgPool3d(1)
        # self.motion_linear = nn.Linear(512, self.motion_dims)
        # self.app_linear = nn.Linear(512, self.app_dims)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x_fea = x

        # x = self.deconv6(x)
        x = self.deconv5(x)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.relu2(self.bn2(self.deconv1(x)))

        # x = self.pool(x)
        # x = x.view(-1, 512)
        #
        # motion_out = self.motion_linear(x)
        # app_out = self.app_linear(x)

        return x, x_fea


if __name__ == '__main__':
    device = torch.device("cuda:0")
    r3d = R3DNet((1, 1, 1, 1)).to(device)
    input = torch.randn((1, 3, 16, 256, 256)).to(device)
    output, fea = r3d(input)
    print(output.shape, fea.shape)
