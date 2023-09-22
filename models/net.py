import torch
import torch.nn as nn
import functools
from .net_func import get_norm_layer, init_net
from .r3d import SpatioTemporalResLayer, SpatioTemporalResBlock, SpatioTemporalConv,\
                SpatioTemporalDeResLayer, SpatioTemporalDeResBlock, SpatioTemporalDeConv,\
                R3DNet
import matplotlib.pyplot as plt


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'encoder':
        net = Encoder(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG == 'decoder':
        net = Decoder(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG == 'encoder_ts':
        net = Encoder_ts(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG == 'decoder_ts':
        net = Decoder_ts(input_nc, output_nc, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)



def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator_2x2(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'img':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'z':
        net = MLP(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'map':
        net = DisMAP(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y


##############################################################################
# Model structure Class
##############################################################################

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1]), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=[kw,kw,kw], stride=2, padding=[1,1,1], bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=[1,kw,kw], stride=1, padding=[0,1,1])]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class NLayerDiscriminator_2x2(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1]), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=[1,kw,kw], stride=1, padding=[0,1,1], bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=[1,kw,kw], stride=1, padding=[0,1,1])]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class NLayerDiscriminator_img(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=[1,kw,kw], stride=[1,2,2], padding=[0,padw,padw]), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=[1,kw,kw], stride=[1,2,2], padding=[0,padw,padw], bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, ndf * nf_mult, kernel_size=[kw,1,1], stride=[2,1,1], padding=[padw, 0, 0], bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * nf_mult, ndf * nf_mult, kernel_size=[kw,1,1], stride=[2,1,1], padding=[padw, 0, 0], bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * nf_mult, ndf * nf_mult, kernel_size=[kw,1,1], stride=[2,1,1], padding=[padw, 0, 0], bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=[1,kw,kw], stride=1, padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=[1,kw,kw], stride=1, padding=[0,1,1])]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Encoder_ts(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, output_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(Encoder_ts, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.model = [
            # 256
            nn.Conv3d(input_nc, ndf, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.LeakyReLU(0.2, True),
            # 128
            nn.Conv3d(ndf, ndf*2, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0], bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 64
            nn.Conv3d(ndf*2, ndf*4, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1], bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 32
            nn.Conv3d(ndf*4, ndf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.Conv3d(ndf*8, ndf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf*8),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 2
            nn.Conv3d(ndf*8, output_nc, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1])
            # 1
        ]

        self.net = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        output = self.net(input)
        # out = output.view(-1)
        return output


class Decoder_ts(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d):
        super(Decoder_ts, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            # 256
            nn.ConvTranspose3d(input_nc, ngf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1]),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True),
            # 128
            nn.ConvTranspose3d(ngf*8, ngf * 8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 64
            nn.ConvTranspose3d(ngf * 8, ngf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 32
            nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.ConvTranspose3d(ngf*8, ngf*4, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf*4),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1], bias=use_bias),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0], bias=use_bias),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True),
        ]
        self.output1 = [
            # 2
            nn.ConvTranspose3d(ngf, output_nc, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0], bias=use_bias),
            nn.Tanh(),
            # 1
        ]
        # self.output2 = [
        #     nn.ConvTranspose3d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
        #     nn.Tanh()
        # ]
        self.net = nn.Sequential(*self.net)
        self.output1 = nn.Sequential(*self.output1)
        # self.output2 = nn.Sequential(*self.output2)

    def forward(self, input):

        middle = self.net(input)
        output1 = self.output1(middle)
        # output2 = self.output2(middle)
        # output = {'output1':output1, 'output2':output2}
        return output1




class Encoder(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, output_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # self.channel_att1 = TAM()
        # self.channel_att2 = TAM()
        self.model = [
            # 256
            nn.Conv3d(input_nc, ndf, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1]),
            nn.LeakyReLU(0.2, True),
            # 128
            nn.Conv3d(ndf, ndf * 2, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 64
            nn.Conv3d(ndf * 2, ndf*4, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # 32
            nn.Conv3d(ndf*4, ndf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.Conv3d(ndf*8, ndf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ndf*8),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1], bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.Conv3d(ndf * 8, ndf * 8, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1], bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 2
            nn.Conv3d(ndf*8, output_nc, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1])
            # 1
        ]

        self.net = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        output = self.net(input)
        # out = output.view(-1)
        return output


class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d):
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            # 256
            nn.ConvTranspose3d(input_nc, ngf*8, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1]),
            norm_layer(ngf*8),
            nn.LeakyReLU(0.2, True),
            # 128
            nn.ConvTranspose3d(ngf*8, ngf * 8, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1], bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 64
            nn.ConvTranspose3d(ngf * 8, ngf*8, kernel_size=[4,4,4], stride=[2,2,2], padding=[1,1,1], bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 32
            nn.ConvTranspose3d(ngf*8, ngf*8, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.ConvTranspose3d(ngf*8, ngf*4, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf*4),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1], bias=use_bias),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True),
        ]
        self.output1 =[
            # 2
            nn.ConvTranspose3d(ngf, output_nc, kernel_size=[1,4,4], stride=[1,2,2], padding=[0,1,1]),
            nn.Tanh()
            # 1
        ]
        # self.output2 = [
        #     nn.ConvTranspose3d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
        #     nn.Tanh()
        # ]
        self.net = nn.Sequential(*self.net)
        self.output1 = nn.Sequential(*self.output1)
        # self.output2 = nn.Sequential(*self.output2)

    def forward(self, input):

        middle = self.net(input)
        output1 = self.output1(middle)
        # output2 = self.output2(middle)
        # output = {'output1':output1, 'output2':output2}
        return output1



class MLP(nn.Module):
    def __init__(self, input_nc, ndf, norm_layer=None):
        super(MLP, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ##### can use residual (refer to bigbigan -- tf2.0)

        self.net = [
            # 128 ndf
            # nn.Conv2d(input_nc, ndf, 1, 1),
            nn.utils.spectral_norm(nn.Linear(input_nc, ndf)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 64
            nn.utils.spectral_norm(nn.Linear(ndf, ndf//2)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 32
            nn.utils.spectral_norm(nn.Linear(ndf//2, ndf // 4)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.utils.spectral_norm(nn.Linear(ndf//4, ndf // 8)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.utils.spectral_norm(nn.Linear(ndf//8, ndf //16)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.utils.spectral_norm(nn.Linear(ndf//16, 1)),
            nn.Sigmoid()
            # 1
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        input_vec = input.squeeze()
        # input_vec = input
        out = self.net(input_vec)
        return out


class DisMAP(nn.Module):
    def __init__(self, input_nc, ndf, norm_layer=None):
        super(DisMAP, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d


        self.net = [
            # 128 ndf
            nn.Conv3d(input_nc, input_nc, kernel_size=[1,3,3], stride=[1,2,2], padding=1, bias=use_bias),
            norm_layer(input_nc),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 64
            nn.Conv3d(input_nc, input_nc, kernel_size=[1,3,3], stride=[1,2,2], padding=1, bias=use_bias),
            norm_layer(input_nc),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 32
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(input_nc*4*4, input_nc//2)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.utils.spectral_norm(nn.Linear(input_nc//2, input_nc//4)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.utils.spectral_norm(nn.Linear(input_nc//4, input_nc //8)),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.utils.spectral_norm(nn.Linear(input_nc //8, 1)),
            nn.Sigmoid()
            # 1
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        input_vec = input.squeeze(3).squeeze(2)
        out = self.net(input_vec)
        return out


class SpatialTemporalAutoEncoder(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, output_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(SpatialTemporalAutoEncoder, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model = [
            # 256
            TimeDistributed(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            TimeDistributed(nn.LeakyReLU(0.2, True)),
            # 128
            TimeDistributed(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            TimeDistributed(norm_layer(ndf * 2)),
            TimeDistributed(nn.LeakyReLU(0.2, True)),
            # 64
            TimeDistributed(nn.Conv2d(ndf * 2, ndf*4, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            TimeDistributed(norm_layer(ndf * 4)),
            TimeDistributed(nn.LeakyReLU(0.2, True)),
            # 32
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 16
            nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf*8),
            nn.LeakyReLU(0.2, True),
            # 8
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 4
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 2
            nn.Conv2d(ndf*8, output_nc, kernel_size=4, stride=2, padding=1)
            # 1
        ]

        self.net = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        output = self.net(input)
        # out = output.view(-1)
        return output




