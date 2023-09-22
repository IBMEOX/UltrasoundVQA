import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import net
import numpy as np
import imageio
from utils import ssim, util, gazemetrics
import matplotlib.pyplot as plt
import os


class CycleGANFlowModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_C', type=float, default=10, help='weight for ssim loss for A')
            parser.add_argument('--lambda_H', type=float, default=0, help='weight for gaze loss for A')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        ## ============================== Names for saving and visualization ========================= ##
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        if opt.isTrain:
            self.loss_names = ['G', 'cycle_A', 'cycle_B', 'ssim_A']

            if opt.lambda_H > 0:
                self.loss_names += ['gaze']

        else:
            self.loss_names = ['cycle_A']
        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'ad_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'ad_B']

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'G_C']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'G_C']

        ## ============================== Model stucture ========= ========================= ##
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # AD: G_A (encoder), G_B (decoder), D_A (z_dis), D_B (img_dis)

        self.netG_A = net.define_G(opt.flow_channel+1, opt.nz, opt.ngf, opt.netG_A, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = net.define_G(opt.nz, opt.output_nc, opt.ngf, opt.netG_B, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_C = net.define_G(opt.nz, opt.output_nc, opt.ngf, opt.netG_B, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_D = net.define_G(opt.nz, opt.output_nc, opt.ngf, opt.netG_B, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define discriminators
        if self.isTrain | True:

            self.netD_A = net.define_D(opt.nz, opt.ndf, 'z',
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = net.define_D(opt.input_nc, opt.ndf, 'img',
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.real_B_pool = ImagePool(opt.pool_size_real)

            ## ==================================== loss function ======================================= ##
            # define loss functions
            self.criterionGAN = net.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionCycle_B = torch.nn.L1Loss()

            self.criterionSSIM = ssim.SSIM()
            self.criterionGaze = torch.nn.MSELoss()

            ## ==================================== optimizer settings =================================== ##
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            G_paremters = itertools.chain(
                self.netG_A.parameters(),
                self.netG_B.parameters(),
                self.netG_C.parameters(),
                self.netG_D.parameters()
            )
            self.optimizer_G = torch.optim.Adam(G_paremters, lr=opt.lr, betas=(opt.beta1, 0.999))

            D_parameters = itertools.chain(
                self.netD_A.parameters(),
                self.netD_B.parameters()
            )
            self.optimizer_D = torch.optim.Adam(D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
        self.raw_C = input['C_raw'].to(self.device)
        self.flow_A = input['flow'].to(self.device)


    def set_input_val(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # AtoB = self.opt.direction == 'BtoA'
        self.real_A = input['A'].unsqueeze(0).to(self.device)
        self.real_B = input['B'].unsqueeze(0).to(self.device)
        self.image_paths = input['A_paths']
        self.flow_A = input['flow'].unsqueeze(0).to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # a = self.criterionGAN(self.real_A, True)
        # add flow
        self.real_A_input = torch.cat( (self.real_A, self.flow_A), 1)

        self.fake_B = self.netG_A(self.real_A_input)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        #add flow
        self.fake_A_input = torch.cat( (self.fake_A, self.flow_A), 1)

        self.rec_B = self.netG_A(self.fake_A_input)   # G_A(G_B(B))

        # add flow
        self.rec_A_input = torch.cat( (self.rec_A, self.flow_A), 1)

        self.ad_B = self.netG_A(self.rec_A_input)
        self.ad_A = self.netG_B(self.rec_B)
        self.gaze_pred = self.netG_C(self.fake_B)

        1


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        # print(self.loss_D_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""

        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        lambda_H = self.opt.lambda_H

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle_B(self.rec_B, self.real_B) * lambda_B


        # squeeze the channel (1) for ssim per image caculation
        self.loss_ssim_A = (1 - self.criterionSSIM(self.rec_A.squeeze(1), self.real_A.squeeze(1)) ) * lambda_C

        """
        rec_tem = self.rec_A.squeeze()
        real_tem = self.real_A.squeeze()
        score = 0
        for i in range(8):
            rec_img_tem = rec_tem[:,i,:,:].unsqueeze(1)
            real_img_tem = real_tem[:,i,:,:].unsqueeze(1)
            score += (1-self.criterionSSIM(rec_img_tem,real_img_tem))
        """

        # gaze loss
        if lambda_H > 0:
            self.loss_gaze = self.criterionGaze(self.gaze_pred, self.raw_C) * lambda_H
            # ((self.raw_C+1.)/2.>0).type(torch.float)
        else:
            self.loss_gaze = 0

        # combined loss and calculate gradients
        self.basic = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B

        self.loss_G =  self.basic + self.loss_ssim_A + self.loss_gaze

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # by doing so, weight_cent would not impact on the learning of centers


        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def calc_loss(self):

        self.criterionCycle = torch.nn.L1Loss()
        loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)

        return loss_cycle_A

    def compute_visual_result(self, iters=0, visuals=None, flag=False):
        if not visuals:
            visuals = self.get_current_visuals()
            flag = True

        with torch.no_grad():
            real_A = visuals['real_A']
            rec_A = visuals['rec_A']

            real_A_array_all, _ = util.seqTensor2im(real_A)
            rec_A_array_all, _ = util.seqTensor2im(rec_A)

            real_A_array, rec_A_array = real_A_array_all[-4:], rec_A_array_all[-4:]

            batch = True
            if batch:
                score, img4 = ssim.ssim(visuals['real_A'].squeeze(1), visuals['rec_A'].squeeze(1), map=True)
            else:
                score = 0
                aa = visuals['real_A'].permute(0,2,1,3,4)
                bb = visuals['rec_A'].permute(0,2,1,3,4)
                for iii in range(8):
                    score_, img4 = ssim.ssim(aa[0,iii:iii+1,:,:,:], bb[0,iii:iii+1,:,:,:], map=True)
                    score += score_
                score = score / real_A.size(0)

            img = np.concatenate((real_A_array, rec_A_array), 0)

            if flag:
                img_path = self.opt.checkpoints_dir + '/' + self.opt.name + '/images'
                import os
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                imageio.imsave(img_path + '/' + str(iters) + '.png', np.uint8(self.merge(img, [2, 4])*255))
                1
            else:
                return img, score.cpu().data.numpy()

    def revert_img(self, img):
        return (img + 1.) / 2.

    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image

        return img

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
