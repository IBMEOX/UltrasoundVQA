import copy
import os.path
from dataFile.base_dataset import BaseDataset, get_transform
from dataFile.image_folder import make_dataset, make_dataset_seq
from PIL import Image, ImageOps
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

class clipGazeDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        if opt.isTrain:
            self.dir_A = os.path.join(opt.dataroot, 'train')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, 'train_gaze')  # create a path '/path/to/data/trainB'
        else:
            self.dir_A = os.path.join(opt.dataroot, 'test')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, 'test')  # create a path '/path/to/data/trainB'
        
        #import pdb;pdb.set_trace()

        self.A_paths = sorted(make_dataset_seq(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset_seq(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A, self.transform_B = self.transform_func(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        
        B_path = self.B_paths[index % self.B_size]
        #import pdb;pdb.set_trace()
        A_path = '/'.join(B_path.split('/')[0:3]+[B_path.split('/')[3].split('_')[0]]+[B_path.split('/')[-1]])
        flow_path = '/'.join(B_path.split('/')[0:3]+[B_path.split('/')[3].split('_')[0]+ '_color']+[B_path.split('/')[-1]])

        index_S = random.randint(0, self.B_size - 1)
        B_path_self = self.B_paths[index_S]
        S_path = '/'.join(B_path_self.split('/')[0:3]+[B_path_self.split('/')[3].split('_')[0]]+[B_path_self.split('/')[-1]])


        PATH = {'video': A_path, 'gaze': B_path, 'flow': flow_path, 'self': S_path}

        # apply image transformation
        out = self.read_clip(PATH, self.transform_A, self.opt.num_frames, self.opt.sample_rate)

        A = out['video']
        B = out['gaze']
        flow = out['flow']



        C = torch.normal(mean=0,std=1, size=(self.opt.nz, 1,1,1))


        A_label = A_path.split('/')[-1].split('-')[0]
        if A_label == 'HCP':
            A_label = 0
        elif A_label == 'Cereb':
            A_label = 1
        elif A_label == 'AC':
            A_label = 3
        elif A_label == 'NoHCP':
            A_label = 2
        else:
            A_label = 0

        return {'A': A, 'B': C, 'C_raw': B,
                'flow': flow,
                'A_paths': A_path, 'B_paths': B_path,  'A_label': A_label}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(self.A_size, self.B_size)

    def read_clip(self, PATH, transforms, num_frames, sample_rate):
        A_path = PATH['video']
        B_path = PATH['gaze']
        C_path = PATH['flow']

        seq_num = list(range(120, -1, -sample_rate))
        seq_ori = seq_num[:num_frames]

        if num_frames> len(seq_ori):
            seq_out = seq_ori + list(range(seq_ori[-1]))[::-1]
            seq_out = seq_out + [seq_out[-1]]*(num_frames-len(seq_out))
        else:
            seq_out = seq_ori

        seq_out.reverse()

        video_clip = []
        gaze_clip = []
        flow_clip = []
        aug_param = None

        for frame in seq_out:
            cur_img_path = A_path + '/clip' + str(frame).zfill(3) + '.png'
            img = Image.open(cur_img_path)#.convert('RGB')
            if img is None:
                print(cur_img_path)
            if aug_param is None:
                img_dict = transforms(image=np.array(img))
                aug_param = img_dict['replay']
            else:
                img_dict = transforms.replay(aug_param, image=np.array(img))

            video_clip.append(img_dict['image'])

        for frame in seq_out:
            cur_gaze_path = B_path + '/clip' + str(frame).zfill(3) + '.png'
            gaze = Image.open(cur_gaze_path)#.convert('RGB')
            if gaze is None:
                print(cur_gaze_path)
            gaze_dict = transforms.replay(aug_param, image=np.array(gaze))

            gaze_clip.append(gaze_dict['image'])

        for frame in seq_out:
            cur_flow_path = C_path + '/clip' + str(frame).zfill(3) + '.png'
            flow = Image.open(cur_flow_path)#.convert('RGB')
            if self.opt.flow_channel == 1:
                flow = ImageOps.grayscale(flow)
            if flow is None:
                print(cur_flow_path)
            flow_dict = transforms.replay(aug_param, image=np.array(flow))

            flow_clip.append(flow_dict['image'])


        video_clip = torch.stack(video_clip)
        video_clip = video_clip.permute( (1,0,2,3))

        gaze_clip = torch.stack(gaze_clip)
        gaze_clip = gaze_clip.permute( (1,0,2,3))

        flow_clip = torch.stack(flow_clip)
        flow_clip = flow_clip.permute( (1,0,2,3))

        video_gaze_clip = torch.cat((video_clip,gaze_clip),dim=0)

        output = {'video': video_clip,
                  'gaze': gaze_clip, 'video_gaze': video_gaze_clip,
                  'flow': flow_clip}

        return output


    def transform_func(self, opt):

        def divide255(image, **kwargs):
            image = image / 255.0
            return image.astype('float32')

        def toGray(image, **params):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray

        if opt.isTrain:
            # transform_A = A.ReplayCompose([
            #     # A.ToGray(always_apply=True),
            #     A.Resize(256,256),
            #     # A.Lambda(name='togray',image=toGray),
            #     A.HorizontalFlip(p=0.5),
            #     # A.VerticalFlip(p=0.5),
            #     # A.GaussNoise(p=0.2),
            #     # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            #     # A.RandomBrightnessContrast(p=0.5),
            #     # A.Lambda(image=divide255),
            #     A.Normalize((0.5,),(0.5,), always_apply=True),
            #     ToTensorV2()
            #     # A.RandomCrop(200,200, always_apply=True),
            # ])

            transform_list = []
            transform_list.append(A.Resize(256,256))

            if not opt.A_no_aug:

                if not opt.A_no_flip:
                    transform_list.append(A.HorizontalFlip(p=0.5))

                if not opt.A_no_shift_rotate_scale:
                    transform_list.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT))

                if not opt.A_no_contrast:
                    transform_list.append(A.RandomBrightnessContrast(p=0.5))

            transform_list += [A.Normalize((0.5,),(0.5,), always_apply=True),ToTensorV2()]

            transform_A = A.ReplayCompose(transform_list)


        else:
            transform_A = A.ReplayCompose([
                # A.ToGray(always_apply=True),
                A.Resize(256,256),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.GaussNoise(p=0.2),
                # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                # A.RandomBrightnessContrast(p=0.5),
                # A.RandomCrop(200,200, always_apply=True),
                A.Normalize((0.5,),(0.5,), always_apply=True),
                ToTensorV2()
            ])

        return transform_A, transform_A



