"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_seq(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, folders, _ in sorted(os.walk(dir)):
        for folder in folders:
            for subroot, _, fnames in sorted(os.walk(os.path.join(root,folder))):
                path = subroot
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_seq_dataset(dir, sample_rate, num_frames):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    seq_num = list(range(120, -1, -sample_rate))
    seq_ori = seq_num[:num_frames]

    if num_frames> len(seq_ori):
        seq_out = seq_ori + list(range(seq_ori[-1]))[::-1]
        seq_out = seq_out + [seq_out[-1]]*(num_frames-len(seq_out))
    else:
        seq_out = seq_ori

    seq_list = os.listdir(dir)
    seq_out.reverse()

    return seq_list, seq_out



def read_seq_dataset(file_name, sample_rate, num_frames):

    seq_num = list(range(120, -1, -sample_rate))
    seq_ori = seq_num[:num_frames]

    if num_frames> len(seq_ori):
        seq_out = seq_ori + list(range(seq_ori[-1]))[::-1]
        seq_out = seq_out + [seq_out[-1]]*(num_frames-len(seq_out))
    else:
        seq_out = seq_ori

    seq_out.reverse()

    seq_list = []

    if len(file_name) == 1:
        with open(file_name[0], 'r') as f:
            my_data = f.readlines()
            for line in my_data:
                line_data = line.split()[0]

                seq_list.append(os.path.join(*file_name[0].split('/')[:-1], line_data))
    elif len(file_name) > 1:
        for cls_name in file_name:
            with open(cls_name + '/tstList.txt', 'r') as f:
                my_data = f.readlines()
                for line in my_data:
                    line_data = line.split()[0]
                    seq_list.append(os.path.join(cls_name, line_data))




    return seq_list, seq_out




def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
