from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
from torchvision import transforms

transform_bg = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)], p=0.2
    ),
]
)

transform_fg = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomApply(
    #     transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05), p=0.8),
    transforms.RandomApply(
        [transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=10, resample=0, fillcolor=0)], p=0.8),
    transforms.RandomPerspective(p=0.2, distortion_scale=0.2)
])



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class MergingDataset(Dataset):
    def __init__(self, fg_dir, bg_dir, scale=1, mask_suffix=''):
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        logging.info(f'Creating dataset with {len(glob(self.fg_dir+"*")) * 10} examples ')

    def __len__(self):
        length = len(glob(self.fg_dir+"*")) * 10
        return length

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def create_img_mask(self, bg, fg):

        bg = transform_bg(bg)
        fg = transform_fg(fg)

        # fg.show()
        if bg.size[0] < fg.size[0] or bg.size[1] < fg.size[1]:
            case = fg.size[0] / fg.size[1] > bg.size[0] / bg.size[1]
            if case:  # сжимать fg 0 до bg 0
                base_width = int(bg.size[0] * max((0.9, random.random())))
                wpercent = (base_width / float(fg.size[0]))
                hsize = int((float(fg.size[1]) * float(wpercent)))
                h_range = bg.size[1] - hsize
                w_range = bg.size[0] - base_width
                fg = fg.resize((base_width, hsize), Image.ANTIALIAS)
                wh = (random.randrange(w_range), random.randrange(h_range))
            if not case:
                base_hight = int(bg.size[1] * max((0.9, random.random())))
                wpercent = (base_hight / float(fg.size[1]))
                wsize = int((float(fg.size[0]) * float(wpercent)))
                w_range = bg.size[0] - wsize
                h_range = bg.size[1] - base_hight
                fg = fg.resize((wsize, base_hight), Image.ANTIALIAS)
                wh = (random.randrange(w_range), random.randrange(h_range))
        else:
            wh = (random.randrange(bg.size[0] - fg.size[0]), random.randrange(bg.size[1] - fg.size[1]))

        img = bg
        fg = fg.convert('RGBA')
        img.paste(fg, wh, fg)

        fg_mask = Image.new('1', fg.size, 0)
        data = fg.getdata()

        new_data = []

        for item in data:
            if item[3] > 0:
                new_data.append(1)
            else:
                new_data.append(0)

        fg_mask.putdata(new_data)

        mask = Image.new("1", bg.size, 0)
        mask.paste(fg_mask, wh)

        return img, mask

    def __getitem__(self, i):

        fg_file = glob(self.fg_dir + '*')
        bg_file = glob(self.bg_dir + '*')

        fg = Image.open(fg_file[i // 10], )
        bg = Image.open(bg_file[int(i * (len(bg_file) / len(fg_file))) // 10])

        img, mask = self.create_img_mask(bg, fg)
        # My transforms

        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
