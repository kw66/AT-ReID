import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
import numpy as np


class gray(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if self.p == 0:
            return img
        elif np.random.uniform(0, 1) > self.p:
            return img
        else:
            idx = np.random.randint(0, 4)
            if idx == 0:
                img[1, :, :] = img[0, :, :]
                img[2, :, :] = img[0, :, :]
            elif idx == 1:
                img[0, :, :] = img[1, :, :]
                img[2, :, :] = img[1, :, :]
            elif idx == 2:
                img[0, :, :] = img[2, :, :]
                img[1, :, :] = img[2, :, :]
            elif idx == 3:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
            return img


def get_transform(args):
    transform_train = T.Compose([
        T.Resize((args.ih, args.iw), interpolation=T.InterpolationMode.BILINEAR),#BICUBIC
        T.Pad(10),
        T.RandomCrop((args.ih, args.iw)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        RandomErasing(probability=0.5, min_area=0.02, max_area=args.era,
                      mode='pixel', max_count=1, device='cpu'),
        #gray(p=args.gray),
    ])
    transform_test = T.Compose([
        T.Resize((args.ih, args.iw), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return transform_train, transform_test
