import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
import numpy as np


def get_transform(args, *, test_pre_resized=False):
    transform_train = T.Compose([
        T.Resize((args.ih, args.iw), interpolation=T.InterpolationMode.BILINEAR),#BICUBIC
        T.Pad(10),
        T.RandomCrop((args.ih, args.iw)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        RandomErasing(probability=0.5, min_area=0.02, max_area=args.era,
                      mode='pixel', max_count=1, device='cpu'),
    ])
    test_ops = []
    if not test_pre_resized:
        test_ops.append(T.Resize((args.ih, args.iw), interpolation=T.InterpolationMode.BILINEAR))
    test_ops.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    transform_test = T.Compose(test_ops)
    return transform_train, transform_test
