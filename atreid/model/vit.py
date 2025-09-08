import torch.nn as nn
from model.vit_pytorch import vit_base_patch16_224_ReID
from model.vit_pytorch2 import vit_base_patch16_224_ReID_moe


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)
        m.bias.requires_grad_(False)


class vit(nn.Module):
    def __init__(self, num_p, imsize=(256, 128), drop=0.1, stride=16):
        super(vit, self).__init__()
        model_path = "/home/lixulin/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"
        dim = 768
        self.base = vit_base_patch16_224_ReID(
            img_size=imsize, stride_size=stride, drop_path_rate=drop)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.bottleneck = nn.BatchNorm1d(dim)
        self.classifier = nn.Linear(dim, num_p, bias=False)
        self.bottleneck.apply(weights_init)
        self.classifier.apply(weights_init)

    def forward(self, x):
        x = self.base(x)
        cls = x[:, 0]
        f = self.bottleneck(cls)
        if self.training:
            y = self.classifier(f)
            return cls, y
        else:
            return f
