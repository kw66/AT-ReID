import torch.nn as nn
from model.uniat_module import vit_base_patch16_224_ReID_moe

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)
        m.bias.requires_grad_(False)


class uniat(nn.Module):
    def __init__(self, num_p, num_c, imsize=(256, 128), drop=0.2, stride=16, moe=True):
        super(vitmoe, self).__init__()
        model_path = "/home/lixulin/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"
        dim = 768
        self.base = vit_base_patch16_224_ReID_moe(
            img_size=imsize, stride_size=stride, drop_path_rate=drop, ncls=6, moe=moe)
        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(dim) for i in range(6)])
        num = [num_c, num_p, num_c, num_p, num_c, num_p]
        self.classifier = nn.ModuleList([nn.Linear(dim, num[i], bias=False) for i in range(6)])
        self.bottleneck.apply(weights_init)
        self.classifier.apply(weights_init)

    def forward(self, x):
        cls = self.base(x, mids).unbind(1)
        f = [self.bottleneck[i](cls[i]) for i in range(6)]
        y = [self.classifier[i](f[i]) for i in range(6)]
        if self.training:
            return [cls[i] for i in range(6)], y
        else:
            return f
