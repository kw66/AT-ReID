from torchvision import models
import torch.nn as nn
import torch
from collections import OrderedDict


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)
        m.bias.requires_grad_(False)


class resnet(nn.Module):
    def __init__(self, num_p):
        super(resnet, self).__init__()
        net = models.resnet50(pretrained=True)
        for mo in net.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        net_layer0 = nn.Sequential(nn.Sequential(OrderedDict([
            ('conv1', net.conv1),('bn1', net.bn1),
            ('relu', net.relu),('maxpool', net.maxpool)])))
        self.base = nn.Sequential(OrderedDict([
            ('layer0', net_layer0), ('layer1', net.layer1), ('layer2', net.layer2),
            ('layer3', net.layer3), ('layer4', net.layer4)]))
        dim = 2048
        self.bottleneck = nn.BatchNorm1d(dim)#eps=1e-5
        self.bottleneck.apply(weights_init)
        self.classifier = nn.Linear(dim, num_p, bias=False)
        self.classifier.apply(weights_init)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'parameters:{sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000.0}')

    def forward(self, x):
        for i in range(5):
            for j in range(layers[i]):
                x = self.base[i][j](x)
        p1 = self.avgpool(x).squeeze()
        f1 = self.bottleneck(p1)
        if self.training:
            y1 = self.classifier(f1)
            return p1, y1
        else:
            return f1
