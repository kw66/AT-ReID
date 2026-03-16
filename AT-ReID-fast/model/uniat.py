from pathlib import Path

import torch
import torch.nn as nn

from model.moae import vit_base_patch16_224_ReID_moe


DEFAULT_PRETRAINED_PATH = Path.home() / ".cache" / "torch" / "checkpoints" / "jx_vit_base_p16_224-80ecf9dd.pth"
TASK_NAMES = ("dt-st", "dt-lt", "nt-st", "nt-lt", "ad-st", "ad-lt")


def weights_init(module):
    classname = module.__class__.__name__
    if "Linear" in classname:
        nn.init.normal_(module.weight.data, 0, 0.001)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.01)
        nn.init.zeros_(module.bias.data)
        module.bias.requires_grad_(False)


def resolve_pretrained_path(pretrained_path=None):
    if pretrained_path:
        return str(Path(pretrained_path).expanduser())
    return str(DEFAULT_PRETRAINED_PATH)


class uniat(nn.Module):
    def __init__(
        self,
        num_p,
        num_c,
        imsize=(256, 128),
        drop=0.1,
        stride=16,
        ncls=1,
        moae=False,
        moae_router_noise=0.01,
        use_pretrained=True,
        pretrained_path=None,
        attention_backend="auto",
    ):
        super().__init__()
        if ncls not in {1, 6}:
            raise ValueError(f"UniAT expects ncls to be 1 or 6, but got {ncls}.")
        if moae and ncls != 6:
            raise ValueError("MOAE is defined for the 6-CLS formulation. Please use ncls=6 when moae is enabled.")
        dim = 768
        self.ncls = ncls
        self.base = vit_base_patch16_224_ReID_moe(
            img_size=imsize,
            stride_size=stride,
            drop_path_rate=drop,
            ncls=ncls,
            moae=moae,
            moae_router_noise=moae_router_noise,
            attention_backend=attention_backend,
        )
        self.vit_attention_info = dict(getattr(self.base, "attention_info", {}))

        if use_pretrained:
            resolved_path = resolve_pretrained_path(pretrained_path)
            if not Path(resolved_path).is_file():
                raise FileNotFoundError(
                    "Pretrained ViT weight not found. "
                    f"Expected: {resolved_path}. "
                    "Please pass --pretrained-path or use --no-pretrained."
                )
            self.base.load_param(resolved_path)
            print(f"Loading pretrained ImageNet model from {resolved_path}")
        else:
            print("Training without ImageNet pretrained initialization.")

        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(6)])
        num_classes = [num_c, num_p, num_c, num_p, num_c, num_p]
        self.classifier = nn.ModuleList([nn.Linear(dim, num_classes[i], bias=False) for i in range(6)])
        self.head_names = TASK_NAMES
        self.bottleneck.apply(weights_init)
        self.classifier.apply(weights_init)

    def forward(self, x):
        cls = self.base(x)[:, :self.ncls]
        if self.ncls == 1:
            cls = cls.expand(-1, 6, -1)
        cls = tuple(cls.unbind(1))
        features = [self.bottleneck[i](cls[i]) for i in range(6)]
        logits = [self.classifier[i](features[i]) for i in range(6)]
        if self.training:
            return list(cls), logits
        return features
