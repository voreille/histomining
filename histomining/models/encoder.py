import os

import timm
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
    "resnet50_trunc": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "uni_v1": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "conch_v1": {"mean": OPENAI_MEAN, "std": OPENAI_STD},
    "conch_v1_5": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
}


def get_eval_transforms(mean, std, target_img_size=-1):
    trsforms = []

    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    trsforms.append(transforms.ToTensor())
    trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms


class TimmCNNEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50.tv_in1k",
        kwargs: dict = {
            "features_only": True,
            "out_indices": (3,),
            "pretrained": True,
            "num_classes": 0,
        },
        pool: bool = True,
    ):
        super().__init__()
        assert kwargs.get("pretrained", False), "only pretrained models are supported"
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out


def get_encoder(target_img_size=224):
    model_name = "resnet50_trunc"
    print("loading model checkpoint")
    model = TimmCNNEncoder()

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(
        mean=constants["mean"], std=constants["std"], target_img_size=target_img_size
    )

    return model, img_transforms
