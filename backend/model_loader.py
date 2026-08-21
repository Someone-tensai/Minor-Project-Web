"""
Handles loading arbitrary .pth files saved either as:
  - a full model object      (torch.save(model, path))
  - a raw state_dict         (torch.save(model.state_dict(), path))
  - a wrapped checkpoint     (torch.save({"state_dict": ..., "epoch": ..., ...}, path))

If your files fail to load, the error message printed will tell you which
case it hit and what went wrong -- send that back and we'll patch this.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_architecture(arch: str, num_classes: int) -> nn.Module:
    """
    Matches our_model() from same_param_training/models.py exactly:
    same torchvision arch, same final-layer swap. weights=None here since
    we immediately overwrite with the trained state_dict anyway -- no need
    to download pretrained ImageNet weights just to discard them.
    """
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def load_model(file_path: str, arch: str, num_classes: int) -> nn.Module:
    checkpoint = torch.load(file_path, map_location="cpu")

    # Case 1: a full nn.Module object was saved directly
    if isinstance(checkpoint, nn.Module):
        model = checkpoint

    else:
        # Case 2/3: it's a dict of some kind -- try to find the actual state_dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint  # assume it's already a raw state_dict
        else:
            raise ValueError(
                f"Unrecognized checkpoint format in {file_path}: {type(checkpoint)}"
            )

        # Strip a "module." prefix if the model was saved from nn.DataParallel
        state_dict = { (k[7:] if k.startswith("module.") else k): v
                        for k, v in state_dict.items() }

        model = build_architecture(arch, num_classes)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load failed for {file_path}, retrying non-strict: {e}")
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model
