"""ResNet-18 2D aligned with the Keras MobileNet MRI slice protocol.

Source architecture: KAN-MRI ``ResNet18_2D`` (ImageNet stem, FC 512→256→C).
This module does not import that package. Defaults follow the trained
MobileNet classifier on ROI 4 mm MRI:

- 256×256 RGB tiles (grayscale slice repeated on 3 channels, values in [0, 1])
- 2-class logits (CrossEntropyLoss; softmax only at evaluation)
- ImageNet initialization when ``pretrained`` is true

``ResNet2D`` (kernel-3 / stride-1 stem for small inputs) is intentionally absent.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# Same axial window and tile size as classifiers/classifiers_roi_4mm/mri.
MOBILENET_IMAGE_SIZE = 256
MOBILENET_IN_CHANNELS = 3
MOBILENET_NUM_CLASSES = 2
MOBILENET_SLICE_RANGE = (78, 112)

DEFAULT_CONFIG: Dict[str, Any] = {
    "in_channels": MOBILENET_IN_CHANNELS,
    "num_classes": MOBILENET_NUM_CLASSES,
    "pretrained": True,
    "image_size": MOBILENET_IMAGE_SIZE,
}


def adapt_conv_weights(
    conv_layer: nn.Conv2d,
    new_in_channels: int,
    pretrained_weights: torch.Tensor,
) -> None:
    """Adapt pretrained conv weights when the input channel count changes."""
    original_weights = pretrained_weights
    original_in_channels = int(original_weights.shape[1])

    if new_in_channels == original_in_channels:
        conv_layer.weight = nn.Parameter(original_weights.detach().clone())
        return

    new_weights = original_weights.mean(dim=1, keepdim=True)
    new_weights = new_weights.repeat(1, new_in_channels, 1, 1)
    new_weights = new_weights.detach().clone()
    new_weights.requires_grad_(True)

    conv_layer.weight = nn.Parameter(new_weights)
    conv_layer.in_channels = new_in_channels


def _init_classifier(classifier: nn.Module) -> None:
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def _make_classifier(in_features: int, num_classes: int, hidden: int) -> nn.Sequential:
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, hidden),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(hidden, num_classes),
    )
    _init_classifier(classifier)
    return classifier


class ResNet18_2D(nn.Module):
    """ImageNet ResNet-18 with a 512→256→C head, on MobileNet-sized RGB tiles.

    The stem keeps the ImageNet 7×7 stride-2 convolution. Spatial size 256 is
    valid because the backbone ends in adaptive average pooling. With
    ``in_channels=3`` the pretrained RGB filters are copied as-is.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        self.in_channels = int(cfg["in_channels"])
        self.num_classes = int(cfg["num_classes"])
        self.pretrained = bool(cfg["pretrained"])
        self.image_size = int(cfg["image_size"])

        weights = ResNet18_Weights.DEFAULT if self.pretrained else None
        self.model = resnet18(weights=weights)
        original_conv1_weights = None
        if weights is not None:
            original_conv1_weights = self.model.conv1.weight.data.clone()
        old = self.model.conv1
        new_conv1 = nn.Conv2d(
            self.in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if original_conv1_weights is not None:
            adapt_conv_weights(new_conv1, self.in_channels, original_conv1_weights)
        self.model.conv1 = new_conv1

        num_features = int(self.model.fc.in_features)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes),
        )
        for module in self.model.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Registered here so its parameters are in the optimizer from step 0.
        self.input_bn = nn.BatchNorm2d(self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW with {self.in_channels} channels, got {tuple(x.shape)}"
            )
        height, width = int(x.shape[-2]), int(x.shape[-1])
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} tiles, got {height}x{width}"
            )
        x = self.input_bn(x)
        return self.model(x)
