from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

#media y desviación del canal gris equivalente de ImageNet (promedio de las de RGB)
IMAGENET_GRAY_MEAN = 0.449
IMAGENET_GRAY_STD = 0.226


class ResNet50Radical(nn.Module):
    '''ResNet50 de torchvision adaptada a imágenes de 1 canal (64×64) y `n_classes` radicales.
    Con `pretrained` parte de los pesos de ImageNet; para cargar un checkpoint basta `pretrained=False`'''

    def __init__(self, n_classes: int = 5, pretrained: bool = False) -> None:
        super().__init__()
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        #conv1 de 3 canales → 1 canal: sumar los pesos RGB da la misma respuesta que la
        #convolución original sobre una imagen gris replicada en los 3 canales
        old = self.net.conv1
        self.net.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            self.net.conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True))

        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_GRAY_MEAN))
        self.register_buffer("std", torch.tensor(IMAGENET_GRAY_STD))

    def forward(self, x):
        #normalizar como ImageNet aquí permite usar el mismo pipeline de datos que la CNN
        return self.net((x - self.mean) / self.std)
