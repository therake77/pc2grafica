from __future__ import annotations

from torch import nn

from kanji_cnn.cnn.radical_cnn import KanjiRadicalCNN
from kanji_cnn.cnn.resnet import ResNet50Radical

MODEL_NAMES = ("cnn", "resnet50")

#la ResNet50 parte de pesos de ImageNet: un lr alto los destruiría en las primeras iteraciones
DEFAULT_LR = {"cnn": 1e-3, "resnet50": 1e-4}


def build_model(name: str, n_classes: int, *, pretrained: bool = False) -> nn.Module:
    '''Construye el modelo `name`. `pretrained` solo aplica a resnet50 (pesos de ImageNet)'''
    if name == "cnn":
        return KanjiRadicalCNN(n_classes)
    if name == "resnet50":
        return ResNet50Radical(n_classes, pretrained=pretrained)
    raise ValueError(f"modelo desconocido: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def size_mb(model: nn.Module) -> float:
    '''Tamaño de los pesos en float32 (parámetros + buffers como las estadísticas de BatchNorm)'''
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20
