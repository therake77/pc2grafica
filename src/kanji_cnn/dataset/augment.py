from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from kanji_cnn.dataset.drawings import DrawingsDataset

#Cada radical termina con entre MIN y MAX imágenes en train (originales + variantes)
MIN_PER_CLASS = 500
MAX_PER_CLASS = 1000

#Las transformaciones trabajan sobre la imagen ya preprocesada (64×64, tinta = 1): sobre el PNG de
#300×300, el ruido repartido por el lienzo arruinaría el recorte a la caja del kanji.
#Los filtros son de 3×3: a 64×64 el trazo mide ~2 px y uno de 5×5 lo borraría

Transform = Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]


def gaussian_noise(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    '''Suma ruido gaussiano N(0, σ) a cada píxel'''
    sigma = rng.uniform(0.05, 0.15)
    noisy = img + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 1).astype(np.float32), {"sigma": round(float(sigma), 4)}


def salt_and_pepper(img: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    '''Ruido impulsivo: una proporción `amount` de píxeles pasa a 0 o a 1, mitad y mitad'''
    out = img.copy()
    u = rng.random(img.shape)
    out[u < amount / 2] = 0.0
    out[u > 1 - amount / 2] = 1.0
    return out


def sp_mean(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    '''Sal y pimienta y luego filtro de media 3×3: el ruido no desaparece, se difumina'''
    amount = rng.uniform(0.02, 0.06)
    filtered = ndimage.uniform_filter(salt_and_pepper(img, rng, amount), size=3, mode="constant")
    return filtered.astype(np.float32), {"p": round(float(amount), 4), "k": 3}


def sp_median(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    '''Sal y pimienta y luego filtro de mediana 3×3: elimina casi todo el ruido impulsivo'''
    amount = rng.uniform(0.02, 0.06)
    filtered = ndimage.median_filter(salt_and_pepper(img, rng, amount), size=3, mode="constant")
    return filtered.astype(np.float32), {"p": round(float(amount), 4), "k": 3}


def rotation(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    '''Rota el kanji entre 3° y 15° hacia un lado u otro, alrededor del centro'''
    angle = rng.uniform(3, 15) * rng.choice([-1, 1])
    rotated = ndimage.rotate(img, angle, reshape=False, order=1, mode="constant", cval=0.0)
    return np.clip(rotated, 0, 1).astype(np.float32), {"angle": round(float(angle), 2)}


TRANSFORMS: dict[str, Transform] = {
    "gaussian": gaussian_noise,
    "sp_mean": sp_mean,
    "sp_median": sp_median,
    "rotation": rotation,
}
TRANSFORM_LABELS = {
    "original": "original",
    "gaussian": "ruido gaussiano",
    "sp_mean": "sal y pimienta + media",
    "sp_median": "sal y pimienta + mediana",
    "rotation": "rotación",
}


class AugmentedTrain(Dataset):
    '''Conjunto de train aumentado, en memoria: los originales de train y sus variantes.
    `sources[i]` es el índice en DrawingsDataset del dibujo del que sale la imagen i'''

    def __init__(
        self,
        images: torch.Tensor,
        targets: np.ndarray,
        sources: np.ndarray,
        transforms: np.ndarray,
        rounds: np.ndarray,
        params: list[dict],
        class_names: list[str],
        target_per_class: int,
    ):
        self.images = images
        self.targets = targets
        self.sources = sources
        self.transforms = transforms
        self.rounds = rounds
        self.params = params
        self.class_names = class_names
        self.target_per_class = target_per_class

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.images[i], int(self.targets[i])

    def counts(self) -> dict[str, dict[str, int]]:
        '''Por clase: originales, variantes de cada transformación y total'''
        result = {}
        for c, name in enumerate(self.class_names):
            in_class = self.targets == c
            result[name] = {t: int((in_class & (self.transforms == t)).sum()) for t in TRANSFORM_LABELS}
            result[name]["total"] = int(in_class.sum())
        return result

    def summary(self, min_per_class: int = MIN_PER_CLASS, max_per_class: int = MAX_PER_CLASS) -> dict:
        return {
            "min_per_class": min_per_class,
            "max_per_class": max_per_class,
            "target_per_class": self.target_per_class,
            "counts": self.counts(),
        }


def target_per_class(n_per_class: list[int], min_per_class: int, max_per_class: int) -> int:
    '''Tamaño común de todas las clases: 5 imágenes (original + 4 variantes) por original de la clase
    más grande, acotado a [min, max]. Así las clases quedan equilibradas'''
    return int(np.clip((1 + len(TRANSFORMS)) * max(n_per_class), min_per_class, max_per_class))


def augment_train(
    dataset: DrawingsDataset,
    train_idx: np.ndarray,
    seed: int,
    min_per_class: int = MIN_PER_CLASS,
    max_per_class: int = MAX_PER_CLASS,
) -> AugmentedTrain:
    '''Aumenta los originales de train hasta `target_per_class` imágenes por clase. Cada ronda genera
    las 4 variantes de cada original con parámetros aleatorios nuevos; se hacen las rondas necesarias
    (al menos una) y se conservan las variantes de las primeras rondas hasta llegar al objetivo.
    Determinista para unos mismos `train_idx` y `seed`'''
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    train_targets = dataset.targets[train_idx]
    n_per_class = [int((train_targets == c).sum()) for c in range(dataset.num_classes)]
    if min(n_per_class) == 0:
        raise ValueError("Hay clases sin dibujos en train: no se pueden aumentar")
    target = target_per_class(n_per_class, min_per_class, max_per_class)

    images, targets, sources, transforms, rounds, params = [], [], [], [], [], []
    for c in range(dataset.num_classes):
        originals = train_idx[train_targets == c]
        if len(originals) >= target:
            print(f"Aviso: {dataset.class_names[c]} tiene {len(originals)} originales en train; se submuestrean a {target}")
            originals = np.sort(rng.choice(originals, size=target, replace=False))
        for i in originals:
            images.append(dataset.images[i, 0].numpy())
            targets.append(c)
            sources.append(i)
            transforms.append("original")
            rounds.append(0)
            params.append({})

        n_variants = target - len(originals)
        n_rounds = math.ceil(n_variants / (len(TRANSFORMS) * len(originals)))
        #Cada ronda se reparte en 4 pasadas y en cada pasada cada original recibe una de sus transformaciones
        #(en orden aleatorio). Si la última ronda se recorta, todos los originales quedan con el mismo
        #número de variantes ±1, en vez de unos con 4 y otros con 1
        names = list(TRANSFORMS)
        planned = []
        for r in range(1, n_rounds + 1):
            order = rng.permutation(originals)
            transform_order = {i: rng.permutation(len(names)) for i in order}
            for k in range(len(names)):
                planned += [(r, i, names[transform_order[i][k]]) for i in order]
        for r, i, name in planned[:n_variants]:
            image, p = TRANSFORMS[name](dataset.images[i, 0].numpy(), rng)
            images.append(image)
            targets.append(c)
            sources.append(i)
            transforms.append(name)
            rounds.append(r)
            params.append(p)

    return AugmentedTrain(
        images=torch.from_numpy(np.stack(images)).unsqueeze(1),
        targets=np.array(targets),
        sources=np.array(sources),
        transforms=np.array(transforms),
        rounds=np.array(rounds),
        params=params,
        class_names=dataset.class_names,
        target_per_class=target,
    )


def write_augmented_manifest(train_set: AugmentedTrain, dataset: DrawingsDataset, path: Path) -> None:
    '''CSV con una fila por imagen de train: el dibujo del que sale, su etiqueta y la transformación aplicada'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "label", "class_index", "kanji", "writer", "transform", "round", "params"])
        for i in range(len(train_set)):
            src = train_set.sources[i]
            writer.writerow([
                dataset.paths[src], dataset.class_names[train_set.targets[i]], train_set.targets[i],
                dataset.kanjis[src], dataset.writers[src], train_set.transforms[i], train_set.rounds[i],
                json.dumps(train_set.params[i]),
            ])
