from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from kanji_cnn.collect.kanji_list import CLASSES

DRAWINGS_DIR = Path("drawings")     # relativo al directorio de trabajo, como en collect/storage.py
IMG_SIZE = 64                       # lado de la imagen que recibe la red
MARGIN = 0.08                       # fracción del lado que queda libre a cada lado del kanji

#glifo del radical de cada clase, para las gráficas. La etiqueta en sí es el nombre de la carpeta
RADICAL_GLYPH = {cls.name: cls.radical for cls in CLASSES}

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Drawing:
    path: Path
    label: str      # carpeta de primer nivel: drawings/<label>/<kanji>/<writer>-<NN>.png
    kanji: str
    writer: str
    number: int


def scan_drawings(root: Path = DRAWINGS_DIR) -> tuple[list[str], list[Drawing]]:
    '''Recorre root/<clase>/<kanji>/<escritor>-<NN>.png. La etiqueta de cada dibujo es el nombre de su
    carpeta de clase; las clases se ordenan alfabéticamente y su índice es su posición en esa lista'''
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de dibujos {root} (¿se ejecuta desde la raíz del repo?)")
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_names:
        raise ValueError(f"{root} no tiene carpetas de clase")

    samples = []
    for label in class_names:
        found = 0
        for path in sorted((root / label).glob("*/*.png")):
            writer, sep, number = path.stem.rpartition("-")
            if not sep or not writer or not number.isdigit():
                print(f"Aviso: se ignora {path}: el nombre no es <escritor>-<NN>.png")
                continue
            samples.append(Drawing(path, label, path.parent.name, writer, int(number)))
            found += 1
        if found == 0:
            raise ValueError(f"La clase {label} no tiene dibujos en {root / label}")
    return class_names, samples


def ink(img: Image.Image) -> Image.Image:
    '''Imagen en gris con la tinta en 255 y el fondo en 0. En los PNG de la app el trazo solo vive
    en el canal alfa (como en collect/app.py:is_blank); sin alfa se invierte el gris'''
    if "A" in img.getbands():
        return img.getchannel("A")
    return ImageOps.invert(img.convert("L"))


def preprocess(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    '''Recorta el kanji a su caja, lo centra en un cuadrado con margen (sin deformarlo) y lo reduce a
    size×size. Así no influye dónde ni a qué tamaño se dibujó en el lienzo de 300×300.
    Devuelve float32 en [0, 1] con tinta = 1'''
    gray = ink(img)
    bbox = gray.getbbox()
    if bbox is None:        # lienzo vacío: la app no los guarda, pero no debe romper nada
        return np.zeros((size, size), dtype=np.float32)
    crop = gray.crop(bbox)
    side = round(max(crop.size) * (1 + 2 * MARGIN))
    square = Image.new("L", (side, side), 0)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return np.asarray(square.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255


def load_image(path: Path, size: int = IMG_SIZE) -> torch.Tensor:
    '''Lee y preprocesa un dibujo: tensor (1, size, size)'''
    with Image.open(path) as img:
        return torch.from_numpy(preprocess(img, size)).unsqueeze(0)


class DrawingsDataset(Dataset):
    '''Todos los dibujos de `root`, preprocesados y en memoria: (N, 1, 64, 64), ~26 MB con 1600 dibujos.
    `transform`, si se da, se aplica al tensor en cada __getitem__'''

    def __init__(
        self,
        root: Path = DRAWINGS_DIR,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        size: int = IMG_SIZE,
    ):
        self.root = Path(root)
        self.size = size
        self.class_names, self.samples = scan_drawings(self.root)
        class_index = {name: i for i, name in enumerate(self.class_names)}
        self.images = torch.stack([load_image(s.path, size) for s in self.samples])
        self.targets = np.array([class_index[s.label] for s in self.samples])
        self.kanjis = np.array([s.kanji for s in self.samples])
        self.writers = np.array([s.writer for s in self.samples])
        #rutas relativas a root (con "/"): identifican cada dibujo aunque el repo esté en otra carpeta
        self.paths = np.array([s.path.relative_to(self.root).as_posix() for s in self.samples])
        self.num_classes = len(self.class_names)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = self.images[i]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[i])

    @property
    def fingerprint(self) -> str:
        '''Huella del conjunto de archivos. Si cambia (dibujos nuevos o borrados), el split de una
        semilla dada ya no es el mismo'''
        return hashlib.sha1("\n".join(sorted(self.paths)).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        writers, counts = np.unique(self.writers, return_counts=True)
        return {
            "n_images": len(self),
            "n_kanji": len(set(self.kanjis)),
            "writers": {str(w): int(c) for w, c in zip(writers, counts)},
            "fingerprint": self.fingerprint,
        }


class ImageFiles(Dataset):
    '''Dibujos leídos bajo demanda a partir de sus rutas, p. ej. las de test guardadas en predictions.npz'''

    def __init__(self, paths: Sequence[Path], size: int = IMG_SIZE):
        self.paths = [Path(p) for p in paths]
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return load_image(self.paths[i], self.size), -1


def class_labels(dataset: DrawingsDataset) -> list[dict]:
    '''Metadatos de cada clase: nombre (= carpeta), glifo del radical, kanji distintos y nº de dibujos'''
    return [
        {
            "class_index": i,
            "name": name,
            "radical_glyph": RADICAL_GLYPH.get(name, ""),
            "n_kanji": len(set(dataset.kanjis[dataset.targets == i])),
            "count": int((dataset.targets == i).sum()),
        }
        for i, name in enumerate(dataset.class_names)
    ]


def split_column(n: int, splits: Sequence[np.ndarray]) -> np.ndarray:
    '''Nombre del conjunto (train/val/test) de cada muestra'''
    column = np.full(n, "", dtype=object)
    for name, idx in zip(SPLIT_NAMES, splits):
        column[idx] = name
    return column


def write_manifest(dataset: DrawingsDataset, splits: Sequence[np.ndarray], path: Path) -> None:
    '''CSV con una fila por dibujo: ruta, etiqueta (carpeta), índice de clase, kanji, escritor, número y split'''
    path.parent.mkdir(parents=True, exist_ok=True)
    split = split_column(len(dataset), splits)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_index", "kanji", "writer", "number", "split"])
        for i, s in enumerate(dataset.samples):
            writer.writerow([dataset.paths[i], s.label, dataset.targets[i], s.kanji, s.writer, s.number, split[i]])
