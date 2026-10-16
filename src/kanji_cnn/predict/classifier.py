from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image

from kanji_cnn.cnn.models import build_model
from kanji_cnn.dataset.drawings import preprocess

DEFAULT_CHECKPOINT = Path("data/runs/cnn/model.pt")    # lo que escribe `make train`; relativo al directorio de trabajo


class Candidate(TypedDict):
    '''Una clase posible para el dibujo. El navegador la recibe tal cual, en JSON'''
    name: str               # carpeta de la clase, p. ej. "fire"
    radical: str            # glifo del radical, p. ej. "火"
    probability: float      # entre 0 y 1; las de todas las clases suman 1


class RadicalClassifier:
    '''Modelo entrenado, cargado una sola vez, que clasifica el PNG del lienzo'''

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT):
        ckpt = torch.load(checkpoint, map_location="cpu")     # igual que main.load_checkpoint
        self.model = build_model(ckpt["model"], ckpt["n_classes"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()                                     # BatchNorm con sus estadísticas de train, sin Dropout
        self.class_labels = ckpt["class_labels"]
        self.size = ckpt["img_size"]

    def classify(self, png: bytes) -> list[Candidate]:
        '''Todas las clases con su probabilidad para el dibujo `png`, de la más probable a la menos'''
        with Image.open(BytesIO(png)) as img:
            x = torch.from_numpy(preprocess(img, self.size))  # el mismo preprocesado que en el entrenamiento
        with torch.no_grad():
            probs = self.model(x.reshape(1, 1, self.size, self.size)).softmax(dim=1)[0]  # (lote, canal, alto, ancho)
        candidates = [
            Candidate(name=c["name"], radical=c["radical_glyph"], probability=p)
            for c, p in zip(self.class_labels, probs.tolist())
        ]
        return sorted(candidates, key=lambda c: c["probability"], reverse=True)
