"""Runtime torch.utils.data.Dataset for the ETL9B visual-radical task.

Loads only the small artifacts produced by `kanji_cnn.preprocess.build_dataset`
(`sample_index.npy` + `class_labels.json`) and seeks directly into the raw
ETL9B binary files per sample -- no IDS/radical logic and no image
pre-extraction needed at train time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from kanji_cnn.preprocess import etl9b


def default_transform(img: Image.Image) -> torch.Tensor:
    arr = etl9b.image_to_array(img)  # uint8 HxW, values 0/1
    return torch.from_numpy(arr).unsqueeze(0).float()  # (1, 63, 64)


class EtlVisualRadicalDataset(Dataset):
    def __init__(
        self,
        index_path: Path = Path("data/build/sample_index.npy"),
        class_labels_path: Path = Path("data/build/class_labels.json"),
        etl_root: Path = Path("data/ETL9B"),
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        self._index = np.load(index_path)  # (N, 4): file_index, byte_offset, writer_id, class_index
        self._class_meta = json.loads(Path(class_labels_path).read_text(encoding="utf-8"))
        self._paths = etl9b.get_etl9b_paths(etl_root)
        self.num_classes = len(self._class_meta)
        self.transform = transform or default_transform

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        file_index, byte_offset, _writer_id, class_index = (int(x) for x in self._index[i])
        record = etl9b.read_record_at(self._paths[file_index], file_index, byte_offset)
        return self.transform(record.image), class_index
