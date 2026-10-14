from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kanji_cnn.preprocess import dict_builder, etl9b
from kanji_cnn.preprocess.radicals import KANGXI_RADICALS

ARTIFACT_DIR = Path("data/build")   #Path to the pipeline output directory 
SAMPLE_INDEX_PATH = ARTIFACT_DIR / "sample_index.npy"   
CLASS_LABELS_PATH = ARTIFACT_DIR / "class_labels.json"

SEED = 1234     #For reproducibility
MIN_SAMPLES_PER_CLASS = 500     #Classes constraints
MAX_SAMPLES_PER_CLASS = 1000    
MAX_CLASSES = 50                #Max of classes 

_HIRAGANA_RANGE = ("ぁ", "ゟ")  #ETL database includes some hiragana, which need to be removed


@dataclass(frozen=True)
class RecordLoc:    #Clase más simple que EtlRecord. Representa posición de un registro
    file_index: int
    byte_offset: int
    writer_id: int


def is_hiragana(ch: str) -> bool:
    '''Utilidad para saber si un caracter es hiragana. ETL contiene caracteres hiragana'''
    return _HIRAGANA_RANGE[0] <= ch <= _HIRAGANA_RANGE[1]


def scan_classes(etl_paths: Sequence[Path]) -> dict[str, list[RecordLoc]]:
    '''Escanea todos los archivos indicados por `etl_paths`, y devuelve un diccionario de pares (`kanji_en_caracter`, `registros_ETL_correspondientes`) '''
    by_char: dict[str, list[RecordLoc]] = defaultdict(list)
    for record in etl9b.iter_all_records(etl_paths, with_image=False):
        by_char[record.char].append(
            RecordLoc(record.file_index, record.byte_offset, record.writer_id)
        )
    return by_char


def filter_kanji(by_char: dict[str, list[RecordLoc]]) -> dict[str, list[RecordLoc]]:
    '''Elimina kanjis que no sean hiragana'''
    return {char: locs for char, locs in by_char.items() if not is_hiragana(char)}


def group_by_radical(
    by_char: dict[str, list[RecordLoc]], kanji_to_radical: dict[str, int]
) -> dict[int, list[RecordLoc]]:
    pools: dict[int, list[RecordLoc]] = defaultdict(list)
    for char, locs in by_char.items():
        radical = kanji_to_radical.get(char)
        if radical is not None:
            pools[radical].extend(locs)
    return pools


def balance_classes(
    pools: dict[int, list[RecordLoc]],
    *,
    min_count: int,
    max_count: int,
    max_classes: int,
    seed: int,
) -> dict[int, list[RecordLoc]]:
    surviving = {
        radical: locs for radical, locs in pools.items() if len(locs) >= min_count
    }
    if len(surviving) > max_classes:
        ranked = sorted(surviving.items(), key=lambda kv: len(kv[1]), reverse=True)
        surviving = dict(ranked[:max_classes])

    rng = random.Random(seed)
    balanced: dict[int, list[RecordLoc]] = {}
    for radical, locs in surviving.items():
        ordered = sorted(locs, key=lambda loc: (loc.file_index, loc.byte_offset))
        if len(ordered) > max_count:
            ordered = rng.sample(ordered, max_count)
        balanced[radical] = ordered
    return balanced


def finalize_and_write(balanced: dict[int, list[RecordLoc]], out_dir: Path) -> None:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    radical_main_form = {entry.index: entry.main_form for entry in KANGXI_RADICALS}

    rows = []
    class_meta = []
    for class_index, radical in enumerate(sorted(balanced.keys())):
        locs = balanced[radical]
        class_meta.append(
            {
                "class_index": class_index,
                "radical_index": radical,
                "radical_glyph": radical_main_form[radical],
                "count": len(locs),
            }
        )
        for loc in locs:
            rows.append((loc.file_index, loc.byte_offset, loc.writer_id, class_index))

    sample_index = np.array(rows, dtype=np.int64)
    np.save(out_dir / "sample_index.npy", sample_index)
    (out_dir / "class_labels.json").write_text(
        json.dumps(class_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    etl_paths = etl9b.get_etl9b_paths()

    print("Scanning all ETL9B records...")
    by_char = scan_classes(etl_paths)
    total_records = sum(len(v) for v in by_char.values())
    print(f"  {len(by_char)} distinct characters, {total_records} records")

    kanji_only = filter_kanji(by_char)
    print(f"  {len(kanji_only)} kanji after dropping hiragana")

    print("Labeling visual radicals via IDS decomposition...")
    kanji_to_radical = dict_builder.build_visual_radical_labels(kanji_only.keys())
    print(
        f"  {len(kanji_to_radical)} kanji labeled, "
        f"{len(kanji_only) - len(kanji_to_radical)} excluded (ambiguous / no match)"
    )

    pools = group_by_radical(kanji_only, kanji_to_radical)
    print(f"  {len(pools)} radical classes have at least one kanji")

    balanced = balance_classes(
        pools,
        min_count=MIN_SAMPLES_PER_CLASS,
        max_count=MAX_SAMPLES_PER_CLASS,
        max_classes=MAX_CLASSES,
        seed=SEED,
    )
    n_before_cap = sum(1 for locs in pools.values() if len(locs) >= MIN_SAMPLES_PER_CLASS)
    print(
        f"  {n_before_cap} classes clear the {MIN_SAMPLES_PER_CLASS}-sample floor; "
        f"kept {len(balanced)} after the {MAX_CLASSES}-class cap"
    )

    finalize_and_write(balanced, ARTIFACT_DIR)
    total_samples = sum(len(v) for v in balanced.values())
    print(f"Wrote {len(balanced)} classes, {total_samples} total samples to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
