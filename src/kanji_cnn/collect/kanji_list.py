from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KanjiClass:
    name: str                   # nombre de la carpeta: drawings/<name>/
    radical: str                # radical que define la clase (sólo para mostrarlo)
    kanjis: tuple[str, ...]
    times: int                  # veces que cada escritor dibuja cada kanji de la clase


# Clases en el orden de rotación de kanji-work-split.md
CLASSES: tuple[KanjiClass, ...] = (
    KanjiClass("water", "氵", tuple("池河油泳波海洋流湖"), 9),
    KanjiClass("tree", "木", tuple("机村材林板松校桜杉"), 9),
    KanjiClass("earth", "土", tuple("地坂均坊城型域場坪"), 9),
    KanjiClass("fire", "火", tuple("灯炉炊炒畑炎炭灰"), 10),
    KanjiClass("metal", "金", tuple("針釣鈴鉄鉛銀銅銭"), 10),
)

  # Cada escritor empieza en una clase distinta y sigue la misma rotación
WRITER_START = {"jarem": "water", "amir": "tree", "piero": "earth", "diego": "fire"}

KANJI_TO_CLASS = {kanji: cls for cls in CLASSES for kanji in cls.kanjis}


def writer_plan(writer: str) -> list[tuple[KanjiClass, str]]:
    '''Devuelve los pares (`clase`, `kanji`) en el orden en que `writer` debe dibujarlos'''
    start = [cls.name for cls in CLASSES].index(WRITER_START[writer])
    rotated = CLASSES[start:] + CLASSES[:start]
    return [(cls, kanji) for cls in rotated for kanji in cls.kanjis]
