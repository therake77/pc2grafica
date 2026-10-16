from __future__ import annotations
from pathlib import Path
from kanji_cnn.collect.kanji_list import KANJI_TO_CLASS, WRITER_START, KanjiClass, writer_plan

DRAWINGS_DIR = Path("drawings")     # relativo al directorio de trabajo, como data/


def drawing_dir(kanji: str) -> Path:
    '''Carpeta de los dibujos de `kanji`: drawings/<clase>/<kanji>/. La clase sale de la tabla'''
    return DRAWINGS_DIR / KANJI_TO_CLASS[kanji].name / kanji


def count_done(writer: str, kanji: str) -> int:
    '''Cuántas veces `writer` ya dibujó `kanji`, contando sus archivos en disco'''
    return len(list(drawing_dir(kanji).glob(f"{writer}-*.png")))


def next_to_draw(writer: str) -> tuple[KanjiClass, str] | None:
    '''Primer (`clase`, `kanji`) del plan de `writer` que no llega a su cuota. None si ya terminó'''
    for cls, kanji in writer_plan(writer):
        if count_done(writer, kanji) < cls.times:
            return cls, kanji
    return None


def progress(writer: str) -> tuple[int, int]:
    '''(dibujos hechos, dibujos totales) de `writer`. Lo que pase de la cuota no cuenta'''
    plan = writer_plan(writer)
    done = sum(min(count_done(writer, kanji), cls.times) for cls, kanji in plan)
    return done, sum(cls.times for cls, _ in plan)


def list_drawings(writer: str) -> list[tuple[str, list[int]]]:
    '''(kanji, números de dibujo) de `writer`, en el orden de su plan. Omite los kanji sin dibujos'''
    result = []
    for _, kanji in writer_plan(writer):
        numbers = sorted(int(p.stem.rsplit("-", 1)[1]) for p in drawing_dir(kanji).glob(f"{writer}-*.png"))
        if numbers:
            result.append((kanji, numbers))
    return result


def drawing_path(writer: str, kanji: str, n: int) -> Path:
    '''Ruta de un dibujo que ya existe. FileNotFoundError si no está; KeyError/ValueError si kanji o escritor no son válidos'''
    if writer not in WRITER_START:
        raise ValueError(f"Escritor desconocido: {writer}")
    path = drawing_dir(kanji) / f"{writer}-{n:02d}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def replace_drawing(writer: str, kanji: str, n: int, png: bytes) -> Path:
    '''Sobreescribe un dibujo existente (el que se retocó). A diferencia de save_drawing, no crea archivos nuevos'''
    path = drawing_path(writer, kanji, n)
    path.write_bytes(png)
    return path


def save_drawing(writer: str, kanji: str, png: bytes) -> Path:
    '''Guarda `png` como drawings/<clase>/<kanji>/<writer>-<NN>.png, sin pisar ningún archivo'''
    if writer not in WRITER_START:
        raise ValueError(f"Escritor desconocido: {writer}")
    folder = drawing_dir(kanji)
    folder.mkdir(parents=True, exist_ok=True)
    n = count_done(writer, kanji) + 1
    while (folder / f"{writer}-{n:02d}.png").exists():    # si borraron uno del medio, hay huecos
        n += 1
    path = folder / f"{writer}-{n:02d}.png"
    path.write_bytes(png)
    return path
