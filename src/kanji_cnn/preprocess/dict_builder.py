from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

from kanji_cnn.preprocess.radicals import build_glyph_to_radical_index

#Fuente del diccionario de kanjis - radicales. Obtenido del proyecto cjkvi-ids
IDS_URL = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"

#Regex que ayudará a obtener el IDC correcto (japonés) de las entradas del archivo ids.txt
_TAG_RE = re.compile(r"(\[[A-Za-z]+\])+$")

# Unicode Ideographic Description Characters. Describen cómo la imagen está segmentada 
# en distintos caracteres más pequeños. Las relacionamos con el número de caracteres 
# que representan
_IDC_ARITY = {
    "⿰": 2,
    "⿱": 2,
    "⿲": 3,
    "⿳": 3,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
}

#IDC permite relacionar un kanji con sus componentes. Sin embargo, 
# una entrada del diccionario kanji - radicales puede contener varios IDCs que describen
# al caracter de manera recursiva: Por ejemplo, ⿱⿲ significa: primero ⿱ y cada caracter que cae dentro
# de los espacios es un caracter de la forma ⿲. Así, es necesario recursión para poder leer los caracteres de manera jerárquica
class _Node:
    #Un nodo está definido por un caracter, y los caracteres pequeños que lo componen
    __slots__ = ("char", "children")

    def __init__(self, char: Optional[str], children: list["_Node"]):
        self.char = char
        self.children = children

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _parse_ids(s: str) -> _Node:
    '''Método que resuelve la cadena IDC para obtener los 
    caracteres que componen al kanji que representa de forma
    recursiva'''
    pos = 0
    #método recursivo local
    def parse() -> _Node:
        nonlocal pos
        ch = s[pos]
        pos += 1
        arity = _IDC_ARITY.get(ch)
        if arity is None:
            return _Node(ch, [])
        return _Node(None, [parse() for _ in range(arity)])

    return parse()


def fetch_ids_file(
    cache_path: Path = Path("data/build/ids/ids.txt"), *, force: bool = False
) -> Path:
    if cache_path.exists() and not force:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(IDS_URL, timeout=30) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _pick_alternative(alternatives: list[str]) -> str:
    """ETL9B is a Japanese handwriting dataset, so prefer a Japan-tagged
    ([...J...]) alternative when one exists; otherwise take the first
    listed alternative. Strips the trailing region-tag suffix either way."""
    for alt in alternatives:
        tag_match = _TAG_RE.search(alt)
        if tag_match and "J" in tag_match.group(0):
            return _TAG_RE.sub("", alt)
    return _TAG_RE.sub("", alternatives[0])


def parse_ids_file(path: Path) -> dict[str, str]:
    '''Retorna un diccionario de pares (`kanji_char`,`IDC_str`). Se prefiere tomar
    descomposiciones japonesas, pero en caso de que no estén indicadas explicitamente se toma 
    la primera'''
    ids_map: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            char = parts[1]
            ids_map[char] = _pick_alternative(parts[2:])
    return ids_map


def direct_radical_matches(
    char: str, ids_map: dict[str, str], glyph_to_radical: dict[str, int]
) -> set[int]:
    '''Dado un caracter, devuelve los radicales de primer nivel que lo componen.'''
    ids_string = ids_map.get(char)  #Primero obtén la cadena de caracteres IDC de la ya construida tabla caracter - cadena IDC
    if ids_string is None:  #Si no hay nada
        return set()
    try:
        #obtén la raíz. Los hijos representan posibles radicales
        root = _parse_ids(ids_string)
    except IndexError:
        return set()
    # Root es o bien un caracter sólo, o es un IDC. 
    # Si es el último, lo ignoramos y pasamos al siguiente nivel
    slots = [root] if root.is_leaf else root.children
    matches = set()
    for slot in slots:
        #Sólo si es una hoja y es un radical
        if slot.is_leaf and slot.char in glyph_to_radical:
            matches.add(glyph_to_radical[slot.char])
    return matches


def build_kanji_radical_map(
    kanji_chars: Iterable[str],
    ids_map: dict[str, str],
    glyph_to_radical: dict[str, int],
) -> dict[str, int]:
    '''De una lista de kanjis en formato caracter, da el único radical que le corresponde a cada uno en un diccionario. 
    Si es un kanji con varios radicales, es filtrado'''
    result: dict[str, int] = {}
    for char in kanji_chars:
        matches = direct_radical_matches(char, ids_map, glyph_to_radical)
        if len(matches) == 1: #Excluir kanjis con más de un radical (eso sería un caso de clasificación multiclase)
            result[char] = next(iter(matches))
    return result


def build_visual_radical_labels(
    kanji_chars: Iterable[str],
    *,
    ids_cache_path: Path = Path("data/build/ids/ids.txt"),
    force_refetch: bool = False,
) -> dict[str, int]:
    '''Dado una lista de kanjis como caracteres, devuelve los radicales que le corresponde a cada uno.
    Filtra kanjis que tienen radicales ambiguos'''
    ids_path = fetch_ids_file(ids_cache_path, force=force_refetch)
    ids_map = parse_ids_file(ids_path)
    glyph_to_radical = build_glyph_to_radical_index()
    return build_kanji_radical_map(kanji_chars, ids_map, glyph_to_radical)
