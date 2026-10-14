import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
from PIL import Image

RECORD_SIZE = 576   #Formato ETL -> Cada registro tiene de tamaño 576
HEADER_SIZE = 576  #El primer registro del archivo ETL es uno vacío que debe ignorarse
IMAGE_SIZE = (64, 63)  # Dimensiones en bytes de la imagen en cada registro
IMAGE_BYTES = 504       #Longitud en bytes de la imagen

#Nombres de los archivos ETL, hardcodeados
DEFAULT_FILENAMES = ["ETL9B_1", "ETL9B_2", "ETL9B_3", "ETL9B_4", "ETL9B_5"]

#Forma rápida de obtener la dirección de los archivos dada la carpeta raíz de éstos
def get_etl9b_paths(root: str | Path = "data/ETL9B") -> list[Path]:
    root = Path(root)
    return [root / name for name in DEFAULT_FILENAMES]

#Utilidad para transformar el código JIS en un caracter imprimible
def jis_to_char(jis: int) -> str:
    return bytes([(jis >> 8) | 0x80, (jis & 0xFF) | 0x80]).decode("euc_jp")

#Clase que encapsula un registro ETL
@dataclass(frozen=True)
class EtlRecord:
    file_index: int #En cuál de los archivos ETL se encuentra, siguiendo el orden de `DEFAULT FILENAMES`
    byte_offset: int    # Cual es el offset en bytes del registro respecto al inicio del archivo
    writer_id: int  # ID del escritor del kanji
    jis_code: int   # Código JIS del caracter
    char: str   # Forma de caracter imprimible
    tag: str    # Forma romaji comprimida en 4 bytes
    image: Optional[Image.Image]    #Imagen asociada



def decode_record(
    chunk: bytes, file_index: int, byte_offset: int, *, with_image: bool = True
) -> EtlRecord:
    
    '''Dado un archivo ETL identificado por `file_index` y el offset en bytes, retorna el registro `EtlRecord` que le corresponde a dicha posición  '''
    
    writer_id, jis = struct.unpack(">HH", chunk[:4])
    char = jis_to_char(jis)
    tag = chunk[4:8].decode("ascii")
    image = (
        Image.frombytes("1", IMAGE_SIZE, chunk[8 : 8 + IMAGE_BYTES])
        if with_image
        else None
    )
    return EtlRecord(file_index, byte_offset, writer_id, jis, char, tag, image)


def iter_records(
    path: str | Path, file_index: int, *, with_image: bool = True
) -> Iterator[EtlRecord]:
    '''Dado una ruta a un archivo ETL, devuelve un iterable de registros `EtlRecord` del archivo ETL '''
    with open(path, "rb") as f:
        f.read(HEADER_SIZE)
        offset = HEADER_SIZE
        while True:
            chunk = f.read(RECORD_SIZE)
            if len(chunk) < RECORD_SIZE:
                break
            yield decode_record(chunk, file_index, offset, with_image=with_image)
            offset += RECORD_SIZE


def iter_all_records(
    paths: Sequence[str | Path], **kwargs
) -> Iterator[EtlRecord]:
    '''Dado un conjunto de archivos ETL, devuelve un iterable representando los registros `EtlRecord` de todos los archivos'''
    for file_index, path in enumerate(paths):
        yield from iter_records(path, file_index, **kwargs)


def read_record_at(paths: list[str] | list[Path], file_index: int, byte_offset: int) -> EtlRecord:
    '''Permite leer el registro indicado por `file_index` y `byte_offset` de la lista de direcciones `path`'''
    with open(paths[file_index], "rb") as f:
        f.seek(byte_offset)
        chunk = f.read(RECORD_SIZE)
    return decode_record(chunk, file_index, byte_offset, with_image=True)


def image_to_array(image: Image.Image) -> np.ndarray:
    '''Transforma una imagen en un ndarray'''
    return np.array(image, dtype=np.uint8)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    root_path = "data/ETL9B"
    file_path = get_etl9b_paths(root=root_path)[0]
    registers = iter_records(file_path,0,with_image=True)
    for i in range(80):
        next(registers)
    image = next(registers).image
    assert image is not None
    plt.imshow(image)
    plt.show()