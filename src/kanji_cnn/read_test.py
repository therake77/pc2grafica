import struct
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy


archivos_ETL = [
    "data/ETL9B/ETL9B_1",
    "data/ETL9B/ETL9B_2",
    "data/ETL9B/ETL9B_3",
    "data/ETL9B/ETL9B_4",
    "data/ETL9B/ETL9B_5"
]


def open_etl_file(index : int):
    path = archivos_ETL[index]

    if os.path.isfile(path):
        print(os.path.getsize(path))
    else:
        raise Exception("FATAL not a file")
    with open(path,'rb') as f:
        

        f.read(576)
        images : list[Image.Image] = []
        kanji_char = []
        romaji = [] 
        while True:
            if len(images) >= 200:
                break
            chunk = f.read(576)
            if chunk == b'' or len(chunk) < 576:
                break
            else:
                writer_id, jis = struct.unpack('>HH', chunk[:4])
                kanji = bytes([(jis >> 8 )| 0x80 , (jis & 0xFF) | 0x80]).decode('euc_jp')
                label = chunk[4:8].decode('ascii')
                kanji_image = Image.frombytes('1', (64,63), chunk[8:512])
                kanji_char.append(kanji)
                romaji.append(label)
                images.append(kanji_image)
        return (images, kanji_char,romaji)
images, kanji, romaji = open_etl_file(0)

images, kanji, romaji = images[142:], kanji[142:], romaji[142:]


for i,img in enumerate(images):
    print(kanji[i])
    print(romaji[i])
    plt.imshow(img)
    plt.show()


