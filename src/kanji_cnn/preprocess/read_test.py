import matplotlib.pyplot as plt
from PIL import Image

from kanji_cnn.preprocess import etl9b


def open_etl_file(index: int):
    paths = etl9b.get_etl9b_paths()
    path = paths[index]
    print(path.stat().st_size)

    images: list[Image.Image] = []
    kanji_char = []
    romaji = []
    for record in etl9b.iter_records(path, index):
        if len(images) >= 200:
            break
        images.append(record.image)
        kanji_char.append(record.char)
        romaji.append(record.tag)
    return (images, kanji_char, romaji)


images, kanji, romaji = open_etl_file(0)

images, kanji, romaji = images[142:], kanji[142:], romaji[142:]


for i, img in enumerate(images):
    print(kanji[i])
    print(romaji[i])
    plt.imshow(img)
    plt.show()
