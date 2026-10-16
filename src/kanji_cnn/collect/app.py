from __future__ import annotations

import base64
from io import BytesIO

from flask import Flask, abort, redirect, request, url_for
from PIL import Image

from kanji_cnn.collect import storage
from kanji_cnn.collect.kanji_list import WRITER_START

app = Flask(__name__)

PNG_PREFIX = "data:image/png;base64,"   # así empieza lo que devuelve canvas.toDataURL()


def is_blank(png: bytes) -> bool:
    '''True si el canvas no tiene trazos. Como en letras_uni, el trazo sólo vive en el canal alfa'''
    return Image.open(BytesIO(png)).getchannel("A").getbbox() is None


@app.route("/")
def choose_writer():
    '''Página inicial: un enlace por escritor'''
    links = "".join(f'<li><a href="{url_for("draw", writer=w)}">{w}</a></li>' for w in WRITER_START)
    return f"<h1>¿Quién dibuja?</h1><ul>{links}</ul>"


@app.route("/<writer>")
def draw(writer: str):
    '''Lo que le toca dibujar a `writer` ahora'''
    if writer not in WRITER_START:
        abort(404)
    done, total = storage.progress(writer)
    target = storage.next_to_draw(writer)
    if target is None:
        return f"<h1>{writer}: terminaste tus {total} dibujos</h1>"
    cls, kanji = target
    return f"{writer}: dibuja {kanji} ({cls.name} {cls.radical}) · {done}/{total}"  # la pieza 4 pone aquí el canvas


@app.route("/<writer>/save", methods=["POST"])
def save(writer: str):
    '''Recibe el canvas como data URL, lo valida y lo guarda en drawings/'''
    data_url = request.form.get("image", "")
    if not data_url.startswith(PNG_PREFIX):
        abort(400)
    try:
        png = base64.b64decode(data_url.removeprefix(PNG_PREFIX))
        if not is_blank(png):                       # un canvas vacío no se guarda: vuelve al mismo kanji
            storage.save_drawing(writer, request.form.get("kanji", ""), png)
    except (KeyError, ValueError, OSError):         # kanji fuera de la tabla, escritor desconocido, PNG roto
        abort(400)
    return redirect(url_for("draw", writer=writer))


if __name__ == "__main__":
    app.run()
