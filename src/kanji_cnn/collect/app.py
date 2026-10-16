from __future__ import annotations

import base64
from io import BytesIO

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from PIL import Image

from kanji_cnn.collect import storage
from kanji_cnn.collect.kanji_list import KANJI_TO_CLASS, WRITER_START

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
        return (f"<h1>{writer}: terminaste tus {total} dibujos</h1>"
                f'<p><a href="{url_for("edit_list", writer=writer)}">Retocar un dibujo ya guardado</a></p>')
    cls, kanji = target
    return render_template(
              "draw.html", writer=writer, cls=cls, kanji=kanji,
              kanji_done=storage.count_done(writer, kanji), done=done, total=total,
              strokes=f"strokes/{ord(kanji):05x}.svg",
          )



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


@app.route("/<writer>/editar")
def edit_list(writer: str):
    '''Galería con todos los dibujos de `writer`, para elegir cuál retocar'''
    if writer not in WRITER_START:
        abort(404)
    return render_template("edit_list.html", writer=writer, drawings=storage.list_drawings(writer))


@app.route("/<writer>/dibujos/<kanji>/<int:n>.png")
def drawing_png(writer: str, kanji: str, n: int):
    '''El PNG guardado, para mostrarlo en la galería y cargarlo en el lienzo. Sin caché: se sobreescribe'''
    try:
        path = storage.drawing_path(writer, kanji, n)
    except (KeyError, ValueError, FileNotFoundError):
        abort(404)
    response = send_file(path.resolve(), mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/<writer>/editar/<kanji>/<int:n>")
def edit(writer: str, kanji: str, n: int):
    '''Lienzo con el dibujo `n` de `kanji` ya cargado'''
    try:
        storage.drawing_path(writer, kanji, n)
    except (KeyError, ValueError, FileNotFoundError):
        abort(404)
    return render_template(
              "edit.html", writer=writer, kanji=kanji, n=n,
              cls=KANJI_TO_CLASS[kanji], strokes=f"strokes/{ord(kanji):05x}.svg",
          )


@app.route("/<writer>/editar/<kanji>/<int:n>", methods=["POST"])
def edit_save(writer: str, kanji: str, n: int):
    '''Sobreescribe el dibujo con lo que hay en el canvas'''
    data_url = request.form.get("image", "")
    if not data_url.startswith(PNG_PREFIX):
        abort(400)
    try:
        png = base64.b64decode(data_url.removeprefix(PNG_PREFIX))
        if is_blank(png):                           # un lienzo vacío dejaría un dibujo inservible: no se guarda
            abort(400)
        storage.replace_drawing(writer, kanji, n, png)
    except (KeyError, ValueError, OSError):         # kanji/escritor inválidos o PNG roto
        abort(400)
    return redirect(url_for("edit_list", writer=writer))


if __name__ == "__main__":
    app.run()
