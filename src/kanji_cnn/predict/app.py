from __future__ import annotations

import base64

from flask import Flask, render_template, request

from kanji_cnn.collect.app import PNG_PREFIX, is_blank
from kanji_cnn.predict.classifier import RadicalClassifier

app = Flask(__name__)
classifier = RadicalClassifier()        # una sola vez, al arrancar: si falta model.pt, falla aquí y no en la primera petición


@app.route("/")
def index():
    '''Página con el lienzo y el panel de resultados'''
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    '''Recibe {"image": data URL del lienzo} y devuelve {"candidates": [...]}, de la clase más probable a la menos'''
    body = request.get_json(silent=True)                    # None si no llega JSON
    data_url = body.get("image") if isinstance(body, dict) else None
    if not isinstance(data_url, str) or not data_url.startswith(PNG_PREFIX):
        return {"error": 'se esperaba {"image": "data:image/png;base64,..."}'}, 400
    try:
        png = base64.b64decode(data_url.removeprefix(PNG_PREFIX))
        blank = is_blank(png)                               # abre el PNG: si no lo es, salta aquí
    except (ValueError, OSError):                           # base64 roto o bytes que no son un PNG
        return {"error": "la imagen no es un PNG válido"}, 400
    if blank:
        return {"error": "el lienzo está vacío"}, 400
    return {"candidates": classifier.classify(png)}


if __name__ == "__main__":
    app.run(port=5001)                  # la app de dibujo usa 5000: pueden correr a la vez
