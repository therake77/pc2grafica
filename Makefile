# Atajos del proyecto. Correr desde pc2grafica/: las rutas de datos son relativas a esta carpeta.
.PHONY: help draw-setup draw sync dataset train

help:        ## lista los comandos
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ →/'

draw-setup:  ## entorno liviano para dibujar: sólo Flask y Pillow, sin torch
	test -d .venv || uv venv
	uv pip install flask pillow

draw: draw-setup  ## app de dibujo en http://127.0.0.1:5000
	PYTHONPATH=src .venv/bin/python -m kanji_cnn.collect.app

sync:        ## entorno completo para entrenar (torch con CUDA, ~3 GB)
	uv sync

dataset:     ## ETL9B → data/build/ (una sola vez)
	uv run python -m kanji_cnn.preprocess.build_dataset

train:       ## entrena el modelo (50 épocas)
	uv run python -m kanji_cnn.training.train_model
