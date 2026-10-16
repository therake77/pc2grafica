# Atajos del proyecto. Correr desde pc2grafica/: las rutas drawings/ y data/ son relativas a esta carpeta.
.PHONY: help draw-setup draw sync dataset train train-resnet compare predict notebooks

help:        ## lista los comandos
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ →/'

draw-setup:  ## entorno liviano para dibujar: sólo Flask y Pillow, sin torch
	test -d .venv || uv venv
	uv pip install flask pillow

draw: draw-setup  ## app de dibujo en http://127.0.0.1:5000
	PYTHONPATH=src .venv/bin/python -m kanji_cnn.collect.app

sync:        ## entorno completo para entrenar (torch con CUDA, ~3 GB)
	uv sync

dataset:     ## drawings/ → data/build/ (manifiestos con etiquetas, split y aumento de datos, clases y figuras)
	uv run kanji-cnn --prepare-only

train:       ## entrena y evalúa la CNN (20 épocas) → data/runs/cnn/
	uv run kanji-cnn --run-dir data/runs/cnn

train-resnet:  ## lo mismo con la ResNet50 preentrenada en ImageNet → data/runs/resnet50/
	uv run kanji-cnn --model resnet50 --run-dir data/runs/resnet50

compare:     ## CNN vs ResNet50 sobre el mismo test → data/runs/compare/
	uv run kanji-compare data/runs/cnn data/runs/resnet50

predict:     ## reconoce el radical de un dibujo en http://127.0.0.1:5001 (antes: make train)
	uv run python -m kanji_cnn.predict.app

notebooks:   ## regenera CNN_Kanji.ipynb y CNN_Kanji_Colab.ipynb
	uv run python src/kanji_cnn/notebook/build_notebooks.py
