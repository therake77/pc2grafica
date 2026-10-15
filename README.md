# kanji-cnn: clasificación del radical de kanji manuscritos

Red convolucional que, a partir de la imagen de un kanji escrito a mano, predice su **radical visual** (componente gráfico según la clasificación Kangxi de 214 radicales). Se entrena sobre la base de datos de caracteres manuscritos japoneses **ETL9B** y se compara con una **ResNet50 preentrenada en ImageNet** (ver [Comparación con ResNet50](#comparación-con-resnet50)).

## Flujo del proyecto

```
data/ETL9B/ETL9B_{1..5}            (imágenes binarias 64×63 + código JIS)
        │  preprocess/build_dataset.py
        ▼
data/build/sample_index.npy        (archivo, offset, escritor, clase de cada muestra)
data/build/class_labels.json       (clase → radical Kangxi y su glifo)
        │  training/train_model.py   (split por escritor 70/15/15)
        ▼
data/runs/<fecha-hora>-<modelo>/   (modelo, historial, métricas de test, figuras)
        │  compare.py                (kanji-compare: CNN vs ResNet50)
        ▼
data/runs/compare/                 (tabla, test de McNemar, figuras comparativas)
```

Todo el flujo se ejecuta con un solo comando: `uv run kanji-cnn` (ver [Uso](#uso)).

### 1. Construcción del dataset (`src/kanji_cnn/preprocess/`)

1. **Lectura de ETL9B** (`etl9b.py`): cada registro ocupa 576 bytes (id del escritor, código JIS y una imagen de 1 bit de 64×63). El índice solo guarda la posición de cada registro; las imágenes se leen bajo demanda.
2. **Filtro**: se descarta el hiragana. De los 3036 caracteres (610 236 registros) quedan 2965 kanji.
3. **Etiquetado por radical** (`dict_builder.py`, `radicals.py`): se descarga la descomposición IDS del proyecto [cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) (se cachea en `data/build/ids/ids.txt`), preferentemente su variante japonesa. Un kanji recibe una etiqueta si exactamente uno de sus componentes de primer nivel es un radical Kangxi o una de sus variantes (氵→水, 亻→人…). Los kanji ambiguos o sin coincidencia se excluyen, y quedan 2159 kanji etiquetados.
4. **Balanceo** (`build_dataset.py`): se agrupan las muestras por radical, se exigen ≥500 muestras por clase y se conservan las **50 clases** más numerosas, con un muestreo de **1000 muestras por clase**. Resultado: 50 000 muestras.

### 2. División train / val / test por escritor

Los 4000 escritores de ETL9B escribieron muchos caracteres cada uno. Si se divide por muestra, la red se evalúa con la letra de personas que ya vio al entrenar y la métrica sale optimista. Por eso el split (`make_splits` en `training/train_model.py`) usa `StratifiedGroupKFold` agrupando por `writer_id`:

| Conjunto | Muestras | Escritores | Uso |
|---|---|---|---|
| train | ~35 000 (70 %) | ~2800 | ajuste de pesos (con augmentación) |
| val | ~7 500 (15 %) | ~600 | `ReduceLROnPlateau` y elección de la mejor época |
| test | ~7 500 (15 %) | ~600 | métricas finales; no interviene en ninguna decisión |

Ningún escritor aparece en dos conjuntos y las 50 clases quedan balanceadas en cada uno. El split es determinista para una `--seed` dada (por defecto 42).

### 3. Modelo (`src/kanji_cnn/cnn/radical_cnn.py`)

`KanjiRadicalCNN` tiene 4 bloques `[Conv3×3 → BN → ReLU] ×2 → MaxPool` (32→64→128→256 canales), seguidos de *global average pooling* y de un MLP `256 → 256 → n_clases` con Dropout 0.4. La entrada es la imagen binaria de 1 canal, de 63×64.

### 4. Entrenamiento y evaluación

- **Entrenamiento**: Adam (lr 1e-3), cross-entropy con *label smoothing* 0.1 y `ReduceLROnPlateau` sobre `val_loss` (factor 0.5, paciencia 3). La augmentación es `RandomAffine` (±8°, traslación ±8 %, escala 0.9–1.1), solo en train.
- **Mejor época**: se conservan los pesos de la época con menor `val_loss`, no los de la última.
- **Métricas de test**: accuracy, top-5, precision/recall/F1 macro, reporte por clase y matriz de confusión. También se miden parámetros, tamaño, segundos por época y milisegundos de inferencia por imagen.

## Requisitos e instalación

- Python ≥ 3.14 y [uv](https://docs.astral.sh/uv/).
- El dataset **ETL9B**. No se distribuye con el repositorio: se solicita en la [ETL Character Database](http://etlcdb.db.aist.go.jp/) (AIST) y los archivos `ETL9B_1` … `ETL9B_5` se colocan en `data/ETL9B/`.
- La primera vez que se entrena la ResNet50 se descargan sus pesos de ImageNet (~100 MB, caché en `~/.cache/torch`).
- Para ver los glifos de los radicales en las gráficas hace falta la fuente **Noto Sans CJK JP**. Sin ella, las etiquetas muestran solo el número de radical Kangxi.

```bash
uv sync
```

## Uso

Todos los comandos se ejecutan **desde la raíz del repositorio**, porque las rutas `data/...` son relativas al directorio actual.

```bash
uv run kanji-cnn                         # flujo completo: dataset → entrenar la CNN (20 épocas) → evaluar en test
uv run kanji-cnn --model resnet50        # lo mismo con la ResNet50 preentrenada (lr 1e-4)
uv run kanji-cnn --prepare-only          # solo construye data/build/
uv run kanji-cnn --epochs 2              # prueba rápida
uv run kanji-cnn --run-dir data/runs/exp1 --seed 7
uv run kanji-cnn --eval-only data/runs/exp1/model.pt   # reevalúa un modelo en test sin entrenar
uv run kanji-cnn --show                  # además de guardar las figuras, las abre
```

| Opción | Descripción |
|---|---|
| `--rebuild` | Regenera `data/build/` aunque ya exista (si existe, por defecto se omite este paso) |
| `--prepare-only` | Solo prepara el dataset |
| `--model {cnn,resnet50}` | Arquitectura: `cnn` (`KanjiRadicalCNN`, por defecto) o `resnet50` (preentrenada en ImageNet) |
| `--epochs N` | Épocas de entrenamiento (por defecto 20) |
| `--lr LR` | Learning rate (por defecto 1e-3 para `cnn` y 1e-4 para `resnet50`) |
| `--device {cpu,cuda}` | Dispositivo (por defecto `cuda` si está disponible) |
| `--seed N` | Semilla del split train/val/test (por defecto 42) |
| `--run-dir DIR` | Carpeta de salida (por defecto `data/runs/<fecha-hora>-<modelo>`) |
| `--eval-only CKPT` | Evalúa un checkpoint en test; usa la semilla guardada en el checkpoint, así que el test es el mismo |
| `--show` | Muestra las figuras en pantalla |

## Salidas

Cada ejecución crea `data/runs/<fecha-hora>-<modelo>/` (o la carpeta de `--run-dir`) con:

| Archivo | Contenido |
|---|---|
| `model.pt` | Pesos de la mejor época, `model`, `class_labels`, `seed`, `best_epoch` y `lr` |
| `history.json` | Por época: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr` y `epoch_time` (s). Se reescribe en cada época, así que si la ejecución se interrumpe no se pierde el historial |
| `metrics.json` | Test: `acc`, `top5`, `precision_macro`, `recall_macro`, `f1_macro`, reporte por clase, tamaños del split y coste (`n_params`, `size_mb`, `mean_epoch_time_s`, `inference_ms_per_img`) |
| `predictions.npz` | `y_true` e `y_pred` de test, para `kanji-compare` |
| `curves.png` | Curvas de loss y accuracy de train vs. val, con la mejor época marcada |
| `confusion_matrix.png` | Matriz de confusión de test, normalizada por clase real |
| `per_class_accuracy.png` | Accuracy por radical, ordenada de peor a mejor |
| `errors.png` | Ejemplos de test mal clasificados (radical real vs. predicho) |

Para cargar el modelo en otro script:

```python
import torch
from kanji_cnn.cnn.models import build_model

ckpt = torch.load("data/runs/exp1/model.pt", map_location="cpu")
model = build_model(ckpt.get("model", "cnn"), ckpt["n_classes"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
# ckpt["class_labels"][i]["radical_glyph"] → glifo del radical de la clase i
```

## Resultados

Esta sección se completa tras un entrenamiento completo. Copia los valores de `metrics.json` (`acc`, `top5`, `f1_macro`) y el `best_epoch` de la ejecución, e indica el número de épocas y el dispositivo usado.

Las `train_acc`/`train_loss` se miden sobre imágenes aumentadas y con dropout activo, así que es normal que estén por debajo de las de validación en las primeras épocas.

## Rendimiento

En CPU (sin CUDA), una época de la CNN tarda unos 7–8 minutos, por lo que las 20 épocas por defecto suponen unas 2,5 horas. La ResNet50 tarda unos 14 minutos por época (~5 horas). Con GPU, usa `--device cuda` (es el valor por defecto si está disponible).

## Estructura del repositorio

```
src/kanji_cnn/
├── main.py                  # orquestación del flujo completo (CLI `kanji-cnn`)
├── compare.py               # comparación de dos corridas (CLI `kanji-compare`)
├── preprocess/
│   ├── etl9b.py             # lectura del formato binario ETL9B
│   ├── radicals.py          # los 214 radicales Kangxi y sus variantes
│   ├── dict_builder.py      # etiquetado kanji → radical vía IDS (cjkvi-ids)
│   ├── build_dataset.py     # filtrado, balanceo y escritura de data/build/
│   └── read_test.py         # visor rápido de registros ETL
├── dataset/etl_dataset.py   # Dataset de PyTorch sobre sample_index.npy
├── cnn/
│   ├── radical_cnn.py       # arquitectura KanjiRadicalCNN
│   ├── resnet.py            # ResNet50 de torchvision adaptada a 1 canal
│   └── models.py            # registro de modelos (--model) y lr por defecto
├── training/train_model.py  # split por escritor, bucle de entrenamiento y evaluación
├── evaluation/
│   ├── metrics.py           # accuracy, top-5, precision/recall/F1, IC bootstrap, matriz de confusión
│   ├── comparison.py        # McNemar, F1 por clase y tablas de la comparación
│   └── plots.py             # figuras con matplotlib
└── notebook/CNN_Kanji.ipynb # todo el proyecto, comparación incluida, en un notebook
```
