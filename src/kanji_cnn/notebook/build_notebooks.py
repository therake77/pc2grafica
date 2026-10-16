'''Genera los dos notebooks del proyecto a partir de las mismas celdas:

- CNN_Kanji.ipynb: en local, lee drawings/ del repo y guarda todo en notebook/artifacts/
- CNN_Kanji_Colab.ipynb: en Google Colab, lee drawings/ de Google Drive y guarda allí los resultados

Solo cambian la introducción y la configuración; el resto del código es idéntico, así que los
notebooks no se desincronizan. Para regenerarlos: `make notebooks`.

Los notebooks son autocontenidos (en Colab no está instalado el paquete kanji_cnn) y su código debe
funcionar en el Python de Colab (3.12).
'''
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
COLAB_URL = "https://colab.research.google.com/github/therake77/pc2grafica/blob/main/src/kanji_cnn/notebook/CNN_Kanji_Colab.ipynb"


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}


# ---------------------------------------------------------------------------------------------
# Introducción y configuración (lo único que cambia entre local y Colab)
# ---------------------------------------------------------------------------------------------

CONTENTS = r'''
1. Configuración
2. El dataset: kanji dibujados por el equipo
3. Escaneo y etiquetado
4. Preprocesado
5. Dataset de PyTorch y augmentación
6. División train / val / test
7. Arquitectura de la CNN
8. Entrenamiento
9. Curvas de aprendizaje
10. Evaluación en test
11. Comparación con ResNet50 (ImageNet)
12. Inferencia
13. Conclusiones

**Objetivo.** Dada la imagen de un kanji dibujado a mano, predecir cuál de **5 radicales** contiene: 土 (`earth`), 火 (`fire`), 金 (`metal`), 木 (`tree`) u 氵 (`water`). Hay muchos más kanji (43) que clases: la red no tiene que reconocer el kanji, sino el radical que comparte con otros (池, 海 y 湖 son todos `water`).
'''

INTRO_LOCAL = r'''
# CNN para clasificar el radical de kanji dibujados a mano

Este notebook reúne **todo el proyecto `kanji_cnn`** en un único documento ejecutable de principio a fin, en local:
''' + CONTENTS + r'''
### Cómo ejecutarlo

1. Entorno: desde la raíz del repo, `uv sync` y luego, por ejemplo, `uv run --with jupyter jupyter lab`. Abre este archivo (`src/kanji_cnn/notebook/CNN_Kanji.ipynb`).
2. **Ejecuta todo.** El notebook busca la carpeta `drawings/` subiendo desde el directorio actual, así que funciona con el repo en cualquier ruta. Los dibujos nuevos de tus compañeros (los que llegan con `git pull`) se incluyen solos.

Todo lo que genera se guarda en `src/kanji_cnn/notebook/artifacts/` (ignorado por git), con la misma estructura que `data/` en el proyecto:

| ruta | contenido |
|---|---|
| `build/` | manifiesto `drawings_index.csv` (ruta, etiqueta, kanji, escritor y split de cada dibujo) y `class_labels.json` |
| `runs/cnn/` | `model.pt`, `history.json`, `metrics.json`, `predictions.npz` y figuras de la CNN |
| `runs/resnet50/` | lo mismo para la ResNet50 (sección 11) |
| `runs/compare/` | `comparison.json` y figuras de la comparación |
| `figures/` | figuras del dataset y de inferencia |

Las carpetas de `runs/` tienen el formato del proyecto, así que sirven para `kanji-cnn --eval-only` y `kanji-compare`.

Si ya hay un modelo entrenado **con los mismos dibujos**, se carga en vez de reentrenar (ver `RETRAIN` y `RETRAIN_RESNET`). Si los dibujos cambiaron, se reentrena solo.
'''

INTRO_COLAB = r'''
# CNN para clasificar el radical de kanji dibujados a mano — versión Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](''' + COLAB_URL + r''')

Este notebook reúne **todo el proyecto `kanji_cnn`** en un único documento ejecutable en Google Colab, de principio a fin:
''' + CONTENTS + r'''
### Cómo ejecutarlo en Colab

1. **GPU** (opcional): *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU* (T4 basta). El dataset es pequeño, así que en CPU también funciona: la CNN tarda unos segundos por época y la ResNet50 unos 10-20 s.
2. **Dibujos en Drive**: sube la carpeta `drawings/` del repo, entera, a `/content/drive/MyDrive/UNI/computacion-grafica/pc2/data` (en adelante, `data/`). Basta con arrastrarla en [drive.google.com](https://drive.google.com):

   ```
   MyDrive/UNI/computacion-grafica/pc2/data/
   └── drawings/
       ├── earth/地/diego-01.png …
       ├── fire/…
       ├── metal/…
       ├── tree/…
       └── water/…
   ```

   Cuando tus compañeros agreguen dibujos (con `make draw`), vuelve a subir la carpeta reemplazando la anterior.
3. **Ejecuta todo** (*Entorno de ejecución → Ejecutar todas*). La primera celda pide permiso para acceder a Google Drive.

Qué se guarda en Drive (la misma estructura que `data/` en el proyecto, así que las carpetas se pueden copiar al repo y usar con `kanji-cnn --eval-only` o `kanji-compare`):

| ruta en `data/` | contenido |
|---|---|
| `build/` | manifiesto `drawings_index.csv` (ruta, etiqueta, kanji, escritor y split de cada dibujo) y `class_labels.json` |
| `runs/cnn/` | `model.pt`, `history.json`, `metrics.json`, `predictions.npz` y figuras de la CNN |
| `runs/resnet50/` | lo mismo para la ResNet50 (sección 11) |
| `runs/compare/` | `comparison.json` y figuras de la comparación |
| `figures/` | figuras del dataset y de inferencia |

Notas:

- Al empezar, los dibujos se copian de Drive al disco local de la sesión (`/content/drawings`): leer cientos de archivos pequeños desde Drive es lento.
- Si ya hay en Drive un modelo entrenado **con los mismos dibujos**, se carga en vez de reentrenar (ver `RETRAIN` y `RETRAIN_RESNET`). Si los dibujos cambiaron, se reentrena solo.
- Tras cada época se guarda un punto de control (`last.pt`). Si Colab corta la sesión, al volver a ejecutar todo el entrenamiento **se reanuda** desde la última época completada.
- En GPU hay operaciones no deterministas (cuDNN): los resultados pueden variar ligeramente respecto de una ejecución en local.
'''

IMPORTS = r'''
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageOps
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet50_Weights, resnet50
'''

COLAB_SETUP = r'''
# Google Drive: de ahí se leen los dibujos y ahí se guarda todo lo que genera el notebook
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:   # fuera de Colab (p. ej. para probar el notebook en local)
    IN_COLAB = False

# Fuente CJK para dibujar los kanji en las gráficas (la primera vez tarda ~1 min)
if IN_COLAB and not list(Path("/usr/share/fonts").rglob("NotoSansCJK*")):
    subprocess.run("apt-get -qq update && apt-get -qq install -y fonts-noto-cjk", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
'''

PATHS_LOCAL = r'''
# --- Rutas ---
# drawings/ está en la raíz del repo: se busca subiendo desde la carpeta actual
REPO_ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "drawings").is_dir()), None)
if REPO_ROOT is None:
    raise FileNotFoundError("No se encontró la carpeta drawings/: abre el notebook desde dentro del repo.")
DRAWINGS_DIR = Path(os.environ.get("KANJI_DRAWINGS_DIR", REPO_ROOT / "drawings"))
# todo lo generado, con la misma estructura que data/ en el proyecto (ignorado por git)
ARTIFACT_DIR = Path(os.environ.get("KANJI_DATA_DIR", REPO_ROOT / "src" / "kanji_cnn" / "notebook" / "artifacts"))
'''

PATHS_COLAB = r'''
# --- Rutas: todo en Google Drive, con la misma estructura que data/ en el proyecto ---
ARTIFACT_DIR = Path(os.environ.get("KANJI_DATA_DIR", "/content/drive/MyDrive/UNI/computacion-grafica/pc2/data"))
DRIVE_DRAWINGS_DIR = ARTIFACT_DIR / "drawings"     # aquí se sube la carpeta drawings/ del repo
# Copia local de los dibujos: leer cientos de archivos pequeños desde Drive es lento
DRAWINGS_DIR = Path(os.environ.get("KANJI_LOCAL_DRAWINGS_DIR", "/content/drawings"))
'''

SETTINGS = r'''
BUILD_DIR = ARTIFACT_DIR / "build"                 # manifiesto (etiquetas y split) y clases
RUN_DIR = ARTIFACT_DIR / "runs" / "cnn"            # modelo, historial, métricas y figuras de la CNN
RESNET_DIR = ARTIFACT_DIR / "runs" / "resnet50"    # lo mismo para la ResNet50 (sección 11)
COMPARE_DIR = ARTIFACT_DIR / "runs" / "compare"    # resultados de la comparación (sección 11)
FIG_DIR = ARTIFACT_DIR / "figures"                 # figuras del dataset y de inferencia

# --- Dataset ---
IMG_SIZE = 64          # lado de la imagen que recibe la red
MARGIN = 0.08          # fracción del lado que queda libre a cada lado del kanji
TEST_SIZE = 0.15       # 70 % train, 15 % val, 15 % test
VAL_SIZE = 0.15
MIN_PER_KANJI = 3      # un kanji con menos dibujos no se puede repartir en tres conjuntos

# --- Entrenamiento ---
SPLIT_SEED = 42        # split train/val/test y semilla de PyTorch
EPOCHS = int(os.environ.get("KANJI_EPOCHS", 40))
BATCH_SIZE = 32
LR_CNN = 1e-3
RETRAIN = False        # True: reentrena aunque exista runs/cnn/model.pt con los mismos dibujos

# --- Comparación con ResNet50 (sección 11) ---
EPOCHS_RESNET = EPOCHS   # mismas épocas que la CNN
LR_RESNET = 1e-4         # fine-tuning: un lr alto destruiría los pesos de ImageNet
RETRAIN_RESNET = False   # True: reentrena aunque exista runs/resnet50/model.pt con los mismos dibujos
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''

CHECK_LOCAL = r'''
for directory in (BUILD_DIR, RUN_DIR, RESNET_DIR, COMPARE_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print(f"Dibujos en: {DRAWINGS_DIR}")
print(f"Resultados en: {ARTIFACT_DIR}")
print(f"Dispositivo: {DEVICE}")
'''

CHECK_COLAB = r'''
if not DRIVE_DRAWINGS_DIR.is_dir():
    raise FileNotFoundError(
        f"No existe {DRIVE_DRAWINGS_DIR}.\nSube la carpeta drawings/ del repo a {ARTIFACT_DIR} en Google Drive."
    )
# Copia nueva en cada ejecución: así se recogen los dibujos añadidos (y desaparecen los borrados)
if DRAWINGS_DIR.exists():
    shutil.rmtree(DRAWINGS_DIR)
shutil.copytree(DRIVE_DRAWINGS_DIR, DRAWINGS_DIR)
for directory in (BUILD_DIR, RUN_DIR, RESNET_DIR, COMPARE_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print(f"Dibujos copiados de {DRIVE_DRAWINGS_DIR} a {DRAWINGS_DIR}")
print(f"Resultados en: {ARTIFACT_DIR}")
print(f"Dispositivo: {DEVICE}" + ("" if DEVICE == "cuda" else
      "  (sin GPU; para activarla: Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU)"))
'''

FONT = r'''
# Fuente CJK para que matplotlib pueda dibujar los kanji en títulos y ejes.
# Sin ella, los kanji se muestran como su código Unicode y las clases sin glifo.
CJK_FONT = "Noto Sans CJK JP"
for font_path in Path("/usr/share/fonts").rglob("NotoSansCJK*"):
    font_manager.fontManager.addfont(str(font_path))   # recién instalada (Colab): hay que registrarla
try:
    font_manager.findfont(CJK_FONT, fallback_to_default=False)
    plt.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans"]
    HAS_CJK_FONT = True
except ValueError:
    HAS_CJK_FONT = False
    print(f"Aviso: no se encontró la fuente {CJK_FONT}; los kanji se muestran como U+XXXX.")


def kanji_label(kanji: str) -> str:
    return kanji if HAS_CJK_FONT else f"U+{ord(kanji):04X}"


def show_and_save(path: Path) -> None:
    """Guarda la figura actual en `path` y la muestra."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
'''


def intro_and_config(colab: bool) -> list[dict]:
    cells = [
        md(INTRO_COLAB if colab else INTRO_LOCAL),
        md("## 1. Configuración\n\n" + (
            "Imports, montaje de Google Drive, rutas e hiperparámetros. Todo lo ajustable está en la celda de configuración."
            if colab else
            "Imports, rutas e hiperparámetros. Todo lo ajustable está en la celda de configuración."
        )),
        code(IMPORTS),
    ]
    if colab:
        cells.append(code(COLAB_SETUP))
    cells.append(code((PATHS_COLAB if colab else PATHS_LOCAL) + SETTINGS + (CHECK_COLAB if colab else CHECK_LOCAL)))
    cells.append(code(FONT))
    return cells


# ---------------------------------------------------------------------------------------------
# Celdas comunes
# ---------------------------------------------------------------------------------------------

COMMON = [
md(r'''
## 2. El dataset: kanji dibujados por el equipo

Las imágenes no vienen de una base de datos: las dibujamos nosotros con la app Flask del proyecto (`make draw`, en http://127.0.0.1:5000). La app muestra a cada escritor el kanji que le toca, con el orden de trazos de referencia, y un lienzo de **300×300** con un pincel de 10 px. Cada escritor dibuja cada kanji **9 veces** (`water`, `tree`, `earth`) o **10** (`fire`, `metal`), y cada uno empieza por una clase distinta.

La app guarda cada dibujo como PNG en:

```
drawings/<clase>/<kanji>/<escritor>-<NN>.png        p. ej. drawings/fire/畑/diego-03.png
```

**La etiqueta de un dibujo es el nombre de su carpeta de clase.** Las clases son fijas:

| clase | radical | kanji |
|---|---|---|
| `earth` | 土 | 地 坂 均 坊 城 型 域 場 坪 |
| `fire` | 火 | 灯 炉 炊 炒 畑 炎 炭 灰 |
| `metal` | 金 | 針 釣 鈴 鉄 鉛 銀 銅 銭 |
| `tree` | 木 | 机 村 材 林 板 松 校 桜 杉 |
| `water` | 氵 | 池 河 油 泳 波 海 洋 流 湖 |

El radical se elige por **apariencia**: 畑 está en `fire` porque lleva 火 a la izquierda, aunque en el diccionario Kangxi se clasifique bajo 田. Tampoco está siempre en el mismo sitio: en 炎 hay dos 火 apilados, en 灰 y 型 el radical va abajo, y en 鈴 o 銀 金 se comprime a la izquierda.

Los dibujos siguen llegando, así que el notebook no asume ningún tamaño: cuenta lo que haya en la carpeta en cada ejecución.
'''),
md(r'''
## 3. Escaneo y etiquetado

Se recorre `drawings/`: las clases son las subcarpetas, ordenadas alfabéticamente (su índice es la etiqueta numérica), y de cada nombre de archivo se sacan el escritor y el número de repetición.
'''),
code(r'''
RADICAL_GLYPH = {"earth": "土", "fire": "火", "metal": "金", "tree": "木", "water": "氵"}


@dataclass(frozen=True)
class Drawing:
    path: Path
    label: str      # carpeta de primer nivel: drawings/<label>/<kanji>/<writer>-<NN>.png
    kanji: str
    writer: str
    number: int


def scan_drawings(root: Path) -> tuple[list[str], list[Drawing]]:
    """Recorre root/<clase>/<kanji>/<escritor>-<NN>.png. La etiqueta de cada dibujo es el nombre de su
    carpeta de clase; las clases se ordenan alfabéticamente y su índice es su posición en esa lista."""
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_names:
        raise ValueError(f"{root} no tiene carpetas de clase")
    samples = []
    for label in class_names:
        found = 0
        for path in sorted((root / label).glob("*/*.png")):
            writer, sep, number = path.stem.rpartition("-")
            if not sep or not writer or not number.isdigit():
                print(f"Aviso: se ignora {path}: el nombre no es <escritor>-<NN>.png")
                continue
            samples.append(Drawing(path, label, path.parent.name, writer, int(number)))
            found += 1
        if found == 0:
            raise ValueError(f"La clase {label} no tiene dibujos en {root / label}")
    return class_names, samples


CLASS_NAMES, samples = scan_drawings(DRAWINGS_DIR)
PLOT_NAMES = [f"{name} {RADICAL_GLYPH.get(name, '')}".strip() if HAS_CJK_FONT else name for name in CLASS_NAMES]
WRITERS = sorted({s.writer for s in samples})
print(f"{len(samples)} dibujos, {len(CLASS_NAMES)} clases, {len({s.kanji for s in samples})} kanji, "
      f"escritores: {', '.join(WRITERS)}\n")

counts = Counter((s.label, s.writer) for s in samples)
print(f"{'clase':<8}{'kanji':>6}" + "".join(f"{w:>8}" for w in WRITERS) + f"{'total':>8}")
for label in CLASS_NAMES:
    n_kanji = len({s.kanji for s in samples if s.label == label})
    row = [counts[label, w] for w in WRITERS]
    print(f"{label:<8}{n_kanji:>6}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}")
'''),
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 3]})
bottom = np.zeros(len(CLASS_NAMES))
for writer in WRITERS:
    row = np.array([counts[label, writer] for label in CLASS_NAMES])
    axes[0].bar(PLOT_NAMES, row, bottom=bottom, label=writer)
    bottom += row
axes[0].set(title="Dibujos por clase y escritor", ylabel="dibujos")
axes[0].legend()

kanji_counts = Counter(s.kanji for s in samples)
kanji_class = {s.kanji: CLASS_NAMES.index(s.label) for s in samples}
kanjis = list(kanji_counts)
axes[1].bar(range(len(kanjis)), [kanji_counts[k] for k in kanjis],
            color=plt.get_cmap("tab10")([kanji_class[k] for k in kanjis]))
axes[1].set_xticks(range(len(kanjis)), [kanji_label(k) for k in kanjis], rotation=0 if HAS_CJK_FONT else 90)
axes[1].set(title="Dibujos por kanji (color = clase)", ylabel="dibujos", xlim=(-0.7, len(kanjis) - 0.3))
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
show_and_save(FIG_DIR / "dataset_counts.png")
'''),
md(r'''
Mientras no terminen todos los escritores, las clases no están equilibradas: quien empezó por una clase tiene más dibujos de esa clase. Por eso, además de la accuracy, se reportan métricas *macro* (la media de las clases, todas con el mismo peso).

## 4. Preprocesado

Los dibujos no están centrados ni tienen el mismo tamaño dentro del lienzo de 300×300. Antes de entrar a la red, cada uno se normaliza así:

1. **Tinta.** El trazo solo vive en el **canal alfa** del PNG (el fondo es transparente), así que ese canal ya es una imagen en gris con tinta = 255 y fondo = 0.
2. **Recorte** a la caja que contiene el kanji.
3. **Cuadrado centrado** con un 8 % de margen por lado, sin deformar la proporción del kanji. Las posiciones relativas (el radical a la izquierda, abajo…) se conservan.
4. **Reducción a 64×64** con antialias, y valores en [0, 1].

El pincel de 10 px queda en unos 2 px, el grosor típico del trazo en bases de datos como ETL9B.
'''),
code(r'''
def ink(img: Image.Image) -> Image.Image:
    """Imagen en gris con la tinta en 255 y el fondo en 0. En los PNG de la app el trazo solo vive en el
    canal alfa; si la imagen no tiene alfa, se invierte el gris."""
    if "A" in img.getbands():
        return img.getchannel("A")
    return ImageOps.invert(img.convert("L"))


def preprocess(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Recorta el kanji a su caja, lo centra en un cuadrado con margen y lo reduce a size×size.
    Devuelve float32 en [0, 1] con tinta = 1."""
    gray = ink(img)
    bbox = gray.getbbox()
    if bbox is None:        # lienzo vacío
        return np.zeros((size, size), dtype=np.float32)
    crop = gray.crop(bbox)
    side = round(max(crop.size) * (1 + 2 * MARGIN))
    square = Image.new("L", (side, side), 0)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return np.asarray(square.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255


def load_image(path: Path, size: int = IMG_SIZE) -> torch.Tensor:
    """Lee y preprocesa un dibujo: tensor (1, size, size)."""
    with Image.open(path) as img:
        return torch.from_numpy(preprocess(img, size)).unsqueeze(0)


rng = np.random.default_rng(0)
fig, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(7, 1.8 * len(CLASS_NAMES)), squeeze=False)
for row, label in zip(axes, CLASS_NAMES):
    picked = rng.choice([s for s in samples if s.label == label], size=2, replace=False)
    for (ax_orig, ax_proc), s in zip((row[:2], row[2:]), picked):
        with Image.open(s.path) as img:
            ax_orig.imshow(np.asarray(ink(img)), cmap="gray_r")
            ax_proc.imshow(preprocess(img), cmap="gray_r", vmin=0, vmax=1)
        ax_orig.set_title(f"{kanji_label(s.kanji)} 300×300", fontsize=9)
        ax_proc.set_title(f"{IMG_SIZE}×{IMG_SIZE}", fontsize=9)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Dibujo original → entrada de la red")
plt.tight_layout()
show_and_save(FIG_DIR / "preprocessing.png")
'''),
md(r'''
## 5. Dataset de PyTorch y augmentación

`DrawingsDataset` lee y preprocesa **todos los dibujos al crearse** y los guarda en memoria (unos 26 MB con 1600 dibujos): así cada época no vuelve a abrir los PNG. Además guarda, para cada muestra, su kanji, su escritor y su ruta relativa a `drawings/`. Con las rutas se calcula una **huella** del conjunto de archivos: si cambia (porque llegaron dibujos nuevos), un modelo entrenado antes ya no se reutiliza.

Para el entrenamiento se aplica una **transformación afín aleatoria** (rotación ±10°, traslación ±10 %, escala 0.85–1.15 y cizalla ±8°), que simula variaciones naturales de la escritura y reduce el sobreajuste, importante con un dataset tan pequeño. **No se usan volteos horizontales**: cambiarían el lado del radical (氵 siempre va a la izquierda).
'''),
code(r'''
class DrawingsDataset(Dataset):
    """Dibujos preprocesados y en memoria: (N, 1, IMG_SIZE, IMG_SIZE). `transform` se aplica al tensor
    en cada __getitem__ (p. ej. la augmentación de train)."""

    def __init__(self, samples: list[Drawing], class_names: list[str], root: Path = DRAWINGS_DIR,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, size: int = IMG_SIZE):
        self.samples, self.class_names, self.size = samples, class_names, size
        self.images = torch.stack([load_image(s.path, size) for s in samples])
        self.targets = np.array([class_names.index(s.label) for s in samples])
        self.kanjis = np.array([s.kanji for s in samples])
        self.writers = np.array([s.writer for s in samples])
        self.paths = np.array([s.path.relative_to(root).as_posix() for s in samples])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        image = self.images[i]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[i])

    def with_transform(self, transform) -> "DrawingsDataset":
        """Copia que comparte las imágenes ya cargadas pero aplica otro `transform`."""
        other = copy.copy(self)
        other.transform = transform
        return other

    @property
    def fingerprint(self) -> str:
        """Huella del conjunto de archivos: cambia si se añaden o borran dibujos."""
        return hashlib.sha1("\n".join(sorted(self.paths)).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        writers, n = np.unique(self.writers, return_counts=True)
        return {"n_images": len(self), "n_kanji": len(set(self.kanjis)),
                "writers": {str(w): int(c) for w, c in zip(writers, n)}, "fingerprint": self.fingerprint}


dataset = DrawingsDataset(samples, CLASS_NAMES)
AUGMENT = T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8)
print(f"{len(dataset)} imágenes {tuple(dataset.images.shape[1:])}, huella {dataset.fingerprint[:12]}")
'''),
code(r'''
n_aug = 6
fig, axes = plt.subplots(len(CLASS_NAMES), n_aug + 1, figsize=(1.6 * (n_aug + 1), 1.7 * len(CLASS_NAMES)), squeeze=False)
with torch.random.fork_rng():   # la semilla de la figura no altera la del entrenamiento
    torch.manual_seed(0)
    for c, row in enumerate(axes):
        i = int(np.argmax(dataset.targets == c))
        row[0].imshow(dataset.images[i, 0], cmap="gray_r", vmin=0, vmax=1)
        row[0].set_ylabel(PLOT_NAMES[c], rotation=0, ha="right", va="center")
        for ax in row[1:]:
            ax.imshow(AUGMENT(dataset.images[i])[0], cmap="gray_r", vmin=0, vmax=1)
for ax, title in zip(axes[0], ["original", *["aumentada"] * n_aug]):
    ax.set_title(title, fontsize=9)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
show_and_save(FIG_DIR / "augmentation.png")
'''),
md(r'''
## 6. División train / val / test

70 % train, 15 % val y 15 % test, **estratificado por kanji**: cada kanji (y por tanto cada radical) aparece en los tres conjuntos en la misma proporción, así que test cubre los 43 kanji. Un kanji con menos de `MIN_PER_KANJI` dibujos, que no se puede repartir entre tres conjuntos, se estratifica solo por su clase.

- **train**: ajuste de pesos, con augmentación.
- **val**: elección de la mejor época y `ReduceLROnPlateau`.
- **test**: métricas finales; no interviene en ninguna decisión.

**Limitación.** Cada escritor dibuja cada kanji 9-10 veces seguidas, y esas repeticiones se reparten entre los tres conjuntos. Test mide, sobre todo, si la red reconoce **otra repetición** de un kanji que ya vio dibujado por la misma persona, lo que es más fácil que un kanji o un escritor nuevos. Las cifras de test son, por tanto, **optimistas**. El manifiesto guarda el escritor de cada dibujo para poder probar otro split (por ejemplo, un escritor entero en test) cuando haya más escritores.
'''),
code(r'''
def make_splits(seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 estratificado por kanji. Devuelve los índices de train, val y test, ordenados."""
    n_per_kanji = Counter(dataset.kanjis)
    labels = np.array(CLASS_NAMES)[dataset.targets]
    strata = np.where([n_per_kanji[k] >= MIN_PER_KANJI for k in dataset.kanjis], dataset.kanjis, labels)
    rest, test = train_test_split(np.arange(len(dataset)), test_size=TEST_SIZE, stratify=strata, random_state=seed)
    train, val = train_test_split(rest, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=strata[rest], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


splits = make_splits()
SPLIT_NAMES = ("train", "val", "test")
print(f"{'':<8}" + "".join(f"{name:>8}" for name in CLASS_NAMES) + f"{'total':>8}{'kanji':>7}")
for name, idx in zip(SPLIT_NAMES, splits):
    per_class = np.bincount(dataset.targets[idx], minlength=len(CLASS_NAMES))
    print(f"{name:<8}" + "".join(f"{n:>8}" for n in per_class) + f"{len(idx):>8}{len(set(dataset.kanjis[idx])):>7}")

# Manifiesto: una fila por dibujo con su etiqueta (la carpeta) y su conjunto
split_of = np.empty(len(dataset), dtype=object)
for name, idx in zip(SPLIT_NAMES, splits):
    split_of[idx] = name
with (BUILD_DIR / "drawings_index.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "class_index", "kanji", "writer", "number", "split"])
    for i, s in enumerate(samples):
        writer.writerow([dataset.paths[i], s.label, dataset.targets[i], s.kanji, s.writer, s.number, split_of[i]])

CLASS_LABELS = [{"class_index": i, "name": name, "radical_glyph": RADICAL_GLYPH.get(name, ""),
                 "n_kanji": len(set(dataset.kanjis[dataset.targets == i])), "count": int((dataset.targets == i).sum())}
                for i, name in enumerate(CLASS_NAMES)]
(BUILD_DIR / "class_labels.json").write_text(json.dumps(CLASS_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nManifiesto guardado en {BUILD_DIR / 'drawings_index.csv'}")
'''),
md(r'''
## 7. Arquitectura de la CNN

`KanjiRadicalCNN` apila **4 bloques convolucionales** que duplican los canales (32 → 64 → 128 → 256) y reducen la resolución a la mitad con *max pooling*. Cada bloque tiene esta forma:

```
[Conv 3×3 → BatchNorm → ReLU] × 2 → MaxPool 2×2
```

Después viene un **global average pooling**, que hace al modelo independiente de la posición exacta del trazo y reduce mucho los parámetros. Al final, un MLP `256 → 256 → 5` con Dropout 0.4. La entrada de 64×64 termina en un mapa de 4×4 antes del pooling global.
'''),
code(r'''
def conv_block(feat_in: int, feat_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(feat_in, feat_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(feat_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(feat_out, feat_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(feat_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KanjiRadicalCNN(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.feature = nn.Sequential(conv_block(1, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256))
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.dense(self.feature(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def size_mb(m: nn.Module) -> float:
    """Tamaño de los pesos en float32 (parámetros + buffers como las estadísticas de BatchNorm)."""
    return sum(t.numel() * t.element_size() for t in [*m.parameters(), *m.buffers()]) / 2**20


probe = KanjiRadicalCNN(len(CLASS_NAMES))
print(f"Parámetros: {count_params(probe):,}  ({size_mb(probe):.1f} MB)")
print("Mapa antes del pooling global:", tuple(probe.feature(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)).shape[1:]))
'''),
md(r'''
## 8. Entrenamiento

- **Pérdida**: entropía cruzada con *label smoothing* 0.1, que evita que la red se vuelva sobreconfiada.
- **Optimizador**: Adam con lr = 1e-3 (`LR_CNN`). `ReduceLROnPlateau` divide el lr entre 2 si `val_loss` no mejora en 3 épocas.
- **Mejor época**: se guardan los pesos con menor `val_loss` y se restauran al final, así que el modelo final no es necesariamente el de la última época.
- **Batches** de 32: con unos cientos de imágenes de train, un batch más grande daría muy pocas actualizaciones por época.

`fit_or_load` reutiliza `runs/cnn/model.pt` si se entrenó con los mismos dibujos (misma huella). El checkpoint tiene el mismo formato que los del proyecto (`kanji-cnn --eval-only`).
'''),
code(r'''
def make_loaders(splits) -> tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders de train (con augmentación), val y test (sin augmentación ni shuffle)."""
    train, val, test = splits
    return (DataLoader(Subset(dataset.with_transform(AUGMENT), train), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(Subset(dataset, val), batch_size=128),
            DataLoader(Subset(dataset, test), batch_size=128))


def make_loss() -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=0.1)


def evaluate(model: nn.Module, loader: DataLoader, device: str = DEVICE) -> dict:
    """Loss y accuracy sobre `loader`, más las etiquetas reales y predichas."""
    loss_func = make_loss()
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += loss_func(out, yb).item() * len(yb)
            y_true.append(yb.cpu())
            y_pred.append(out.argmax(1).cpu())
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return {"loss": total_loss / len(y_true), "acc": float((y_true == y_pred).mean()), "y_true": y_true, "y_pred": y_pred}


def time_inference(m: nn.Module, loader: DataLoader, repeats: int = 5) -> float:
    """ms por imagen del forward, sobre un batch ya en memoria."""
    xb = next(iter(loader))[0].to(DEVICE)
    m.eval()
    with torch.no_grad():
        m(xb)  # calentamiento
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            m(xb)
        if DEVICE == "cuda":
            torch.cuda.synchronize()   # la GPU es asíncrona: hay que esperar a que termine
    return (time.perf_counter() - start) / (repeats * len(xb)) * 1000


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, *, epochs: int, lr: float,
                on_epoch: Optional[Callable[[list[dict]], None]] = None, resume_path: Optional[Path] = None,
                device: str = DEVICE) -> list[dict]:
    """Entrena y restaura los pesos de la mejor época (menor val_loss). Si `resume_path` existe y es de
    estos mismos dibujos, continúa desde ese punto de control; tras cada época lo actualiza."""
    model.to(device)
    loss_func = make_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    history: list[dict] = []
    best_loss, best_state = float("inf"), None

    if resume_path is not None and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        if state.get("fingerprint") == dataset.fingerprint:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, best_loss, best_state = state["history"], state["best_loss"], state["best_state"]
            print(f"Reanudando desde la época {len(history)} ({resume_path})")

    for epoch in range(len(history) + 1, epochs + 1):
        start = time.perf_counter()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_func(out, yb)
            loss.backward()
            optimizer.step()
            # métricas de train sobre batches aumentados y con dropout activo
            train_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)

        val = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["loss"])
        history.append({"epoch": epoch, "train_loss": train_loss / total, "train_acc": correct / total,
                        "val_loss": val["loss"], "val_acc": val["acc"], "lr": lr,
                        "epoch_time": time.perf_counter() - start})
        print(f"época {epoch}/{epochs}: train_loss={train_loss / total:.4f} train_acc={correct / total:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} lr={lr:.1e}")

        if val["loss"] < best_loss:
            best_loss, best_state = val["loss"], copy.deepcopy(model.state_dict())
        if on_epoch is not None:
            on_epoch(history)
        if resume_path is not None:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "history": history, "best_loss": best_loss,
                        "best_state": best_state, "fingerprint": dataset.fingerprint}, resume_path)

    model.load_state_dict(best_state)
    return history


def checkpoint_matches(run_dir: Path) -> bool:
    """¿Hay un modelo en `run_dir` entrenado con estos mismos dibujos?"""
    path = run_dir / "model.pt"
    return path.exists() and torch.load(path, map_location="cpu").get("fingerprint") == dataset.fingerprint


def fit_or_load(model: nn.Module, name: str, run_dir: Path, *, epochs: int, lr: float, retrain: bool,
                retrain_flag: str):
    """Carga run_dir/model.pt si se entrenó con estos dibujos (y no se pide reentrenar); si no, entrena y
    lo guarda con el formato de los checkpoints del proyecto. Devuelve (historial, mejor época, lr)."""
    checkpoint, history_path, resume = run_dir / "model.pt", run_dir / "history.json", run_dir / "last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not retrain and checkpoint_matches(run_dir):
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"{name}: modelo cargado de {checkpoint} (mejor época {ckpt['best_epoch']}); "
              f"pon {retrain_flag} = True para reentrenar.")
        return history, ckpt["best_epoch"], ckpt["lr"]
    if checkpoint.exists() and not retrain:
        print(f"{name}: los dibujos cambiaron desde que se entrenó {checkpoint}; se reentrena.")

    print(f"Entrenando {name} {epochs} épocas en {DEVICE} (lr={lr:g})...")
    history = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, resume_path=resume,
                          on_epoch=lambda h: history_path.write_text(json.dumps(h, indent=2), encoding="utf-8"))
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    torch.save({"state_dict": model.state_dict(), "model": name, "n_classes": len(CLASS_NAMES),
                "class_labels": CLASS_LABELS, "img_size": IMG_SIZE, "seed": SPLIT_SEED,
                "fingerprint": dataset.fingerprint, "best_epoch": best_epoch, "lr": lr,
                "mean_epoch_time_s": float(np.mean([h["epoch_time"] for h in history]))}, checkpoint)
    resume.unlink(missing_ok=True)
    print(f"Mejor época: {best_epoch}. Modelo guardado en {checkpoint}")
    return history, best_epoch, lr
'''),
code(r'''
torch.manual_seed(SPLIT_SEED)   # inicialización, orden de los batches y augmentación reproducibles
train_loader, val_loader, test_loader = make_loaders(splits)
model = KanjiRadicalCNN(len(CLASS_NAMES))
history, best_epoch, cnn_lr = fit_or_load(model, "cnn", RUN_DIR, epochs=EPOCHS, lr=LR_CNN,
                                          retrain=RETRAIN, retrain_flag="RETRAIN")
model.to(DEVICE)
'''),
md(r'''
## 9. Curvas de aprendizaje

Al principio puede pasar que la accuracy de train quede **por debajo** de la de val: train se mide con imágenes aumentadas y con dropout activo, val no. Una val_loss que sube mientras la de train sigue bajando indica sobreajuste. La línea discontinua marca la época cuyos pesos se conservaron.
'''),
code(r'''
def plot_curves(hist: list[dict], path: Path, title: str) -> None:
    epochs_axis = [h["epoch"] for h in hist]
    best = min(hist, key=lambda h: h["val_loss"])["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric, name in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs_axis, [h[f"train_{metric}"] for h in hist], marker="o", label="train")
        ax.plot(epochs_axis, [h[f"val_{metric}"] for h in hist], marker="o", label="val")
        ax.set(title=name, xlabel="época")
    axes[2].plot(epochs_axis, [h["lr"] for h in hist], marker="o", color="tab:green")
    axes[2].set(title="Learning rate", xlabel="época", yscale="log")
    for ax in axes:
        ax.axvline(best, color="gray", linestyle="--", label=f"mejor época ({best})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    show_and_save(path)


plot_curves(history, RUN_DIR / "curves.png", "CNN")
'''),
md(r'''
## 10. Evaluación en test

El conjunto de test solo se usa aquí. Con 5 clases, top-5 no aporta nada (siempre es 1), así que se reportan:

- **accuracy**: proporción de radicales acertados.
- **precision** y **recall** macro: de las veces que el modelo predice un radical, cuántas acierta; y de las muestras de un radical, cuántas reconoce. Se promedian las 5 clases con el mismo peso.
- **F1 macro**: media del F1 de las 5 clases.
- **accuracy por kanji**: muestra qué kanji le cuestan más a la red, por ejemplo aquellos en los que el radical no está en su posición habitual.

`metrics.json` y `predictions.npz` se guardan con el formato de `kanji-cnn`, así que la comparación de la sección 11 también se puede hacer con `kanji-compare`.
'''),
code(r'''
labels = list(range(len(CLASS_NAMES)))
test_idx = splits[2]


def per_kanji_accuracy(y_true: np.ndarray, y_pred: np.ndarray, kanjis: np.ndarray) -> dict:
    """Accuracy de test de cada kanji, con su clase y su número de muestras."""
    result = {}
    for kanji in dict.fromkeys(kanjis.tolist()):
        mask = kanjis == kanji
        result[kanji] = {"label": CLASS_NAMES[int(y_true[mask][0])], "n": int(mask.sum()),
                         "acc": float((y_true[mask] == y_pred[mask]).mean())}
    return result


def save_test_results(m: nn.Module, result: dict, run_dir: Path, *, name: str, lr: float, best_epoch: int,
                      history: list[dict]) -> dict:
    """Escribe metrics.json y predictions.npz con el formato de `kanji-cnn`."""
    rep = classification_report(result["y_true"], result["y_pred"], labels=labels, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    metrics = {
        "model": name, "lr": lr, "seed": SPLIT_SEED, "best_epoch": best_epoch,
        "mean_epoch_time_s": float(np.mean([h["epoch_time"] for h in history])), "img_size": IMG_SIZE,
        "split_sizes": {n: len(idx) for n, idx in zip(SPLIT_NAMES, splits)}, "dataset": dataset.summary(),
        "n_params": count_params(m), "size_mb": size_mb(m), "inference_ms_per_img": time_inference(m, test_loader),
        "device": DEVICE, "loss": result["loss"], "acc": result["acc"],
        "precision_macro": rep["macro avg"]["precision"], "recall_macro": rep["macro avg"]["recall"],
        "f1_macro": rep["macro avg"]["f1-score"], "report": rep,
        "per_kanji": per_kanji_accuracy(result["y_true"], result["y_pred"], dataset.kanjis[test_idx]),
        "class_labels": CLASS_LABELS,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(run_dir / "predictions.npz", y_true=result["y_true"], y_pred=result["y_pred"],
             paths=dataset.paths[test_idx], kanjis=dataset.kanjis[test_idx])
    return metrics


test_result = evaluate(model, test_loader)
metrics = save_test_results(model, test_result, RUN_DIR, name="cnn", lr=cnn_lr, best_epoch=best_epoch, history=history)
print(f"Test ({len(test_idx)} imágenes, mejor época {best_epoch}):")
for key in ("acc", "precision_macro", "recall_macro", "f1_macro", "loss"):
    print(f"  {key:<16} {metrics[key]:.4f}")
print()
print(classification_report(test_result["y_true"], test_result["y_pred"], labels=labels, target_names=CLASS_NAMES,
                            digits=3, zero_division=0))
'''),
md(r'''
### Matriz de confusión

Normalizada por fila: la celda (i, j) es la fracción de muestras del radical *i* que se predijeron como *j*. Una diagonal intensa indica un buen modelo; las celdas fuera de ella muestran qué radicales se confunden.
'''),
code(r'''
def plot_confusion(result: dict, path: Path, title: str) -> None:
    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=labels, normalize="true")
    counts_cm = confusion_matrix(result["y_true"], result["y_pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, PLOT_NAMES)
    ax.set_yticks(labels, PLOT_NAMES)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]:.0%}\n({counts_cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set(title=title, xlabel="predicho", ylabel="real")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    show_and_save(path)


plot_confusion(test_result, RUN_DIR / "confusion_matrix.png", "CNN: matriz de confusión (test)")
'''),
md(r'''
### Accuracy por radical y por kanji

A la derecha de cada barra de kanji, cuántas muestras de test tiene: con pocos dibujos por kanji, un solo error mueve mucho su accuracy.
'''),
code(r'''
def plot_per_class_and_kanji(m: dict, path: Path, title: str) -> None:
    rep, per_kanji = m["report"], m["per_kanji"]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.24 * len(per_kanji) + 1.2)),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    recalls = np.array([rep[name]["recall"] for name in CLASS_NAMES])
    order = np.argsort(recalls)
    axes[0].barh([PLOT_NAMES[i] for i in order], recalls[order], color=plt.get_cmap("tab10")(order))
    axes[0].axvline(m["acc"], color="tab:red", linestyle="--", label=f"accuracy global ({m['acc']:.3f})")
    axes[0].set(title="Accuracy por radical", xlabel="accuracy", xlim=(0, 1))
    axes[0].legend(loc="lower right")

    kanjis = sorted(per_kanji, key=lambda k: per_kanji[k]["acc"])
    colors = [plt.get_cmap("tab10")(CLASS_NAMES.index(per_kanji[k]["label"])) for k in kanjis]
    axes[1].barh([kanji_label(k) for k in kanjis], [per_kanji[k]["acc"] for k in kanjis], color=colors)
    for y, k in enumerate(kanjis):
        axes[1].text(per_kanji[k]["acc"] + 0.01, y, f"n={per_kanji[k]['n']}", va="center", fontsize=7)
    axes[1].set(title="Accuracy por kanji (color = clase)", xlabel="accuracy", xlim=(0, 1.12))
    axes[1].margins(y=0.01)
    fig.suptitle(title)
    plt.tight_layout()
    show_and_save(path)


plot_per_class_and_kanji(metrics, RUN_DIR / "per_class_accuracy.png", "CNN (test)")
'''),
md(r'''
### Ejemplos mal clasificados
'''),
code(r'''
def plot_errors(result: dict, path: Path, title: str, n: int = 16) -> None:
    wrong = np.flatnonzero(result["y_true"] != result["y_pred"])
    if len(wrong) == 0:
        print("No hay errores en test.")
        return
    picked = np.random.default_rng(0).choice(wrong, size=min(n, len(wrong)), replace=False)
    cols = 8
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, picked):
        sample = int(test_idx[i])
        ax.imshow(dataset.images[sample, 0], cmap="gray_r")
        ax.set_title(f"{kanji_label(dataset.kanjis[sample])}: {PLOT_NAMES[result['y_true'][i]]}\n"
                     f"pred {PLOT_NAMES[result['y_pred'][i]]}", fontsize=8, color="tab:red")
    fig.suptitle(f"{title}: {len(wrong)} errores de {len(result['y_true'])}")
    plt.tight_layout()
    show_and_save(path)


plot_errors(test_result, RUN_DIR / "errors.png", "CNN")
'''),
md(r'''
## 11. Comparación con ResNet50 (ImageNet)

¿Merece la pena una red pequeña diseñada para este problema frente a una arquitectura estándar, mucho más grande y **preentrenada**? Con un dataset de unos cientos de imágenes, el *transfer learning* desde ImageNet es justamente lo que suele recomendarse. Para responderlo, se entrena una **ResNet50 con los pesos de ImageNet** (`IMAGENET1K_V2`) en las mismas condiciones y se comparan las dos sobre el mismo test.

**Lo que se mantiene igual:** el split (semilla 42), las 5 clases, la entrada de 1 canal a 64×64, la augmentación, la pérdida, el optimizador, el scheduler, el número de épocas y la elección de la mejor época por `val_loss`.

**Lo que se adapta:**

- **Entrada de 1 canal.** `conv1` espera RGB. Se sustituye por una convolución de 1 canal cuyos pesos son la **suma** de los de R, G y B: sobre una imagen gris replicada en los 3 canales da exactamente la misma respuesta, así que se conserva lo aprendido en ImageNet.
- **Normalización.** Dentro del modelo se normaliza con la media y la desviación del gris de ImageNet (0.449 y 0.226), así los datos llegan igual que a la CNN.
- **Cabeza.** La capa `fc` (1000 clases de ImageNet) se cambia por una de 5.
- **Learning rate 1e-4** (`LR_RESNET`), 10 veces menor: con 1e-3 los primeros pasos destruirían los pesos preentrenados.
'''),
code(r'''
class ResNet50Radical(nn.Module):
    """ResNet50 de torchvision para imágenes de 1 canal y `n_classes` radicales."""

    def __init__(self, n_classes: int = 5, pretrained: bool = False):
        super().__init__()
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        old = self.net.conv1
        self.net.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            self.net.conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        self.register_buffer("mean", torch.tensor(0.449))
        self.register_buffer("std", torch.tensor(0.226))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


probe = ResNet50Radical(len(CLASS_NAMES))
feature_map = nn.Sequential(*list(probe.net.children())[:-2])(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE))
print(f"Parámetros: {count_params(probe):,}  ({size_mb(probe):.1f} MB), "
      f"{count_params(probe) / count_params(model):.0f} veces los de la CNN")
print("Mapa antes del pooling global:", tuple(feature_map.shape[1:]))
'''),
md(r'''
Dos diferencias de diseño:

- **Resolución.** ResNet50 reduce la imagen de 64×64 a **16×16 en sus dos primeras operaciones** (convolución con *stride* 2 + maxpool) y termina en un mapa de **2×2**. La CNN aplica dos convoluciones 3×3 a resolución completa antes de reducir y termina en 4×4. Los trazos miden unos 2 píxeles, y la ResNet los resume muy pronto.
- **Tamaño.** ResNet50 tiene unas 19 veces más parámetros. Con tan pocos datos, eso favorece el sobreajuste, pero sus filtros ya vienen entrenados en ImageNet.
'''),
code(r'''
need_training = RETRAIN_RESNET or not checkpoint_matches(RESNET_DIR)
torch.manual_seed(SPLIT_SEED)
train_loader, val_loader, test_loader = make_loaders(splits)   # mismo orden de batches que la CNN
# Solo se descargan los pesos de ImageNet si hay que entrenar; si no, vienen del checkpoint
resnet = ResNet50Radical(len(CLASS_NAMES), pretrained=need_training)
resnet_history, resnet_best_epoch, resnet_lr = fit_or_load(
    resnet, "resnet50", RESNET_DIR, epochs=EPOCHS_RESNET, lr=LR_RESNET,
    retrain=RETRAIN_RESNET, retrain_flag="RETRAIN_RESNET")
resnet.to(DEVICE)
resnet_result = evaluate(resnet, test_loader)
resnet_metrics = save_test_results(resnet, resnet_result, RESNET_DIR, name="resnet50", lr=resnet_lr,
                                   best_epoch=resnet_best_epoch, history=resnet_history)
plot_curves(resnet_history, RESNET_DIR / "curves.png", "ResNet50")
plot_confusion(resnet_result, RESNET_DIR / "confusion_matrix.png", "ResNet50: matriz de confusión (test)")
plot_per_class_and_kanji(resnet_metrics, RESNET_DIR / "per_class_accuracy.png", "ResNet50 (test)")
plot_errors(resnet_result, RESNET_DIR / "errors.png", "ResNet50")
'''),
md(r'''
### 11.1 Curvas de aprendizaje comparadas

La gráfica de la derecha pone la accuracy de validación frente al **tiempo** de entrenamiento: muestra qué precisión se consigue con el mismo presupuesto de cómputo.
'''),
code(r'''
COLORS = {"CNN": "tab:blue", "ResNet50": "tab:orange"}
histories = {"CNN": history, "ResNet50": resnet_history}

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for name, hist in histories.items():
    color = COLORS[name]
    ep = [h["epoch"] for h in hist]
    minutes = np.cumsum([h["epoch_time"] for h in hist]) / 60
    axes[0].plot(ep, [h["val_loss"] for h in hist], marker="o", color=color, label=name)
    axes[0].axvline(min(hist, key=lambda h: h["val_loss"])["epoch"], color=color, linestyle="--", alpha=0.5)
    axes[1].plot(ep, [h["val_acc"] for h in hist], marker="o", color=color, label=name)
    axes[1].plot(ep, [h["train_acc"] for h in hist], linestyle=":", color=color, label=f"{name} (train)")
    axes[2].plot(minutes, [h["val_acc"] for h in hist], marker="o", color=color, label=name)
axes[0].set(title="Loss de validación", xlabel="época")
axes[1].set(title="Accuracy (val continua, train punteada)", xlabel="época")
axes[2].set(title="Accuracy de validación frente al tiempo", xlabel="minutos de entrenamiento")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
show_and_save(COMPARE_DIR / "curves_comparison.png")
'''),
md(r'''
### 11.2 Métricas en test

Se evalúan los dos modelos sobre las mismas imágenes de test. Con un test tan pequeño, cada imagen pesa mucho en la accuracy, así que los **intervalos de confianza del 95 %** son importantes: se calculan por *bootstrap* (se remuestrea el test con reemplazo 1000 veces y se toman los percentiles 2.5 y 97.5).
'''),
code(r'''
TEST_METRICS = {"acc": "accuracy", "precision_macro": "precision (macro)", "recall_macro": "recall (macro)",
                "f1_macro": "F1 (macro)"}
results = {"CNN": test_result, "ResNet50": resnet_result}
all_metrics = {"CNN": metrics, "ResNet50": resnet_metrics}


def bootstrap_ci(y_true, y_pred, metric, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = [metric(y_true[idx], y_pred[idx]) for idx in rng.integers(0, len(y_true), size=(n, len(y_true)))]
    return tuple(float(v) for v in np.percentile(scores, [2.5, 97.5]))


accuracy = lambda t, p: float((t == p).mean())
f1_macro = lambda t, p: float(f1_score(t, p, labels=labels, average="macro", zero_division=0))
ci = {name: {"acc": bootstrap_ci(r["y_true"], r["y_pred"], accuracy),
             "f1_macro": bootstrap_ci(r["y_true"], r["y_pred"], f1_macro)} for name, r in results.items()}

print(f"{'métrica (test, n=' + str(len(test_idx)) + ')':<24}{'CNN':>24}{'ResNet50':>24}{'Δ (ResNet − CNN)':>18}")
for key, label in TEST_METRICS.items():
    cells = []
    for name in ("CNN", "ResNet50"):
        cell = f"{all_metrics[name][key]:.4f}"
        if key in ci[name]:
            cell += " [{:.3f}, {:.3f}]".format(*ci[name][key])
        cells.append(cell)
    print(f"{label:<24}{cells[0]:>24}{cells[1]:>24}{resnet_metrics[key] - metrics[key]:>+18.4f}")

x = np.arange(len(TEST_METRICS))
fig, ax = plt.subplots(figsize=(10, 4.5))
for offset, name in zip((-0.19, 0.19), ("CNN", "ResNet50")):
    values = np.array([all_metrics[name][k] for k in TEST_METRICS])
    err = np.zeros((2, len(TEST_METRICS)))
    for j, k in enumerate(TEST_METRICS):
        if k in ci[name]:
            err[:, j] = values[j] - ci[name][k][0], ci[name][k][1] - values[j]
    bars = ax.bar(x + offset, values, 0.38, yerr=err, capsize=3, label=name, color=COLORS[name])
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
low = min(all_metrics[n][k] for n in all_metrics for k in TEST_METRICS)
ax.set_xticks(x, list(TEST_METRICS.values()))
ax.set(title="Métricas en test (barras de error: IC 95 % bootstrap)", ylim=(max(0, low - 0.15), 1.05))
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower right")
plt.tight_layout()
show_and_save(COMPARE_DIR / "metrics_comparison.png")
'''),
md(r'''
### 11.3 ¿La diferencia es significativa? Test de McNemar

Los dos modelos se evalúan sobre **las mismas** imágenes, así que la comparación adecuada es pareada. El test de McNemar ignora las muestras en las que ambos aciertan o ambos fallan y se fija solo en las **discordantes**: si los dos modelos fueran igual de buenos, cada discordancia caería de un lado o del otro con probabilidad 1/2 (test binomial exacto). Un p-valor alto significa que, con este test, no se puede afirmar que uno sea mejor que el otro.
'''),
code(r'''
cnn_ok = test_result["y_true"] == test_result["y_pred"]
res_ok = resnet_result["y_true"] == resnet_result["y_pred"]
only_cnn, only_res = int((cnn_ok & ~res_ok).sum()), int((~cnn_ok & res_ok).sum())
p_value = binomtest(only_cnn, only_cnn + only_res, 0.5).pvalue if only_cnn + only_res else 1.0

print(f"{'':<16} {'ResNet acierta':>15} {'ResNet falla':>13}")
print(f"{'CNN acierta':<16} {int((cnn_ok & res_ok).sum()):>15} {only_cnn:>13}")
print(f"{'CNN falla':<16} {only_res:>15} {int((~cnn_ok & ~res_ok).sum()):>13}")
print(f"\nDiscordantes: {only_cnn + only_res}  →  p = {p_value:.3g}")
'''),
md(r'''
### 11.4 F1 por radical y dónde discrepan

A la izquierda, en qué radicales gana cada modelo. Debajo, muestras de test que un modelo acierta y el otro falla, con la predicción del que falla.
'''),
code(r'''
f1_cnn = f1_score(test_result["y_true"], test_result["y_pred"], labels=labels, average=None, zero_division=0)
f1_res = f1_score(resnet_result["y_true"], resnet_result["y_pred"], labels=labels, average=None, zero_division=0)
diff = f1_res - f1_cnn
order = np.argsort(diff)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh([PLOT_NAMES[i] for i in order], diff[order],
        color=[COLORS["ResNet50"] if d > 0 else COLORS["CNN"] for d in diff[order]])
ax.axvline(0, color="black", linewidth=0.8)
ax.set(title="F1 por radical: ResNet50 − CNN\n(naranja: mejor ResNet50; azul: mejor CNN)", xlabel="Δ F1")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
show_and_save(COMPARE_DIR / "per_class_f1_diff.png")

y_true = test_result["y_true"]
groups = [("acierta la CNN, falla la ResNet50", np.flatnonzero(cnn_ok & ~res_ok), resnet_result["y_pred"], "ResNet"),
          ("acierta la ResNet50, falla la CNN", np.flatnonzero(~cnn_ok & res_ok), test_result["y_pred"], "CNN")]
rng = np.random.default_rng(0)
fig = plt.figure(figsize=(13, 5.5))
for subfig, (title, idx, wrong_pred, wrong_name) in zip(fig.subfigures(2, 1), groups):
    axes = subfig.subplots(1, 8)
    for ax in axes:
        ax.axis("off")
    picked = rng.choice(idx, size=min(8, len(idx)), replace=False) if len(idx) else []
    for ax, i in zip(axes, picked):
        sample = int(test_idx[i])
        ax.imshow(dataset.images[sample, 0], cmap="gray_r")
        ax.set_title(f"real {PLOT_NAMES[y_true[i]]}\n{wrong_name}: {PLOT_NAMES[wrong_pred[i]]}", fontsize=8)
    subfig.suptitle(f"{title} ({len(idx)} muestras)")
show_and_save(COMPARE_DIR / "disagreements.png")
'''),
md(r'''
### 11.5 Coste computacional

La inferencia se mide sobre un batch de test ya cargado en memoria.
'''),
code(r'''
cost = {}
for name, m in all_metrics.items():
    best = min(histories[name], key=lambda h: h["val_loss"])
    cost[name] = {"parámetros": f"{m['n_params']:,}", "tamaño (MB)": f"{m['size_mb']:.1f}",
                  "s / época": f"{m['mean_epoch_time_s']:.1f}", "ms / imagen": f"{m['inference_ms_per_img']:.2f}",
                  "mejor época": str(m["best_epoch"]), "val acc (mejor época)": f"{best['val_acc']:.4f}",
                  "brecha train − val acc": f"{best['train_acc'] - best['val_acc']:+.4f}"}
print(f"{'':<24}{'CNN':>14}{'ResNet50':>14}")
for key in cost["CNN"]:
    print(f"{key:<24}{cost['CNN'][key]:>14}{cost['ResNet50'][key]:>14}")
print(f"(dispositivo: {DEVICE})")

comparison = {
    "models": ["cnn", "resnet50"], "test_size": len(test_idx), "dataset": dataset.summary(),
    "metrics": {name.lower(): {k: all_metrics[name][k] for k in (*TEST_METRICS, "loss")} for name in all_metrics},
    "ci95": {name.lower(): ci[name] for name in ci},
    "mcnemar": {"both": int((cnn_ok & res_ok).sum()), "only_a": only_cnn, "only_b": only_res,
                "neither": int((~cnn_ok & ~res_ok).sum()), "p_value": float(p_value)},
    "per_class_f1": {name: {"cnn": float(a), "resnet50": float(b)} for name, a, b in zip(CLASS_NAMES, f1_cnn, f1_res)},
}
(COMPARE_DIR / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nComparación guardada en {COMPARE_DIR}")
'''),
md(r'''
## 12. Inferencia

`predict` recibe la imagen ya preprocesada y devuelve los radicales ordenados por probabilidad. `predict_file` hace lo mismo a partir de un PNG, por ejemplo uno recién dibujado con `make draw`.
'''),
code(r'''
def predict(m: nn.Module, image: torch.Tensor, k: int = 3) -> list[tuple[str, float]]:
    """Top-k radicales para una imagen (1, IMG_SIZE, IMG_SIZE)."""
    m.eval()
    with torch.no_grad():
        probs = m(image.unsqueeze(0).to(DEVICE)).softmax(1)[0].cpu()
    top = probs.topk(k)
    return [(CLASS_NAMES[c], p) for p, c in zip(top.values.tolist(), top.indices.tolist())]


def predict_file(m: nn.Module, path: Path, k: int = 3) -> list[tuple[str, float]]:
    """Top-k radicales para un PNG dibujado con la app (o cualquier imagen de un kanji en negro sobre blanco)."""
    return predict(m, load_image(Path(path)), k)


rng = np.random.default_rng(1)
picked = rng.choice(test_idx, size=min(12, len(test_idx)), replace=False)
fig, axes = plt.subplots(2, 6, figsize=(14, 5.5))
for ax in axes.flat:
    ax.axis("off")
for ax, i in zip(axes.flat, picked):
    top = predict(model, dataset.images[i])
    ok = top[0][0] == CLASS_NAMES[dataset.targets[i]]
    ax.imshow(dataset.images[i, 0], cmap="gray_r")
    ax.set_title(f"{kanji_label(dataset.kanjis[i])} → {top[0][0]} ({top[0][1]:.0%})\n"
                 f"2.º {top[1][0]} ({top[1][1]:.0%})", fontsize=9, color="tab:green" if ok else "tab:red")
fig.suptitle("Predicciones de la CNN sobre muestras de test (verde: acierto)")
plt.tight_layout()
show_and_save(FIG_DIR / "inference.png")

example = DRAWINGS_DIR / dataset.paths[test_idx[0]]
print(f"predict_file({example.relative_to(DRAWINGS_DIR)}):")
for name, p in predict_file(model, example):
    print(f"  {name:<6} {p:.1%}")
'''),
]


CONCLUSIONS = r'''
## 13. Conclusiones

Al ejecutar el notebook, las cifras salen en las secciones 10 y 11. Lo que hay que tener en cuenta para interpretarlas:

- **El test es optimista.** Con el split estratificado por kanji, test contiene otras repeticiones de kanji que la red vio dibujados por la misma persona. Mide bien si el modelo aprendió a reconocer esos 43 kanji, pero no garantiza que generalice a la letra de un escritor nuevo ni a kanji que no estén en el dataset. Cuando terminen de dibujar los cuatro escritores, el manifiesto (que guarda el escritor de cada dibujo) permite probar un split por escritor.
- **El test es pequeño.** Con 15 % de unos cientos de dibujos, cada imagen mueve la accuracy en más de un punto. Por eso se dan intervalos de confianza y el test de McNemar: una diferencia de una o dos imágenes entre la CNN y la ResNet50 no es significativa.
- **Dónde fallan los modelos.** La accuracy por kanji y los ejemplos mal clasificados muestran qué casos son difíciles: kanji en los que el radical no ocupa su posición habitual (炎, 灰, 型) o en los que se comprime (鈴, 銀).
- **CNN frente a ResNet50.** La ResNet50 tiene unas 19 veces más parámetros y tarda más por época, pero parte de pesos de ImageNet. La tabla de coste de la sección 11.5 permite decidir si la diferencia en test justifica ese precio.
- **Más datos.** El dataset sigue creciendo: basta con volver a ejecutar el notebook. Los modelos se reentrenan solos al detectar dibujos nuevos.
'''


def build(colab: bool) -> dict:
    cells = intro_and_config(colab) + COMMON + [md(CONCLUSIONS)]
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:02d}"
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12" if colab else "3.14"},
    }
    if colab:
        metadata |= {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    for name, colab in (("CNN_Kanji.ipynb", False), ("CNN_Kanji_Colab.ipynb", True)):
        path = HERE / name
        path.write_text(json.dumps(build(colab), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Notebook generado: {path}")


if __name__ == "__main__":
    main()
