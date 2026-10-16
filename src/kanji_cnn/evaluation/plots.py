from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from PIL import Image
from torch.utils.data import Dataset

from kanji_cnn.dataset.augment import MAX_PER_CLASS, MIN_PER_CLASS, TRANSFORM_LABELS, TRANSFORMS, AugmentedTrain
from kanji_cnn.dataset.drawings import DrawingsDataset, ink

CJK_FONT = "Noto Sans CJK JP"


def setup(show: bool) -> bool:
    '''Configura matplotlib. Sin `show` usa el backend Agg (solo archivos). Devuelve si hay
    una fuente CJK disponible; si no, las etiquetas deben ir sin glifo para evitar "tofu"'''
    if not show:
        matplotlib.use("Agg")
    try:
        font_manager.findfont(CJK_FONT, fallback_to_default=False)
    except ValueError:
        return False
    plt.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans"]
    return True


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _bars_height(n: int) -> float:
    '''Alto de una figura de barras horizontales con `n` barras'''
    return max(3.0, 0.24 * n + 1.2)


def kanji_label(kanji: str, cjk: bool) -> str:
    '''El kanji si hay fuente CJK; si no, su código Unicode (para no dibujar "tofu")'''
    return kanji if cjk else f"U+{ord(kanji):04X}"


#--- Dataset ---

def plot_dataset_counts(dataset: DrawingsDataset, path: Path, *, cjk: bool) -> plt.Figure:
    '''Dibujos por clase (apilados por escritor) y por kanji (coloreados por clase)'''
    writers = sorted(set(dataset.writers))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 3]})
    bottom = np.zeros(dataset.num_classes)
    for writer in writers:
        counts = np.bincount(dataset.targets[dataset.writers == writer], minlength=dataset.num_classes)
        axes[0].bar(dataset.class_names, counts, bottom=bottom, label=writer)
        bottom += counts
    axes[0].set(title="Dibujos por clase y escritor", ylabel="dibujos")
    axes[0].legend()

    kanjis = list(dict.fromkeys(dataset.kanjis.tolist()))
    counts = [int((dataset.kanjis == k).sum()) for k in kanjis]
    classes = [int(dataset.targets[np.argmax(dataset.kanjis == k)]) for k in kanjis]
    colors = plt.get_cmap("tab10")(classes)
    axes[1].bar(range(len(kanjis)), counts, color=colors)
    axes[1].set_xticks(range(len(kanjis)), [kanji_label(k, cjk) for k in kanjis], rotation=0 if cjk else 90)
    axes[1].set(title="Dibujos por kanji (color = clase)", ylabel="dibujos", xlim=(-0.7, len(kanjis) - 0.3))
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
    return fig


def plot_augmentation_examples(dataset: DrawingsDataset, path: Path, *, cjk: bool, seed: int = 0) -> plt.Figure:
    '''Un dibujo por clase: el PNG original (300×300), la entrada de la red (64×64) y una variante de
    cada transformación del aumento de datos, con sus parámetros debajo'''
    rng = np.random.default_rng(seed)
    cols = 2 + len(TRANSFORMS)
    fig, axes = plt.subplots(dataset.num_classes, cols, figsize=(cols * 1.75, dataset.num_classes * 1.95), squeeze=False)
    for c, row in enumerate(axes):
        i = int(np.argmax(dataset.targets == c))
        with Image.open(dataset.samples[i].path) as img:
            row[0].imshow(np.asarray(ink(img)), cmap="gray_r")
        image = dataset.images[i, 0].numpy()
        row[1].imshow(image, cmap="gray_r", vmin=0, vmax=1)
        for ax, transform in zip(row[2:], TRANSFORMS.values()):
            augmented, params = transform(image, rng)
            ax.imshow(augmented, cmap="gray_r", vmin=0, vmax=1)
            ax.set_xlabel(", ".join(f"{k}={v}" for k, v in params.items()), fontsize=7)
        row[0].set_ylabel(f"{dataset.class_names[c]}\n{kanji_label(dataset.kanjis[i], cjk)}", rotation=0, ha="right", va="center")
    titles = ["original", f"{dataset.size}×{dataset.size}", *(TRANSFORM_LABELS[name] for name in TRANSFORMS)]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Preprocesado y aumento de datos: 4 variantes por imagen")
    _save(fig, path)
    return fig


def plot_augmentation_counts(train_set: AugmentedTrain, path: Path) -> plt.Figure:
    '''Imágenes de train por clase tras el aumento, apiladas por origen (original o transformación)'''
    counts = train_set.counts()
    names = list(counts)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(names))
    for transform, label in TRANSFORM_LABELS.items():
        values = np.array([counts[name][transform] for name in names])
        ax.bar(names, values, bottom=bottom, label=label)
        bottom += values
    ax.axhline(MIN_PER_CLASS, color="black", linestyle="--", linewidth=1, label=f"mínimo ({MIN_PER_CLASS})")
    ax.axhline(MAX_PER_CLASS, color="black", linestyle=":", linewidth=1, label=f"máximo ({MAX_PER_CLASS})")
    ax.set(title=f"Train tras el aumento de datos: {train_set.target_per_class} imágenes por radical",
           ylabel="imágenes", ylim=(0, MAX_PER_CLASS * 1.08))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
    return fig


#--- Evaluación de un modelo ---


def plot_curves(history: list[dict], path: Path) -> plt.Figure:
    '''Curvas de loss y accuracy de train vs. val por época, marcando la mejor época'''
    epochs = [h["epoch"] for h in history]
    best = min(history, key=lambda h: h["val_loss"])["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, [h[f"train_{metric}"] for h in history], marker="o", label="train")
        ax.plot(epochs, [h[f"val_{metric}"] for h in history], marker="o", label="val")
        ax.axvline(best, color="gray", linestyle="--", label=f"mejor época ({best})")
        ax.set(title=title, xlabel="época")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)
    return fig


def plot_confusion_matrix(cm: np.ndarray, names: list[str], path: Path) -> plt.Figure:
    '''Matriz de confusión normalizada por fila: cada fila es la clase real. Con pocas clases,
    cada celda lleva su porcentaje'''
    n = len(names)
    side = min(14, 2.5 + 0.8 * n)
    fig, ax = plt.subplots(figsize=(side + 1, side))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ticks = np.arange(n)
    fontsize = 7 if n > 20 else 10
    ax.set_xticks(ticks, names, rotation=90 if n > 20 else 0, fontsize=fontsize)
    ax.set_yticks(ticks, names, fontsize=fontsize)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set(title="Matriz de confusión (test, normalizada por clase real)", xlabel="predicho", ylabel="real")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)
    return fig


def plot_per_class_accuracy(report: dict, names: list[str], report_keys: list[str], path: Path) -> plt.Figure:
    '''Accuracy (recall) por clase en test, ordenada de peor a mejor'''
    recalls = [report[k]["recall"] for k in report_keys]
    order = np.argsort(recalls)
    fig, ax = plt.subplots(figsize=(8, _bars_height(len(names))))
    ax.barh([names[i] for i in order], [recalls[i] for i in order], color="tab:blue")
    ax.axvline(report["accuracy"], color="tab:red", linestyle="--", label=f"accuracy global ({report['accuracy']:.3f})")
    ax.set(title="Accuracy por radical (test)", xlabel="accuracy", xlim=(0, 1))
    ax.margins(y=0.01)
    ax.legend(loc="lower right")
    _save(fig, path)
    return fig


def plot_per_kanji_accuracy(per_kanji: dict, names: list[str], path: Path, *, cjk: bool) -> plt.Figure:
    '''Accuracy de test de cada kanji, ordenada de peor a mejor y coloreada por clase.
    A la derecha de cada barra, cuántas muestras de test tiene ese kanji'''
    kanjis = sorted(per_kanji, key=lambda k: per_kanji[k]["acc"])
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(names)}
    fig, ax = plt.subplots(figsize=(8, _bars_height(len(kanjis))))
    accs = [per_kanji[k]["acc"] for k in kanjis]
    ax.barh([kanji_label(k, cjk) for k in kanjis], accs, color=[colors[per_kanji[k]["label"]] for k in kanjis])
    for y, k in enumerate(kanjis):
        ax.text(per_kanji[k]["acc"] + 0.01, y, f"n={per_kanji[k]['n']}", va="center", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, names, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set(title="Accuracy por kanji (test)", xlabel="accuracy", xlim=(0, 1.12))
    ax.margins(y=0.01)
    _save(fig, path)
    return fig


def plot_errors(
    dataset: Dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: list[str],
    path: Path,
    *,
    n: int = 32,
    seed: int = 0,
) -> plt.Figure | None:
    '''Rejilla de muestras de test mal clasificadas, con su clase real y la predicha.
    `dataset` debe estar en el mismo orden que `y_true`/`y_pred` (loader sin shuffle)'''
    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return None
    picked = np.random.default_rng(seed).choice(wrong, size=min(n, len(wrong)), replace=False)
    cols = 8
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, picked):
        image, _ = dataset[int(i)]
        ax.imshow(image[0], cmap="gray_r")
        ax.set_title(f"real {names[y_true[i]]}\npred {names[y_pred[i]]}", fontsize=8, color="tab:red")
    fig.suptitle("Errores en test")
    _save(fig, path)
    return fig


#--- Comparación entre dos modelos (kanji-compare) ---

COMPARE_COLORS = ("tab:blue", "tab:orange")


def plot_metrics_comparison(summary: dict, labels: dict[str, str], path: Path) -> plt.Figure:
    '''Barras agrupadas de las métricas de test de los dos modelos, con el IC 95 % donde lo hay'''
    models = summary["models"]
    keys = list(labels)
    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, name, color in zip((-width / 2, width / 2), models, COMPARE_COLORS):
        values = np.array([summary["metrics"][name][k] for k in keys])
        err = np.zeros((2, len(keys)))
        for j, k in enumerate(keys):
            if k in summary["ci95"][name]:
                low, high = summary["ci95"][name][k]
                err[:, j] = values[j] - low, high - values[j]
        bars = ax.bar(x + offset, values, width, yerr=err, capsize=3, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    low = min(summary["metrics"][n][k] for n in models for k in keys)
    ax.set_xticks(x, [labels[k] for k in keys])
    ax.set(title="Métricas en test", ylim=(max(0, low - 0.1), 1.02))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, path)
    return fig


def plot_curves_comparison(histories: dict[str, list[dict]], path: Path) -> plt.Figure:
    '''val_loss y val_acc de cada modelo por época y frente al tiempo de entrenamiento acumulado'''
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for (name, history), color in zip(histories.items(), COMPARE_COLORS):
        epochs = [h["epoch"] for h in history]
        hours = np.cumsum([h.get("epoch_time", np.nan) for h in history]) / 3600
        best = min(history, key=lambda h: h["val_loss"])
        axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", color=color, label=name)
        axes[1].plot(epochs, [h["val_acc"] for h in history], marker="o", color=color, label=name)
        axes[1].plot(epochs, [h["train_acc"] for h in history], linestyle=":", color=color, label=f"{name} (train)")
        axes[2].plot(hours, [h["val_acc"] for h in history], marker="o", color=color, label=name)
        axes[0].axvline(best["epoch"], color=color, linestyle="--", alpha=0.5)
    axes[0].set(title="Loss de validación", xlabel="época")
    axes[1].set(title="Accuracy (val continua, train punteada)", xlabel="época")
    axes[2].set(title="Accuracy de validación frente al tiempo", xlabel="horas de entrenamiento")
    for ax in axes[:2]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)
    return fig


def plot_per_class_f1_diff(summary: dict, names: list[str], path: Path) -> plt.Figure:
    '''Diferencia de F1 por radical (modelo B − modelo A), ordenada'''
    a, b = summary["models"]
    per_class = list(summary["per_class_f1"].values())
    diff = np.array([c[b] - c[a] for c in per_class])
    order = np.argsort(diff)
    fig, ax = plt.subplots(figsize=(8, _bars_height(len(names))))
    colors = [COMPARE_COLORS[1] if diff[i] > 0 else COMPARE_COLORS[0] for i in order]
    ax.barh([names[i] for i in order], diff[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title=f"F1 por radical: {b} − {a}\n(naranja: mejor {b}; azul: mejor {a})", xlabel="Δ F1")
    ax.margins(y=0.01)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path)
    return fig


def plot_disagreements(
    dataset: Dataset,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    names: list[str],
    path: Path,
    *,
    n: int = 16,
    seed: int = 0,
) -> plt.Figure:
    '''Dos rejillas de muestras de test en las que un modelo acierta y el otro falla'''
    (name_a, pred_a), (name_b, pred_b) = preds.items()
    groups = [
        (f"acierta {name_a}, falla {name_b}", np.flatnonzero((pred_a == y_true) & (pred_b != y_true)), pred_b, name_b),
        (f"acierta {name_b}, falla {name_a}", np.flatnonzero((pred_b == y_true) & (pred_a != y_true)), pred_a, name_a),
    ]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 1.6, 2 * rows * 2.1 + 0.8))
    subfigs = fig.subfigures(2, 1)
    rng = np.random.default_rng(seed)
    for subfig, (title, idx, wrong_pred, wrong_name) in zip(subfigs, groups):
        axes = subfig.subplots(rows, cols, squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        picked = rng.choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else []
        for ax, i in zip(axes.flat, picked):
            image, _ = dataset[int(i)]
            ax.imshow(image[0], cmap="gray_r")
            ax.set_title(f"real {names[y_true[i]]}\n{wrong_name}: {names[wrong_pred[i]]}", fontsize=7)
        subfig.suptitle(f"{title} ({len(idx)} muestras)")
    fig.savefig(path, dpi=150)
    return fig
