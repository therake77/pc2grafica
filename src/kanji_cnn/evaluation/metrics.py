from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def class_names(class_labels: list[dict], *, with_glyph: bool = True) -> list[str]:
    '''Nombre legible de cada clase: el de su carpeta y, si se puede dibujar, el glifo del radical'''
    if with_glyph:
        return [f"{c['name']} {c['radical_glyph']}".strip() for c in class_labels]
    return [c["name"] for c in class_labels]


def per_kanji_accuracy(y_true: np.ndarray, y_pred: np.ndarray, kanjis: np.ndarray, names: list[str]) -> dict:
    '''Accuracy de test de cada kanji, con su clase y cuántas muestras tiene. En el orden de `kanjis`'''
    result = {}
    for kanji in dict.fromkeys(kanjis.tolist()):
        mask = kanjis == kanji
        result[kanji] = {
            "label": names[int(y_true[mask][0])],
            "n": int(mask.sum()),
            "acc": float((y_true[mask] == y_pred[mask]).mean()),
        }
    return result


def test_metrics(result: dict, class_labels: list[dict]) -> tuple[dict, np.ndarray]:
    '''A partir del resultado de `evaluate` sobre test, calcula las métricas globales,
    el reporte por clase y la matriz de confusión normalizada por fila (clase real)'''
    y_true, y_pred = result["y_true"], result["y_pred"]
    labels = list(range(len(class_labels)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names(class_labels),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "loss": result["loss"],
        "acc": result["acc"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "report": report,
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return metrics, cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    '''Intervalo de confianza del 95 % de `metric` remuestreando el test con reemplazo'''
    rng = np.random.default_rng(seed)
    scores = [metric(y_true[idx], y_pred[idx]) for idx in rng.integers(0, len(y_true), size=(n, len(y_true)))]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)
