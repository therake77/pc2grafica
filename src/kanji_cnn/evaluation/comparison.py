from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from kanji_cnn.evaluation.metrics import accuracy, bootstrap_ci, f1_macro

#métricas de test que se comparan: clave en metrics.json → nombre legible
TEST_METRICS = {
    "acc": "accuracy",
    "precision_macro": "precision (macro)",
    "recall_macro": "recall (macro)",
    "f1_macro": "F1 (macro)",
}


@dataclass
class Run:
    '''Resultados de una corrida de `kanji-cnn` leídos de su carpeta'''
    name: str
    metrics: dict
    history: list[dict]
    y_true: np.ndarray
    y_pred: np.ndarray
    paths: np.ndarray      #rutas de las imágenes de test, relativas a drawings/, en el orden de y_true

    @classmethod
    def load(cls, run_dir: Path) -> Run:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        preds = np.load(run_dir / "predictions.npz")
        return cls(metrics.get("model", run_dir.name), metrics, history, preds["y_true"], preds["y_pred"], preds["paths"])

    @property
    def correct(self) -> np.ndarray:
        return self.y_true == self.y_pred

    def best_entry(self) -> dict | None:
        if not self.history:
            return None
        return min(self.history, key=lambda h: h["val_loss"])


def check_comparable(a: Run, b: Run) -> None:
    '''Las dos corridas deben haberse evaluado sobre el mismo test, en el mismo orden'''
    if a.metrics.get("seed") != b.metrics.get("seed"):
        raise SystemExit(f"Semillas distintas ({a.metrics.get('seed')} vs {b.metrics.get('seed')}): el test no es el mismo")
    if a.metrics["dataset"]["fingerprint"] != b.metrics["dataset"]["fingerprint"]:
        raise SystemExit("Las corridas se entrenaron con conjuntos de dibujos distintos: el test no es el mismo")
    if not np.array_equal(a.y_true, b.y_true):
        raise SystemExit("Las etiquetas de test no coinciden: las corridas no usan el mismo dataset/split")
    if list(a.metrics["report"])[: -3] != list(b.metrics["report"])[: -3]:
        raise SystemExit("Las clases de las dos corridas no coinciden")


def mcnemar(a: Run, b: Run) -> dict:
    '''Test de McNemar exacto: solo cuentan las muestras en las que un modelo acierta y el otro no.
    Bajo H0 (mismo error), cada discordancia cae de un lado u otro con probabilidad 1/2'''
    only_a = int((a.correct & ~b.correct).sum())
    only_b = int((~a.correct & b.correct).sum())
    both = int((a.correct & b.correct).sum())
    neither = int((~a.correct & ~b.correct).sum())
    n = only_a + only_b
    p_value = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither, "p_value": float(p_value)}


def per_class_f1(run: Run, n_classes: int) -> np.ndarray:
    return f1_score(run.y_true, run.y_pred, labels=list(range(n_classes)), average=None, zero_division=0)


def efficiency(run: Run) -> dict:
    best = run.best_entry()
    return {
        "n_params": run.metrics.get("n_params"),
        "size_mb": run.metrics.get("size_mb"),
        "mean_epoch_time_s": run.metrics.get("mean_epoch_time_s"),
        "inference_ms_per_img": run.metrics.get("inference_ms_per_img"),
        "epochs": len(run.history) or None,
        "best_epoch": run.metrics.get("best_epoch"),
        "lr": run.metrics.get("lr"),
        #train se mide con augmentación y dropout: la brecha es orientativa, no exacta
        "train_val_acc_gap": best["train_acc"] - best["val_acc"] if best else None,
        "best_val_acc": best["val_acc"] if best else None,
    }


def compare(a: Run, b: Run, class_names: list[str]) -> dict:
    '''Resumen completo de la comparación entre dos corridas sobre el mismo test'''
    check_comparable(a, b)
    ci = {
        run.name: {
            "acc": bootstrap_ci(run.y_true, run.y_pred, accuracy),
            "f1_macro": bootstrap_ci(run.y_true, run.y_pred, f1_macro),
        }
        for run in (a, b)
    }
    f1_a, f1_b = per_class_f1(a, len(class_names)), per_class_f1(b, len(class_names))
    return {
        "models": [a.name, b.name],
        "test_size": int(len(a.y_true)),
        "metrics": {run.name: {k: run.metrics[k] for k in (*TEST_METRICS, "loss")} for run in (a, b)},
        "ci95": ci,
        "mcnemar": mcnemar(a, b),
        "efficiency": {run.name: efficiency(run) for run in (a, b)},
        "per_class_f1": {
            name: {a.name: float(fa), b.name: float(fb)} for name, fa, fb in zip(class_names, f1_a, f1_b)
        },
    }


def _fmt(value, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def to_markdown(summary: dict) -> str:
    '''Tablas en Markdown, listas para pegar en el README'''
    a, b = summary["models"]
    m, ci, eff, mc = summary["metrics"], summary["ci95"], summary["efficiency"], summary["mcnemar"]

    lines = [f"| Métrica (test, n={summary['test_size']}) | {a} | {b} | Δ ({b} − {a}) |", "|---|---|---|---|"]
    for key, label in TEST_METRICS.items():
        cells = []
        for name in (a, b):
            cell = f"{m[name][key]:.4f}"
            if key in ci[name]:
                low, high = ci[name][key]
                cell += f" [{low:.3f}, {high:.3f}]"
            cells.append(cell)
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {m[b][key] - m[a][key]:+.4f} |")
    lines.append(f"| loss | {m[a]['loss']:.4f} | {m[b]['loss']:.4f} | {m[b]['loss'] - m[a]['loss']:+.4f} |")
    lines += ["", "Entre corchetes, intervalo de confianza del 95 % (bootstrap, 1000 remuestreos).", ""]

    lines += [
        f"McNemar: aciertan ambos {mc['both']}, solo {a} {mc['only_a']}, solo {b} {mc['only_b']}, "
        f"fallan ambos {mc['neither']} → p = {mc['p_value']:.2e}",
        "",
        f"| Coste | {a} | {b} |",
        "|---|---|---|",
    ]
    rows = [
        ("parámetros", "n_params", ","),
        ("tamaño (MB, float32)", "size_mb", ".1f"),
        ("s / época (train + val)", "mean_epoch_time_s", ".0f"),
        ("ms / imagen (inferencia)", "inference_ms_per_img", ".2f"),
        ("lr", "lr", "g"),
        ("mejor época", "best_epoch", "d"),
        ("val acc en la mejor época", "best_val_acc", ".4f"),
        ("brecha train − val acc", "train_val_acc_gap", "+.4f"),
    ]
    for label, key, spec in rows:
        lines.append(f"| {label} | {_fmt(eff[a][key], spec)} | {_fmt(eff[b][key], spec)} |")
    return "\n".join(lines) + "\n"
