from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from kanji_cnn.cnn.models import DEFAULT_LR, MODEL_NAMES, build_model, count_params, size_mb
from kanji_cnn.dataset.drawings import DRAWINGS_DIR, DrawingsDataset, class_labels, write_manifest
from kanji_cnn.evaluation import plots
from kanji_cnn.evaluation.metrics import class_names, per_kanji_accuracy, test_metrics
from kanji_cnn.training.train_model import (
    AUGMENT, DEVICE, SEED, evaluate, make_loaders, make_splits, time_inference, train_model,
)

DEFAULT_RUNS_DIR = Path("data/runs")  #data/ está en .gitignore
BUILD_DIR = Path("data/build")
MANIFEST_PATH = BUILD_DIR / "drawings_index.csv"
CLASS_LABELS_PATH = BUILD_DIR / "class_labels.json"
DEFAULT_EPOCHS = 40

Splits = tuple[np.ndarray, np.ndarray, np.ndarray]


def write_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_dataset(drawings: Path, seed: int, *, cjk: bool) -> tuple[DrawingsDataset, Splits]:
    '''Escanea drawings/ (la etiqueta de cada dibujo es su carpeta de clase), preprocesa las imágenes,
    hace el split y guarda en data/build/ el manifiesto, las clases y las figuras del dataset.
    Se rehace en cada ejecución: los dibujos siguen llegando y es rápido'''
    print(f"Scanning {drawings}...")
    try:
        dataset = DrawingsDataset(drawings)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{e}\nEjecuta desde la raíz del repo o indica la carpeta con --drawings.")
    splits = make_splits(dataset, seed)
    summary = dataset.summary()
    print(
        f"  {summary['n_images']} drawings, {dataset.num_classes} classes, {summary['n_kanji']} kanji, "
        f"writers: {', '.join(f'{w} ({n})' for w, n in summary['writers'].items())}"
    )
    for c in class_labels(dataset):
        print(f"  {c['class_index']} {c['name']:<6} {c['radical_glyph']}  {c['count']:>4} drawings, {c['n_kanji']} kanji")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(dataset, splits, MANIFEST_PATH)
    write_json(class_labels(dataset), CLASS_LABELS_PATH)
    plots.plot_dataset_counts(dataset, BUILD_DIR / "figures" / "dataset_counts.png", cjk=cjk)
    plots.plot_preprocessing(dataset, AUGMENT, BUILD_DIR / "figures" / "preprocessing.png", cjk=cjk)
    print(f"Manifest saved to {MANIFEST_PATH}")
    return dataset, splits


def save_checkpoint(
    model: nn.Module,
    labels: list[dict],
    path: Path,
    *,
    model_name: str,
    seed: int,
    best_epoch: int,
    lr: float,
    mean_epoch_time_s: float,
    dataset: DrawingsDataset,
) -> None:
    '''Guarda pesos, etiquetas, la semilla del split y la huella del dataset, para que el checkpoint sea
    autocontenido en inferencia y se pueda reevaluar sobre el mismo test'''
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": model_name,
            "n_classes": len(labels),
            "class_labels": labels,
            "img_size": dataset.size,
            "seed": seed,
            "fingerprint": dataset.fingerprint,
            "best_epoch": best_epoch,
            "lr": lr,
            "mean_epoch_time_s": mean_epoch_time_s,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu")
    #pretrained=False: los pesos vienen del checkpoint, no hace falta descargar los de ImageNet
    model = build_model(ckpt["model"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def report_test(
    model: nn.Module,
    test_loader: DataLoader,
    dataset: DrawingsDataset,
    test_idx: np.ndarray,
    labels: list[dict],
    run_dir: Path,
    *,
    device: str,
    cjk: bool,
    extra: dict,
) -> dict:
    '''Evalúa en test, guarda metrics.json y predictions.npz (para `kanji-compare`) y genera
    las figuras de evaluación'''
    model.to(device)
    result = evaluate(model, test_loader, device)
    metrics, cm = test_metrics(result, labels)
    kanjis = dataset.kanjis[test_idx]
    efficiency = {
        "n_params": count_params(model),
        "size_mb": size_mb(model),
        "inference_ms_per_img": time_inference(model, test_loader, device),
        "device": device,
    }
    per_kanji = per_kanji_accuracy(result["y_true"], result["y_pred"], kanjis, class_names(labels))
    metrics = {**extra, **efficiency, **metrics, "per_kanji": per_kanji, "class_labels": labels}
    write_json(metrics, run_dir / "metrics.json")
    np.savez(
        run_dir / "predictions.npz",
        y_true=result["y_true"], y_pred=result["y_pred"], paths=dataset.paths[test_idx], kanjis=kanjis,
    )

    names = class_names(labels, with_glyph=cjk)
    plots.plot_confusion_matrix(cm, names, run_dir / "confusion_matrix.png")
    plots.plot_per_class_accuracy(
        metrics["report"], names, class_names(labels), run_dir / "per_class_accuracy.png"
    )
    plots.plot_per_kanji_accuracy(per_kanji, class_names(labels), run_dir / "per_kanji_accuracy.png", cjk=cjk)
    plots.plot_errors(
        test_loader.dataset, result["y_true"], result["y_pred"], names, run_dir / "errors.png"
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline de kanji_cnn: dibujos → entrenamiento → evaluación en test")
    parser.add_argument("--drawings", type=Path, default=DRAWINGS_DIR,
                        help=f"carpeta de dibujos <clase>/<kanji>/<escritor>-<NN>.png (default: {DRAWINGS_DIR})")
    parser.add_argument("--prepare-only", action="store_true", help="solo prepara el dataset (manifiesto, clases y figuras)")
    parser.add_argument("--model", choices=MODEL_NAMES, default="cnn",
                        help="cnn: KanjiRadicalCNN; resnet50: ResNet50 preentrenada en ImageNet (fine-tuning)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, help="learning rate (default: 1e-3 cnn, 1e-4 resnet50)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED, help="semilla del split train/val/test y de PyTorch")
    parser.add_argument("--run-dir", type=Path, help=f"carpeta de salida (default: {DEFAULT_RUNS_DIR}/<fecha-hora>-<modelo>)")
    parser.add_argument("--eval-only", type=Path, metavar="CKPT", help="evalúa un checkpoint en test sin entrenar")
    parser.add_argument("--show", action="store_true", help="además de guardar las figuras, las muestra")
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs debe ser >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda, pero CUDA no está disponible")
    if args.eval_only is not None and not args.eval_only.exists():
        parser.error(f"no existe el checkpoint {args.eval_only}")

    cjk = plots.setup(args.show)
    ckpt = None
    seed = args.seed
    if args.eval_only is not None:
        model, ckpt = load_checkpoint(args.eval_only)
        #se reusa la semilla del checkpoint para evaluar sobre el mismo test con el que se entrenó
        seed = ckpt["seed"]

    dataset, splits = prepare_dataset(args.drawings, seed, cjk=cjk)
    if args.prepare_only:
        return
    labels = class_labels(dataset)

    if ckpt is not None:
        if ckpt["fingerprint"] != dataset.fingerprint:
            raise SystemExit(
                f"{args.drawings} cambió desde que se entrenó {args.eval_only} (hay dibujos nuevos o borrados): "
                "el split ya no es el mismo y test podría incluir imágenes de train. Reentrena el modelo."
            )
        best_epoch, model_name, lr = ckpt["best_epoch"], ckpt["model"], ckpt["lr"]
        mean_epoch_time = ckpt["mean_epoch_time_s"]
        run_dir = args.run_dir or DEFAULT_RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{model_name}"
        print(f"Evaluating {model_name} checkpoint {args.eval_only} (--model is ignored)")
        train_loader, val_loader, test_loader = make_loaders(dataset, splits)
    else:
        model_name = args.model
        lr = args.lr if args.lr is not None else DEFAULT_LR[model_name]
        run_dir = args.run_dir or DEFAULT_RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{model_name}"
        torch.manual_seed(seed)     #inicialización, orden de los batches y augmentación reproducibles
        train_loader, val_loader, test_loader = make_loaders(dataset, splits)
        print(f"Training {model_name} on {dataset.num_classes} radical classes for {args.epochs} epochs on {args.device} (lr={lr:g})")
        model = build_model(model_name, dataset.num_classes, pretrained=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        history = train_model(
            model,
            train_loader,
            val_loader,
            device=args.device,
            epochs=args.epochs,
            lr=lr,
            on_epoch=lambda h: write_json(h, run_dir / "history.json"),
        )
        best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
        mean_epoch_time = float(np.mean([h["epoch_time"] for h in history]))
        save_checkpoint(
            model,
            labels,
            run_dir / "model.pt",
            model_name=model_name,
            seed=seed,
            best_epoch=best_epoch,
            lr=lr,
            mean_epoch_time_s=mean_epoch_time,
            dataset=dataset,
        )
        plots.plot_curves(history, run_dir / "curves.png")

    run_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = {name: len(idx) for name, idx in zip(("train", "val", "test"), splits)}
    metrics = report_test(
        model,
        test_loader,
        dataset,
        splits[2],
        labels,
        run_dir,
        device=args.device,
        cjk=cjk,
        extra={
            "model": model_name,
            "lr": lr,
            "seed": seed,
            "best_epoch": best_epoch,
            "mean_epoch_time_s": mean_epoch_time,
            "img_size": dataset.size,
            "split_sizes": split_sizes,
            "dataset": dataset.summary(),
        },
    )
    print(
        f"Test {model_name} ({split_sizes['test']} samples, best epoch {best_epoch}): "
        f"acc={metrics['acc']:.4f} precision_macro={metrics['precision_macro']:.4f} "
        f"recall_macro={metrics['recall_macro']:.4f} f1_macro={metrics['f1_macro']:.4f}"
    )
    print(f"Results saved to {run_dir}")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
