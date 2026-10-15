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
from kanji_cnn.evaluation import plots
from kanji_cnn.evaluation.metrics import class_names, test_metrics
from kanji_cnn.preprocess import build_dataset, etl9b
from kanji_cnn.training.train_model import DEVICE, SEED, evaluate, make_loaders, time_inference, train_model

DEFAULT_RUNS_DIR = Path("data/runs")  #data/ está en .gitignore
DEFAULT_EPOCHS = 20


def check_etl_files() -> None:
    '''Aborta si faltan los archivos ETL9B. Las rutas son relativas al cwd, así que hay que ejecutar desde la raíz del repo'''
    missing = [p for p in etl9b.get_etl9b_paths() if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise SystemExit(
            f"Faltan archivos ETL9B: {names}\n"
            "Ejecuta desde la raíz del repo (las rutas son relativas al cwd)."
        )


def prepare_dataset(rebuild: bool) -> None:
    '''Genera el índice de muestras y las etiquetas, salvo que ya existan (o `rebuild`)'''
    artifacts = [build_dataset.SAMPLE_INDEX_PATH, build_dataset.CLASS_LABELS_PATH]
    if not rebuild and all(p.exists() for p in artifacts):
        print("Dataset already prepared, skipping (use --rebuild to regenerate)")
        return
    print("Preparing dataset...")
    build_dataset.main()


def load_class_labels() -> list[dict]:
    return json.loads(build_dataset.CLASS_LABELS_PATH.read_text(encoding="utf-8"))


def write_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    model: nn.Module,
    class_labels: list[dict],
    path: Path,
    *,
    model_name: str,
    seed: int,
    best_epoch: int,
    lr: float,
    mean_epoch_time_s: float,
) -> None:
    '''Guarda pesos, etiquetas y la semilla del split, para que el checkpoint sea autocontenido
    en inferencia y se pueda reevaluar sobre el mismo test'''
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": model_name,
            "n_classes": len(class_labels),
            "class_labels": class_labels,
            "seed": seed,
            "best_epoch": best_epoch,
            "lr": lr,
            "mean_epoch_time_s": mean_epoch_time_s,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu")
    #los checkpoints anteriores a la comparación con ResNet50 no guardan "model": son la CNN
    #pretrained=False: los pesos vienen del checkpoint, no hace falta descargar los de ImageNet
    model = build_model(ckpt.get("model", "cnn"), ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def report_test(
    model: nn.Module,
    test_loader: DataLoader,
    class_labels: list[dict],
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
    metrics, cm = test_metrics(result, class_labels)
    efficiency = {
        "n_params": count_params(model),
        "size_mb": size_mb(model),
        "inference_ms_per_img": time_inference(model, test_loader, device),
        "device": device,
    }
    metrics = {**extra, **efficiency, **metrics}
    write_json(metrics, run_dir / "metrics.json")
    np.savez(run_dir / "predictions.npz", y_true=result["y_true"], y_pred=result["y_pred"])

    names = class_names(class_labels, with_glyph=cjk)
    plots.plot_confusion_matrix(cm, names, run_dir / "confusion_matrix.png")
    plots.plot_per_class_accuracy(
        metrics["report"], names, class_names(class_labels), run_dir / "per_class_accuracy.png"
    )
    plots.plot_errors(
        test_loader.dataset, result["y_true"], result["y_pred"], names, run_dir / "errors.png"
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline de kanji_cnn: dataset → entrenamiento → evaluación en test")
    parser.add_argument("--rebuild", action="store_true", help="regenera el índice aunque ya exista")
    parser.add_argument("--prepare-only", action="store_true", help="solo prepara el dataset")
    parser.add_argument("--model", choices=MODEL_NAMES, default="cnn",
                        help="cnn: KanjiRadicalCNN; resnet50: ResNet50 preentrenada en ImageNet (fine-tuning)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, help="learning rate (default: 1e-3 cnn, 1e-4 resnet50)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED, help="semilla del split train/val/test")
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

    check_etl_files()
    prepare_dataset(args.rebuild)
    if args.prepare_only:
        return

    run_dir = args.run_dir or DEFAULT_RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cjk = plots.setup(args.show)
    class_labels = load_class_labels()

    if args.eval_only is not None:
        model, ckpt = load_checkpoint(args.eval_only)
        if ckpt["class_labels"] != class_labels:
            raise SystemExit("Las clases del checkpoint no coinciden con data/build/class_labels.json")
        #se reusa la semilla del checkpoint para evaluar sobre el mismo test con el que se entrenó
        seed = ckpt.get("seed", SEED)
        best_epoch = ckpt.get("best_epoch")
        model_name, lr = ckpt.get("model", "cnn"), ckpt.get("lr")
        mean_epoch_time = ckpt.get("mean_epoch_time_s")
        print(f"Evaluating {model_name} checkpoint {args.eval_only} (--model is ignored)")
        train_loader, val_loader, test_loader = make_loaders(seed)
    else:
        seed, model_name = args.seed, args.model
        lr = args.lr if args.lr is not None else DEFAULT_LR[model_name]
        train_loader, val_loader, test_loader = make_loaders(seed)
        print(f"Training {model_name} on {len(class_labels)} radical classes for {args.epochs} epochs on {args.device} (lr={lr:g})")
        model = build_model(model_name, len(class_labels), pretrained=True)
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
            class_labels,
            run_dir / "model.pt",
            model_name=model_name,
            seed=seed,
            best_epoch=best_epoch,
            lr=lr,
            mean_epoch_time_s=mean_epoch_time,
        )
        plots.plot_curves(history, run_dir / "curves.png")

    split_sizes = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
        "test": len(test_loader.dataset),
    }
    metrics = report_test(
        model,
        test_loader,
        class_labels,
        run_dir,
        device=args.device,
        cjk=cjk,
        extra={
            "model": model_name,
            "lr": lr,
            "seed": seed,
            "best_epoch": best_epoch,
            "mean_epoch_time_s": mean_epoch_time,
            "split_sizes": split_sizes,
        },
    )
    print(
        f"Test {model_name} ({split_sizes['test']} samples, best epoch {best_epoch}): "
        f"acc={metrics['acc']:.4f} top5={metrics['top5']:.4f} precision_macro={metrics['precision_macro']:.4f} "
        f"recall_macro={metrics['recall_macro']:.4f} f1_macro={metrics['f1_macro']:.4f}"
    )
    print(f"Results saved to {run_dir}")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
