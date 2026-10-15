from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import Subset

from kanji_cnn.dataset.etl_dataset import EtlVisualRadicalDataset
from kanji_cnn.evaluation import plots
from kanji_cnn.evaluation.comparison import TEST_METRICS, Run, compare, to_markdown
from kanji_cnn.evaluation.metrics import class_names
from kanji_cnn.preprocess import build_dataset
from kanji_cnn.training.train_model import make_splits

DEFAULT_OUT = Path("data/runs/compare")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compara dos corridas de kanji-cnn (p. ej. cnn y resnet50) sobre el mismo test"
    )
    parser.add_argument("run_a", type=Path, help="carpeta de la primera corrida (referencia, p. ej. la CNN)")
    parser.add_argument("run_b", type=Path, help="carpeta de la segunda corrida (p. ej. la ResNet50)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"carpeta de salida (default: {DEFAULT_OUT})")
    parser.add_argument("--show", action="store_true", help="además de guardar las figuras, las muestra")
    args = parser.parse_args(argv)

    for run_dir in (args.run_a, args.run_b):
        missing = [f for f in ("metrics.json", "predictions.npz") if not (run_dir / f).exists()]
        if missing:
            parser.error(f"a {run_dir} le faltan {', '.join(missing)} (¿es una corrida de kanji-cnn?)")

    a, b = Run.load(args.run_a), Run.load(args.run_b)
    if a.name == b.name:
        a.name, b.name = args.run_a.name, args.run_b.name
    class_labels = json.loads(build_dataset.CLASS_LABELS_PATH.read_text(encoding="utf-8"))
    summary = compare(a, b, class_names(class_labels))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table = to_markdown(summary)
    (args.out / "comparison.md").write_text(table, encoding="utf-8")

    cjk = plots.setup(args.show)
    names = class_names(class_labels, with_glyph=cjk)
    plots.plot_metrics_comparison(summary, TEST_METRICS, args.out / "metrics_comparison.png")
    if a.history and b.history:
        plots.plot_curves_comparison({a.name: a.history, b.name: b.history}, args.out / "curves_comparison.png")
    plots.plot_per_class_f1_diff(summary, names, args.out / "per_class_f1_diff.png")

    #el test se reconstruye con la semilla de las corridas: mismo orden que y_true/y_pred
    dataset = EtlVisualRadicalDataset()
    _, _, test_idx = make_splits(dataset, a.metrics["seed"])
    plots.plot_disagreements(
        Subset(dataset, test_idx), a.y_true, {a.name: a.y_pred, b.name: b.y_pred}, names,
        args.out / "disagreements.png",
    )

    print(table)
    print(f"Comparison saved to {args.out}")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
