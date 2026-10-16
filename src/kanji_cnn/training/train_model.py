from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Callable, Optional

from typing_extensions import Literal
from kanji_cnn.cnn.models import build_model
from kanji_cnn.dataset.augment import AugmentedTrain, augment_train
from kanji_cnn.dataset.drawings import DrawingsDataset
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

#70/15/15: primero se aparta test y del resto se aparta val
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_PER_KANJI = 3   #un kanji con menos dibujos no se puede repartir en tres conjuntos: se estratifica por clase
BATCH_SIZE = 32


def make_splits(
        dataset : DrawingsDataset,
        seed : int = SEED
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Divide el dataset en train/val/test (70/15/15) estratificado por kanji: cada kanji, y por tanto cada
    radical, aparece en los tres conjuntos en la misma proporción. Un mismo escritor dibuja cada kanji
    varias veces y esas repeticiones se reparten entre los conjuntos, así que test es optimista'''
    counts = Counter(dataset.kanjis)
    labels = np.array(dataset.class_names)[dataset.targets]
    strata = np.where([counts[k] >= MIN_PER_KANJI for k in dataset.kanjis], dataset.kanjis, labels)
    all_idx = np.arange(len(dataset))
    rest_idx, test_idx = train_test_split(all_idx, test_size=TEST_SIZE, stratify=strata, random_state=seed)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=strata[rest_idx], random_state=seed
    )
    #ordenados: test queda en el orden del manifiesto (clase, kanji, archivo)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def make_loaders(
        dataset : DrawingsDataset,
        splits : tuple[np.ndarray, np.ndarray, np.ndarray],
        train_set : AugmentedTrain
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''Construye los DataLoaders de train (el conjunto aumentado, con shuffle), val y test (dibujos
    originales, sin aumentar ni barajar)'''
    tr_idx, val_idx, test_idx = splits
    print(
        f"Split stratified by kanji: train={len(tr_idx)} (augmented to {len(train_set)}) "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=128)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=128)
    return train_loader, val_loader, test_loader


def make_loss() -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=0.1)


def evaluate(
        model : nn.Module,
        loader : DataLoader,
        device : Literal['cpu', 'cuda'] = DEVICE
    ) -> dict:
    '''Evalúa el modelo sobre `loader`. Devuelve loss, accuracy y las etiquetas reales/predichas'''
    loss_func = make_loss()
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += loss_func(out,yb).item()*len(yb)
            y_true.append(yb.cpu())
            y_pred.append(out.argmax(1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    n = len(y_true)
    return {
        "loss": total_loss / n,
        "acc": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def time_inference(
        model : nn.Module,
        loader : DataLoader,
        device : Literal['cpu', 'cuda'] = DEVICE,
        repeats : int = 5
    ) -> float:
    '''Milisegundos por imagen del forward del modelo, sobre un batch ya cargado en memoria
    (así no se mide la carga de datos, que es igual para cualquier modelo)'''
    xb, _ = next(iter(loader))
    xb = xb.to(device)
    model.eval()
    with torch.no_grad():
        model(xb)  #calentamiento
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / (repeats * len(xb)) * 1000


def train_model(
        model : nn.Module,
        train_loader : DataLoader,
        val_loader : DataLoader,
        device : Literal['cpu', 'cuda'] = DEVICE,
        epochs : int = 20,
        lr : float = 1e-3,
        on_epoch : Optional[Callable[[list[dict]], None]] = None
    ) -> list[dict]:
    '''Entrena el modelo y devuelve el historial por época. Al terminar, el modelo
    queda con los pesos de la época con menor val_loss. `on_epoch` recibe el
    historial tras cada época (p. ej. para guardarlo en disco)'''
    model.to(device)
    loss_func = make_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=3)

    history : list[dict] = []
    best_loss = float("inf")
    best_state = None
    print("Starting training")
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        train_loss = 0.0
        correct = total = 0
        for xb, yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_func(out,yb)
            loss.backward()
            optimizer.step()
            #métricas de train con dropout activo (y sobre imágenes aumentadas)
            train_loss += loss.item()*len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)

        val = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["loss"])
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / total,
            "train_acc": correct / total,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "lr": lr,
            "epoch_time": time.perf_counter() - start,
        })
        print(
            f"epoch {epoch}/{epochs}: train_loss={train_loss/total:.4f} train_acc={correct/total:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} lr={lr:.1e}"
        )

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
        if on_epoch is not None:
            on_epoch(history)

    model.load_state_dict(best_state)
    return history


if __name__ == "__main__":

    dataset = DrawingsDataset()
    splits = make_splits(dataset)
    train_loader, val_loader, _ = make_loaders(dataset, splits, augment_train(dataset, splits[0], SEED))
    model = build_model("cnn", dataset.num_classes).to(DEVICE)
    train_model(model, train_loader, val_loader)
