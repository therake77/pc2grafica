from typing_extensions import Literal
from kanji_cnn.cnn.radical_cnn import KanjiRadicalCNN
from kanji_cnn.dataset.etl_dataset import EtlVisualRadicalDataset, default_transform
import torch
from torch import nn
import torchvision.transforms as T
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 50

def train_model( 
        model : KanjiRadicalCNN, 
        device : Literal['cpu', 'cuda'] = DEVICE
    ):
    model.to(device)
    #preparamos transformaciones afines aleatorias al dataset
    aug = T.RandomAffine(degrees=8,translate=(0.08,0.08),scale=(0.9,1.1))
    train_dataset_full = EtlVisualRadicalDataset(transform=lambda x : aug(default_transform(x)))
    print("Train dataset loaded")
    inf_dataset_full = EtlVisualRadicalDataset()
    print("Inference dataset loaded")
    idx = np.arange(len(train_dataset_full))
    tr_idx , inf_idx = train_test_split(idx, test_size=0.2, stratify = train_dataset_full.classes, random_state=42)
    print("Split done")
    train_dataset = DataLoader(Subset(train_dataset_full,tr_idx), batch_size=64, shuffle=True)
    inf_dataset = DataLoader(Subset(inf_dataset_full,inf_idx), batch_size=128, shuffle=True)

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=3)
    print("Starting training")
    for epoch in range(50):
        model.train()
        for xb, yb in train_dataset:
            xb,yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_func(model(xb),yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        inf_loss = 0.0
        with torch.no_grad():
            for xb,yb in inf_dataset:
                xb,yb = xb.to(device), yb.to(device)
                out = model(xb)
                inf_loss += loss_func(out,yb).item()*len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)

        scheduler.step(inf_loss / total)
        print(f"epoch {epoch+1}: val_loss={inf_loss/total:.4f} val_acc={correct/total:.4f}")

if __name__ == "__main__":

    model = KanjiRadicalCNN(50).to(DEVICE)
    train_model(model)
    