from __future__ import annotations
from typing import Any
from torch import nn


def conv_block(feat_in : int, feat_out : int ):
    layers = [
        nn.Conv2d(feat_in, feat_out, 3 , padding=1, bias=False),
        nn.BatchNorm2d(feat_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(feat_out, feat_out, 3 , padding=1, bias=False),
        nn.BatchNorm2d(feat_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    ]
    return nn.Sequential(*layers)




class KanjiRadicalCNN(nn.Module):

    __slots__ = ("feature", "dense")

    def __init__(self, n_classes = 5, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.feature = nn.Sequential(
            conv_block(1,32),
            conv_block(32,64),
            conv_block(64,128),
            conv_block(128,256)
        )

        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256,n_classes)
        )
    
    def forward(self,x):
        return self.dense(self.feature(x))