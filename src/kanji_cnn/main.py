import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms


transfim = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),

])