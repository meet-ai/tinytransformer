
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST
from torch.nn import functional as F
import pytorch_lightning as pl



    
    
class LLMDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, y

