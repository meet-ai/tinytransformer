
import torch
from torch.utils.data import Dataset, DataLoader
from torch import utils
from torchvision import transforms
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST
from torch.nn import functional as F
import pytorch_lightning as pl


import sentencepiece as spm

# 加载预训练好的 SentencePiece 模型
tokenizer  = "tokenizer/tokenizer_cn_en.model"

# load
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(tokenizer)


    

class LLMDataset(Dataset):
    def __init__(self, cn_path, en_path, transform=None):
        with open(cn_path,'r') as f:
            cn_data = f.readlines()
        with open(en_path,'r') as f:
            en_data = f.readlines()
        cn_data = [chinese_sp_model.EncodeAsIds(line) for line in cn_data]
        en_data = [chinese_sp_model.EncodeAsIds(line) for line in en_data]
        self.data = cn_data
        self.labels = en_data
        print("data0:",self.data[0])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.labels[index]
        if self.transform:
            x = self.transform(x)
        
        return x, y


